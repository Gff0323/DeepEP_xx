# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026
# Modified by xixing 2026

"""BF16 autograd bridge and explicit lifetime for the shared SHMEM owner."""

from ctypes import c_float
import math

import torch
from torch.autograd.function import once_differentiable

from deep_ep._native import npu_api, require_native
from deep_ep._runtime import Runtime, integer


class _TrainRuntime(Runtime):
    _megamoe_train = True


class _MegaMoeBF16TrainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, op, x, ids, weights, w1, w2, max_recv, limit, mode, timer):
        op._check_ready()
        result = op._executor.forward(x, ids, weights, w1, w2, max_recv, limit, mode, timer)
        ctx.op = op
        ctx.max_recv, ctx.limit, ctx.mode = max_recv, limit, mode
        # The timer buffer is an output-only side channel, not a differentiable
        # input: keep it for backward and answer None for its gradient slot.
        ctx.timer = timer
        ctx.save_for_backward(weights, w1, w2, result.counts, result.saved_state, result.dispatch_output)
        ctx.mark_non_differentiable(result.counts)
        return result.y, result.counts

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y, grad_counts=None):
        weights, w1, w2, counts, saved, dispatch = ctx.saved_tensors
        op = ctx.op
        op._check_ready()
        result = op._executor.backward(
            grad_y.contiguous(),
            weights,
            w1,
            w2,
            counts,
            saved,
            dispatch,
            ctx.max_recv,
            ctx.limit,
            ctx.mode,
            ctx.timer,
        )
        return None, result.dx, None, result.dtopk_weights, result.dw1, result.dw2, None, None, None, None


class MegaMoeBF16Train(torch.nn.Module):
    """Fused Ascend950 BF16 forward/backward, derived from source b8cff148.

    Select the NPU before construction. Pass an existing EP ProcessGroup or
    StoreGroup. All ranks construct and close in the same order; DEEPEP_SHMEM_ENDPOINT names a separate bootstrap endpoint.
    This instance owns the root Session's symmetric window. There is one SHMEM
    owner per process, shared with DeepEP's exclusion checks.

    Calls use the construction stream and must be serialized in identical rank
    order. Separate forwards retain separate saved tensors. Close only after
    backward (or discard unused graphs). Higher-order derivatives, concurrent
    streams, and simultaneous inference/communication owners are unsupported.
    """

    def __init__(self, ep_group, local_mem_size, *, timeout=120):
        super().__init__()
        if ep_group is None:
            raise ValueError("Pass an existing EP ProcessGroup or StoreGroup")
        from . import _C

        self._npu = npu_api()
        self._stream = self._npu.current_stream()
        self._runtime = _TrainRuntime(ep_group, local_mem_size, timeout)
        # Root Session memset must finish on every peer before any generation signal.
        self._runtime.barrier()
        native = require_native("MegaMoeBF16Train")
        address = native.megamoe_train_workspace_address(self._runtime.native)
        like = torch.empty((), device=f"npu:{self._runtime.device}", dtype=torch.bfloat16)
        self._executor = _C.BF16(address, local_mem_size, self._runtime.rank, self._runtime.world_size, like)
        self._closed = False

    def _check_ready(self):
        if self._closed:
            raise RuntimeError("MegaMoeBF16Train has been closed")
        self._runtime.ensure_alive()
        if self._npu.current_stream() != self._stream:
            raise RuntimeError("MegaMoeBF16Train requires its construction stream")

    def forward(
        self,
        x,
        topk_ids,
        topk_weights,
        w1,
        w2,
        max_recv_tokens,
        *,
        swiglu_limit=0.0,
        save_gmm1_output="bf16",
        timer=None,
    ):
        """Return (y, counts); autograd provides dX, dTopK, dW1 and dW2.

        Floating inputs are contiguous BF16 tensors on the construction NPU;
        expert IDs are int32. Weights use [E,2F,H]/[E,H,F], with legacy layouts
        [E,H,2F]/[E,F,H] retained. Counts are non-differentiable.

        ``timer`` is an optional int64 buffer from :meth:`allocate_timer`. Both
        the forward and the autograd backward pass write into the same buffer;
        the kernel only records when a buffer is passed.
        """
        integer("max_recv_tokens", max_recv_tokens, 1, (1 << 31) - 1)
        limit = float(swiglu_limit)
        if not math.isfinite(limit) or not math.isfinite(c_float(limit).value):
            raise ValueError("swiglu_limit must be finite and representable as float32")
        if save_gmm1_output not in ("bf16", "none", "mxfp8", "auto"):
            raise ValueError("save_gmm1_output must be bf16, none, mxfp8 or auto")
        timer = self._check_timer(timer)
        return _MegaMoeBF16TrainFunction.apply(
            self, x, topk_ids, topk_weights, w1, w2, max_recv_tokens, limit, save_gmm1_output, timer
        )

    @property
    def timer_enabled(self):
        """True when the loaded device SO was built with the device timer."""
        from . import _C

        return bool(_C.timer_enabled())

    @property
    def timer_backward_enabled(self):
        """True when this build also instruments the backward kernel.

        Full timer builds instrument both directions. Forward-only builds are
        available with DEEPEP_MEGAMOE_TRAIN_TIMER_BACKWARD=OFF; a traced run
        reports that mode instead of waiting for a backward dump.
        """
        from . import _C

        return bool(_C.timer_backward_enabled())

    def allocate_timer(self, device=None):
        """Allocate one zeroed timer buffer for forward and backward tracing.

        A fresh (or re-zeroed) buffer per traced iteration keeps slots that
        this invocation never writes from being read as stale events.
        """
        from . import _C

        if not self.timer_enabled:
            raise RuntimeError(
                "the device kernel was built without DEEPEP_MEGAMOE_TRAIN_TIMER=ON; rebuild before tracing"
            )
        target = f"npu:{self._runtime.device}" if device is None else device
        return torch.zeros(_C.timer_buffer_size(), dtype=torch.int64, device=target)

    def _check_timer(self, timer):
        if timer is None:
            return None
        if not self.timer_enabled:
            raise RuntimeError(
                "the device kernel was built without DEEPEP_MEGAMOE_TRAIN_TIMER=ON; rebuild before tracing"
            )
        if not isinstance(timer, torch.Tensor) or timer.dtype != torch.int64:
            raise ValueError("timer must be an int64 tensor from allocate_timer")
        if timer.device != torch.device(f"npu:{self._runtime.device}"):
            raise ValueError("timer must live on the construction device")
        return timer

    def close(self):
        """Collectively release the window; callers must finish backward first."""
        if self._closed:
            return
        self._runtime.ensure_alive()
        self._stream.synchronize()
        self._runtime.close()
        self._executor.close()
        self._closed = True

"""BF16 autograd bridge and explicit lifetime for the shared SHMEM owner."""

from contextlib import contextmanager
from ctypes import c_float
import math
import threading
import warnings
import weakref

import torch
from torch.autograd.function import once_differentiable

from deep_ep._native import npu_api, require_native
from deep_ep._runtime import Runtime, integer


class _TrainRuntime(Runtime):
    _megamoe_train = True


class _PendingBackward:
    pass


def _saved_metadata(mode, limit):
    disabled = float.fromhex("0x1.fffffep+127")
    normalized = c_float(limit).value if limit > 0 else disabled
    resolved = ("mxfp8" if normalized == disabled else "bf16") if mode == "auto" else mode
    masks = resolved == "mxfp8" and normalized != disabled
    return {
        "layout_version": 2 if masks else 1,
        "clip_mask_kind": "three_bitplanes_v1" if masks else "none",
        "mode": resolved,
        "normalized_limit": normalized,
    }


class _MegaMoeBF16TrainFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, owner, x, ids, weights, w1, w2, max_recv, limit, mode, holder, lease):
        with owner._launch():
            y, counts, saved, dispatch, transient = owner._op.compute_train(
                x, ids, weights, w1, w2, max_recv, limit, mode
            )
            owner._record(x, ids, weights, w1, w2)
        ctx.owner = owner
        ctx.max_recv, ctx.limit, ctx.mode = max_recv, limit, mode
        ctx.holder, ctx.lease = holder, lease
        ctx.save_for_backward(weights, w1, w2, counts, saved, dispatch)
        ctx.mark_non_differentiable(counts)
        ctx.saved_gate_up_metadata = _saved_metadata(mode, limit)
        if holder is not None:
            holder.update(
                training_saved_state=saved,
                dispatch_output=dispatch,
                forward_transient_workspace=transient,
                saved_gate_up_metadata=ctx.saved_gate_up_metadata.copy(),
            )
        return y, counts

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_y, grad_counts=None):
        weights, w1, w2, counts, saved, dispatch = ctx.saved_tensors
        owner = ctx.owner
        with owner._launch():
            # Scratch may be captured by private diagnostics; it is never a forward saved tensor.
            dx, dp, dw1, dw2, scratch = owner._op.backward_debug(
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
            )
            owner._record(grad_y, weights, w1, w2, counts, saved, dispatch)
        if ctx.holder is not None and ctx.holder.get("capture_backward_workspace", False):
            ctx.holder["backward_workspace"] = scratch
        if ctx.lease is not None:
            owner._pending.discard(ctx.lease)
        return None, dx, None, dp, dw1, dw2, None, None, None, None, None


class MegaMoeBF16Train(torch.nn.Module):
    """Fused Ascend950 BF16 forward/backward, derived from source b8cff148.

    Select the NPU before construction. Pass an EP ProcessGroup/StoreGroup, or
    group=None with explicit rank/world_size. All ranks construct and close in
    the same order; DEEPEP_SHMEM_ENDPOINT names a separate bootstrap endpoint.
    This instance owns the root Session's symmetric window. There is one SHMEM
    owner per process, shared with DeepEP's exclusion checks.

    Calls use the construction stream and must be serialized in identical rank
    order. Separate forwards retain separate saved tensors. Close only after
    backward (or discard unused graphs). Higher-order derivatives, concurrent
    streams, and simultaneous inference/communication owners are unsupported.
    """

    def __init__(self, group, local_mem_size, *, rank=None, world_size=None, timeout=120):
        super().__init__()
        from . import _C

        if _C.abi_version != 1:
            raise ImportError("Incompatible MegaMoE training extension; reinstall one complete wheel")
        self._closed = True
        self._lock = threading.RLock()
        self._pending = weakref.WeakSet()
        self._npu = npu_api()
        self._stream = self._npu.current_stream()
        self._runtime = _TrainRuntime(group, local_mem_size, timeout, rank=rank, world_size=world_size)
        # Root Session memset must finish on every peer before any generation signal.
        self._runtime.barrier()
        native = require_native("MegaMoeBF16Train")
        address = native.megamoe_train_workspace_address(self._runtime.native)
        like = torch.empty((), device=f"npu:{self._runtime.device}", dtype=torch.bfloat16)
        try:
            self._op = _C.BF16(address, local_mem_size, self._runtime.rank, self._runtime.world_size, like)
        except Exception:
            # No collective cleanup during a rank-local initialization failure.
            # The job must terminate together; never finalize a foreign Session.
            warnings.warn("Training initialization failed; restart the distributed job", RuntimeWarning, stacklevel=2)
            raise
        self._closed = False

    @contextmanager
    def _launch(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("MegaMoeBF16Train has been closed")
            self._runtime.ensure_alive()
            if self._npu.current_stream() != self._stream:
                raise RuntimeError("MegaMoeBF16Train requires its construction stream")
            yield

    def _record(self, *tensors):
        for tensor in tensors:
            tensor.record_stream(self._stream)

    def _forward_diagnostics(
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
        diagnostic_holder=None,
    ):
        integer("max_recv_tokens", max_recv_tokens, 1, (1 << 31) - 1)
        limit = float(swiglu_limit)
        if not math.isfinite(limit) or not math.isfinite(c_float(limit).value):
            raise ValueError("swiglu_limit must be finite and representable as float32")
        if save_gmm1_output not in ("bf16", "none", "mxfp8", "auto"):
            raise ValueError("save_gmm1_output must be bf16, none, mxfp8 or auto")
        lease = None
        if torch.is_grad_enabled() and any(t.requires_grad for t in (x, topk_weights, w1, w2)):
            lease = _PendingBackward()
            self._pending.add(lease)
        return _MegaMoeBF16TrainFunction.apply(
            self,
            x,
            topk_ids,
            topk_weights,
            w1,
            w2,
            max_recv_tokens,
            limit,
            save_gmm1_output,
            diagnostic_holder,
            lease,
        )

    def forward(self, x, topk_ids, topk_weights, w1, w2, max_recv_tokens, *, swiglu_limit=0.0, save_gmm1_output="bf16"):
        """Return (y, counts); differentiate x, topk_weights, w1 and w2.

        All floating inputs are contiguous BF16 NPU tensors; IDs are int32.
        w1/w2 support source canonical [E,2F,H]/[E,H,F] and legacy
        [E,H,2F]/[E,F,H]. See README for capacity and saved-format semantics.
        """
        return self._forward_diagnostics(
            x,
            topk_ids,
            topk_weights,
            w1,
            w2,
            max_recv_tokens,
            swiglu_limit=swiglu_limit,
            save_gmm1_output=save_gmm1_output,
        )

    def _compute_reference(self, x, ids, weights, w1, w2, max_recv, mode):
        with self._launch():
            result = self._op.compute(x, ids, weights, w1, w2, max_recv, mode)
            self._record(x, ids, weights, w1, w2)
            return result

    def local_mem_size(self):
        return self._runtime.capacity

    def close(self):
        """Collectively release the window after outstanding backward consumers."""
        with self._lock:
            if self._closed:
                return
            if self._pending:
                raise RuntimeError("Complete backward or discard pending autograd graphs before close")
            self._runtime.ensure_alive()
            self._stream.synchronize()
            self._runtime.close()
            self._op.close()
            self._closed = True

    destroy = close

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __del__(self):
        if not getattr(self, "_closed", True):
            warnings.warn("Collectively close MegaMoeBF16Train before process-group teardown", ResourceWarning)

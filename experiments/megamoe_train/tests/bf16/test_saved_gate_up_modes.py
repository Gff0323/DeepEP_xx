"""CPU checks for per-call saved-state mode and autograd lifetime (no NPU)."""

from contextlib import nullcontext
from ctypes import c_float
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

from test_workspace_layout import dispatch_output_size, training_saved_state_size


ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def contiguous(self):
        return self


class Context:
    def save_for_backward(self, *values):
        self.saved_tensors = values

    def mark_non_differentiable(self, *values):
        pass


class Operator:
    def __init__(self):
        self.calls = []
        self._op = self
        self._pending = set()

    def _launch(self):
        return nullcontext()

    def _record(self, *tensors):
        pass

    def compute_train(self, *args):
        self.calls.append(("forward", args[-2:]))
        return Tensor(), Tensor(), Tensor(), Tensor(), Tensor()

    def backward(self, *args):
        self.calls.append(("backward", args[-2:]))
        return (Tensor(),) * 4

    def backward_debug(self, *args):
        self.calls.append(("debug", args[-2:]))
        return (Tensor(),) * 5


def load_bridge():
    # Run the actual autograd bridge without importing the NPU extension.
    tree = ast.parse((ROOT / "python/deep_ep_experimental/megamoe_train/bf16.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MegaMoeBF16TrainFunction")
    metadata = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_saved_metadata")
    namespace = {
        "torch": SimpleNamespace(autograd=SimpleNamespace(Function=object)),
        "once_differentiable": lambda f: f,
        "c_float": c_float,
    }
    exec(compile(ast.Module(body=[metadata, cls], type_ignores=[]), "autograd_bridge", "exec"), namespace)
    return namespace[cls.name]


class SavedModeTests(unittest.TestCase):
    def test_allocation_drops_only_gate_up_extension(self):
        args = (4096, 7168, 16, 8, 4, 65536, 8192)
        self.assertEqual(training_saved_state_size(*args, save_gate_up=False), 940185600)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=False), 2013927424)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=True), 1493833728)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=False, embed_dispatch_output=False), 1074272256)
        self.assertEqual(dispatch_output_size(7168, 65536), 939655168)

    def test_dispatch_is_retained_until_backward(self):
        bridge = load_bridge()
        op, ctx, lease = Operator(), Context(), object()
        op._pending.add(lease)
        bridge.forward(ctx, op, *(Tensor() for _ in range(5)), 65536, 0.0, "bf16", None, lease)
        self.assertEqual(len(ctx.saved_tensors), 6)
        self.assertIn(lease, op._pending)
        grads = bridge.backward(ctx, Tensor())
        self.assertEqual(len(grads), 11)
        self.assertNotIn(lease, op._pending)

    def test_interleaved_forwards_keep_modes_and_clipping(self):
        bridge = load_bridge()
        op = Operator()
        contexts = []
        modes = (("none", 0.0), ("bf16", 0.0), ("mxfp8", 0.05), ("auto", 10.0))
        for mode, limit in modes:
            ctx = Context()
            holder = {"capture_backward_workspace": True}
            bridge.forward(ctx, op, *(Tensor() for _ in range(5)), 65536, limit, mode, holder, None)
            self.assertNotIn(holder["forward_transient_workspace"], ctx.saved_tensors)
            contexts.append(ctx)
        for ctx in reversed(contexts):
            grads = bridge.backward(ctx, Tensor())
            self.assertEqual(len(grads), 11)
            self.assertEqual(grads[-5:], (None,) * 5)
            self.assertIn("backward_workspace", ctx.holder)
        self.assertEqual(op.calls[:4], [("forward", (limit, mode)) for mode, limit in modes])
        self.assertEqual(op.calls[4:], [("debug", (limit, mode)) for mode, limit in reversed(modes)])

    def test_cli_modes_and_default(self):
        from test_mega_moe import parse_args

        self.assertEqual(parse_args(["0,1"]).save_gmm1_output, "bf16")
        for mode in ("none", "bf16", "mxfp8", "auto"):
            self.assertEqual(parse_args(["0,1", "--save-gmm1-output", mode]).save_gmm1_output, mode)


if __name__ == "__main__":
    unittest.main()

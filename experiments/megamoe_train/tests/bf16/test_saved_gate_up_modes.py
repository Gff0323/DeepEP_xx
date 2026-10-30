"""CPU checks for per-call saved-state mode and autograd lifetime (no NPU)."""

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
        self._executor = self

    def _check_ready(self):
        pass

    def forward(self, *args):
        self.calls.append(("forward", args[-2:]))
        return SimpleNamespace(y=Tensor(), counts=Tensor(), saved_state=Tensor(), dispatch_output=Tensor())

    def backward(self, *args):
        self.calls.append(("backward", args[-2:]))
        return SimpleNamespace(dx=Tensor(), dtopk_weights=Tensor(), dw1=Tensor(), dw2=Tensor())


def load_bridge():
    # Run the actual autograd bridge without importing the NPU extension.
    tree = ast.parse((ROOT / "python/deep_ep_experimental/megamoe_train/bf16.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_MegaMoeBF16TrainFunction")
    namespace = {
        "torch": SimpleNamespace(autograd=SimpleNamespace(Function=object)),
        "once_differentiable": lambda f: f,
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "autograd_bridge", "exec"), namespace)
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
        op, ctx = Operator(), Context()
        bridge.forward(ctx, op, *(Tensor() for _ in range(5)), 65536, 0.0, "bf16")
        self.assertEqual(len(ctx.saved_tensors), 6)
        grads = bridge.backward(ctx, Tensor())
        self.assertEqual(len(grads), 9)

    def test_interleaved_forwards_keep_modes_and_clipping(self):
        bridge = load_bridge()
        op = Operator()
        contexts = []
        modes = (("none", 0.0), ("bf16", 0.0), ("mxfp8", 0.05), ("auto", 10.0))
        for mode, limit in modes:
            ctx = Context()
            bridge.forward(ctx, op, *(Tensor() for _ in range(5)), 65536, limit, mode)
            contexts.append(ctx)
        for ctx in reversed(contexts):
            grads = bridge.backward(ctx, Tensor())
            self.assertEqual(len(grads), 9)
            self.assertEqual(grads[-3:], (None,) * 3)
        self.assertEqual(op.calls[:4], [("forward", (limit, mode)) for mode, limit in modes])
        self.assertEqual(op.calls[4:], [("backward", (limit, mode)) for mode, limit in reversed(modes)])

    def test_cli_modes_and_default(self):
        from test_mega_moe import parse_args

        self.assertEqual(parse_args(["0,1"]).save_gmm1_output, "bf16")
        for mode in ("none", "bf16", "mxfp8", "auto"):
            self.assertEqual(parse_args(["0,1", "--save-gmm1-output", mode]).save_gmm1_output, mode)


if __name__ == "__main__":
    unittest.main()

"""Exercise the public wrapper without importing Torch or allocating an NPU."""

import ast
from ctypes import c_float
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from deep_ep._runtime import integer


SOURCE = Path(__file__).resolve().parents[2] / "python/deep_ep_experimental/megamoe_train/bf16.py"


def wrapper(apply):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegaMoeBF16Train"]
    namespace = {
        "torch": SimpleNamespace(nn=SimpleNamespace(Module=object)),
        "_MegaMoeBF16TrainFunction": SimpleNamespace(apply=apply),
        "integer": integer,
        "math": math,
        "c_float": c_float,
    }
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    cls = namespace["MegaMoeBF16Train"]
    op = object.__new__(cls)
    return cls, op


def test_public_signature_and_failed_launch_propagates_error():
    def fail(*args):
        raise RuntimeError("launch rejected")

    cls, op = wrapper(fail)
    assert list(inspect.signature(cls).parameters) == ["ep_group", "local_mem_size", "timeout"]
    assert list(inspect.signature(cls.forward).parameters) == [
        "self",
        "x",
        "topk_ids",
        "topk_weights",
        "w1",
        "w2",
        "max_recv_tokens",
        "swiglu_limit",
        "save_gmm1_output",
    ]
    with pytest.raises(ValueError, match="existing EP"):
        cls(None, 2 << 20)
    tensor = SimpleNamespace(requires_grad=True)
    with pytest.raises(RuntimeError, match="launch rejected"):
        op.forward(tensor, tensor, tensor, tensor, tensor, 128)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), 1e100])
def test_invalid_clipping_rejected_before_native_launch(limit):
    _, op = wrapper(lambda *args: pytest.fail("native launch"))
    tensor = SimpleNamespace(requires_grad=True)
    with pytest.raises(ValueError, match="finite"):
        op.forward(tensor, tensor, tensor, tensor, tensor, 128, swiglu_limit=limit)


def test_forward_passes_only_training_arguments_and_keeps_saved_modes():
    calls = []

    def launch(*args):
        calls.append(args)
        return "y", "counts"

    _, op = wrapper(launch)
    tensor = SimpleNamespace(requires_grad=False)
    for mode in ("bf16", "none", "mxfp8", "auto"):
        assert op.forward(tensor, tensor, tensor, tensor, tensor, 128, save_gmm1_output=mode) == ("y", "counts")
        assert len(calls[-1]) == 9
        assert calls[-1][-2:] == (0.0, mode)
    with pytest.raises(ValueError, match="save_gmm1_output"):
        op.forward(tensor, tensor, tensor, tensor, tensor, 128, save_gmm1_output="unknown")

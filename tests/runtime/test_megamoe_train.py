"""Training owner and package selection contracts; no device qualification."""

from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest

from deep_ep import _runtime


class TrainingRuntime(_runtime.Runtime):
    _megamoe_train = True


def test_training_owner_rejects_second_owner_and_releases(monkeypatch):
    calls = []
    session = SimpleNamespace(close=lambda: calls.append("close"), barrier=lambda: calls.append("barrier"))
    native = SimpleNamespace(
        Runtime=lambda *args: pytest.fail("selected core factory"),
        create_megamoe_train_runtime=lambda *args: calls.append(args) or session,
        runtime_busy=lambda: False,
    )
    npu = SimpleNamespace(
        current_device=lambda: 0,
        Stream=lambda **kwargs: SimpleNamespace(synchronize=lambda: None),
    )
    monkeypatch.setattr(_runtime, "require_native", lambda operation: native)
    monkeypatch.setattr(_runtime, "npu_api", lambda: npu)
    monkeypatch.setattr(_runtime, "_OWNER", None)
    monkeypatch.setenv("DEEPEP_SHMEM_ENDPOINT", "tcp://127.0.0.1:29862")
    runtime = TrainingRuntime(None, 3 << 30, rank=767, world_size=768)
    assert calls == [(767, 768, 3 << 30, "tcp://127.0.0.1:29862", 120)]
    assert _runtime._OWNER is runtime
    with pytest.raises(RuntimeError, match="already owns"):
        TrainingRuntime(None, 2 << 20, rank=0, world_size=2)
    runtime.close()
    runtime.close()
    assert calls[-1] == "close" and calls.count("close") == 1
    assert _runtime._OWNER is None


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_optional_package_combinations(monkeypatch, training, inference, fused):
    import setuptools

    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(root)
    for key, value in (
        ("DEEPEP_BUILD_MEGAMOE_TRAIN", training),
        ("DEEPEP_BUILD_MEGAMOE", inference),
        ("DEEPEP_BUILD_EP_GMM_FUSED", fused),
    ):
        monkeypatch.setenv(key, "ON" if value else "OFF")
    captured = {}
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: captured.update(kwargs))
    runpy.run_path(str(root / "setup.py"))
    for component, selected in (("megamoe_train", training), ("megamoe", inference), ("ep_gmm_fused", fused)):
        package = f"deep_ep_experimental.{component}"
        assert (package in captured["packages"]) is selected
        if selected:
            path = captured["package_dir"].get(package)
            if path is None:
                path = captured["package_dir"]["deep_ep_experimental"] + "/" + component
            assert (root / path / "__init__.py").is_file()
    assert len(captured["packages"]) == len(set(captured["packages"]))


def test_training_close_drains_stream_and_is_idempotent():
    # Exercise the actual wrapper's close method without importing Torch/NPU.
    import ast

    source = (
        Path(__file__).resolve().parents[2]
        / "experiments/megamoe_train/python/deep_ep_experimental/megamoe_train/bf16.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MegaMoeBF16Train")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "close")
    namespace = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    calls = []
    op = SimpleNamespace(
        _closed=False,
        _stream=SimpleNamespace(synchronize=lambda: calls.append("sync")),
        _runtime=SimpleNamespace(ensure_alive=lambda: None, close=lambda: calls.append("session-close")),
        _executor=SimpleNamespace(close=lambda: calls.append("op-close")),
    )
    close = namespace["close"]
    close(op)
    close(op)
    assert calls == ["sync", "session-close", "op-close"] and op._closed


def test_expert_ready_builds_use_distinct_cache_and_record_actual_policy(monkeypatch, tmp_path):
    import json
    import sys
    from types import ModuleType

    import build_support as build
    from setuptools import Distribution, Extension

    pybind11 = ModuleType("pybind11")
    pybind11.get_cmake_dir = lambda: str(tmp_path / "pybind11")
    monkeypatch.setitem(sys.modules, "pybind11", pybind11)
    monkeypatch.setenv("DEEPEP_BUILD_EP_GMM_FUSED", "OFF")
    monkeypatch.setenv("CMAKE_BUILD_PARALLEL_LEVEL", "2")
    # Exercise the real build command and cache key without invoking SDK tools.
    configuration = {
        "megamoe_train": {"enabled": True, "force_expert_ready": False},
        "source_sha256": "current",
        "source_version": "1",
        "cmake_path": "cmake",
        "compiler_path": "c++",
        "ASCEND_HOME_PATH": "/sdk/cann",
        "SHMEM_ROOT": "/sdk/shmem",
        "CATLASS_ROOT": "/sdk/catlass",
        "build_type": "Release",
    }
    monkeypatch.setattr(build, "build_configuration", lambda: configuration)
    configure_calls = []

    def run(argv):
        argv = list(map(str, argv))
        if "-B" in argv:
            configure_calls.append(argv)
            directory = Path(argv[argv.index("-B") + 1])
            (directory / "compiler-info.json").write_text('{"id":"test"}', encoding="utf-8")
            (directory / "capabilities.json").write_text('{"capabilities":[]}', encoding="utf-8")

    monkeypatch.setattr(build, "checked_run", run)
    ext = Extension("deep_ep._C", sources=[])
    for forced in (False, True):
        configuration["megamoe_train"]["force_expert_ready"] = forced
        command = build.CMakeBuild(Distribution({"ext_modules": [ext]}))
        command.ensure_finalized()
        command.build_temp = str(tmp_path / "temp")
        command.build_lib = str(tmp_path / "lib")
        destination = Path(command.get_ext_fullpath(ext.name)).parent
        destination.mkdir(parents=True, exist_ok=True)
        command.build_extension(ext)
        args = configure_calls[-1]
        assert f"-DDEEPEP_BF16_FORCE_EXPERT_READY={'ON' if forced else 'OFF'}" in args
        assert "-DDEEPEP_BUILD_MEGAMOE_TRAIN=ON" in args
        info = json.loads((destination / "_build_info.json").read_text(encoding="utf-8"))
        assert info["megamoe_train"]["force_expert_ready"] is forced
    object_directories = [args[args.index("-B") + 1] for args in configure_calls]
    assert len(set(object_directories)) == 2

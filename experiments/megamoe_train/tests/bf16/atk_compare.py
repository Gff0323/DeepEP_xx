"""Optional ATK CV double-benchmark comparison of every rank/output.

Use ATK's registered implementation, without flattening different gradients or
ranks together. The existing launcher owns execution; ATK owns this additional
precision decision. No replacement formulas or open-source fallbacks are used.
"""

from importlib import metadata
import os
from pathlib import Path

import torch

from precision_metrics import OUTPUTS


STANDARD = "cv_fused_double_benchmark"


def require_atk():
    try:
        from atk.tasks.post_process import ACCURACY_REGISTRY
        from atk.tasks.post_process.private_loader import (
            get_accuracy_provider,
            get_private_accuracy_load_error,
            is_private_accuracy_loaded,
        )
    except ImportError as error:
        raise RuntimeError("--atk requires ATK and atk-private-precision in this Python environment") from error
    error = get_private_accuracy_load_error()
    if error is not None or not is_private_accuracy_loaded() or STANDARD not in ACCURACY_REGISTRY:
        raise RuntimeError(f"ATK private comparator {STANDARD} is unavailable: {error!r}")
    comparator = ACCURACY_REGISTRY[STANDARD]
    if get_accuracy_provider(STANDARD) != "private" or not comparator.__module__.startswith("atk_private_precision."):
        raise RuntimeError(f"ATK {STANDARD} must come from atk-private-precision")
    return comparator, {
        "standard": STANDARD,
        "provider": "private",
        "atk_version": metadata.version("ATK"),
        "private_version": metadata.version("atk-private-precision"),
        "implementation": comparator.__module__ + "." + comparator.__name__,
    }


def compare_with_atk(actual, baseline, truth, directory, ratios):
    """ATK local=actual, remote=raw BigOp, bm_path=CPU FP64 golden.

    This is CompareExecutor's three-role contract, called separately for the
    five files in one rank. Keep the golden tensor in FP64 and actual/benchmark
    in their original BF16 dtype. Retain failing triples for later diagnosis.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["ATK_TASK_OUTPUT_PATH"] = str(directory)
    comparator_type, identity = require_atk()
    from atk.configs.case_config import CaseConfig

    thresholds = dict(zip(("max_re_ratio", "avg_re_ratio", "root_mean_squared_ratio"), ratios, strict=True))
    thresholds["number_count_radio"] = 2
    case = CaseConfig(id=0, name="megamoe_bf16_train", inputs=[], standard={"acc": {STANDARD: thresholds}})
    comparator = comparator_type(case, **thresholds)
    # Check that constructor kwargs reached the actual plugin configuration.
    for key, value in thresholds.items():
        if getattr(comparator.benchmark_standard, key) != value:
            raise RuntimeError(f"ATK did not apply configured {key}={value}")
    golden_dir = directory / "golden"
    golden_dir.mkdir()
    comparator.bm_path = str(golden_dir)
    report = {**identity, "thresholds": thresholds, "passed": True, "outputs": {}}
    for index, name in enumerate(OUTPUTS):
        a, b, g = (group[name].detach().cpu().contiguous() for group in (actual, baseline, truth))
        if a.shape != b.shape or a.shape != g.shape or a.dtype != b.dtype or g.dtype != torch.float64:
            raise ValueError(f"ATK {name}: shapes must match, actual/BigOp dtypes must match, golden must be FP64")
        data_file = f"output_{index}.pt"
        golden_file = golden_dir / data_file
        if all(bool(torch.isfinite(value).all()) for value in (a, b, g)):
            torch.save(g.reshape(-1), golden_file)
            result = comparator.compute_accuracy_result(a.reshape(-1), b.reshape(-1), data_file)
            detail = result.model_dump(mode="json")
            if type(detail.get("result")) is not bool:
                raise RuntimeError(f"ATK returned no boolean decision for {name}: {detail}")
            passed = detail["result"]
        else:
            passed, detail = False, {"result": False, "error_info": "Non-finite actual, BigOp or CPU FP64 golden"}
        record = {"passed": passed, "shape": list(a.shape), "dtype": str(a.dtype), "file": data_file, "result": detail}
        if passed:
            golden_file.unlink(missing_ok=True)
        else:
            torch.save(g, golden_file)
            for role, value in (("actual", a), ("benchmark", b)):
                target = directory / role
                target.mkdir(exist_ok=True)
                torch.save(value, target / data_file)
            record["artifacts"] = str(directory)
        report["outputs"][name] = record
        report["passed"] &= passed
    return report

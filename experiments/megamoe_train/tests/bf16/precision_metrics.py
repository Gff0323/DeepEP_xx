"""Torch CPU precision criteria and JSON reports.

Acceptance follows ascend_deepep_hkq@568ac7c at the public operator boundary.
Dual-baseline errors use FP64 truth and every element; mixed tolerance follows
the actual output dtype, with both the matched-ratio and maximum-error gates.
"""

import json
import math
from pathlib import Path

import torch


OUTPUTS = ("y", "dx", "dtopk_weights", "dw1", "dw2")
RATIO_LIMITS = {"MARE": 5.0, "MERE": 1.5, "RMSE": 1.5}
ERROR_FLOOR = 2.0**-8
RELATIVE_EPS = 1e-7
DTYPE_TOLERANCES = {
    torch.bfloat16: (2.0**-7, 0.1),
    torch.float16: (2.0**-10, 0.1),
    torch.float32: (2.0**-13, 0.01),
}


def _cpu_flat(value):
    return value.detach().to(device="cpu").reshape(-1)


def error_metrics(actual, truth, chunk_elements=1 << 20):
    actual, truth = _cpu_flat(actual), _cpu_flat(truth)
    if actual.numel() != truth.numel() or chunk_elements <= 0:
        raise ValueError("Mismatched element counts or invalid metric chunk size")
    invalid = {"finite": False, "MARE": math.inf, "MERE": math.inf, "RMSE": math.inf}
    maximum, relative_sums, squared_sums = 0.0, [], []
    for begin in range(0, truth.numel(), chunk_elements):
        a = actual[begin : begin + chunk_elements].double()
        g = truth[begin : begin + chunk_elements].double()
        if not bool((torch.isfinite(a) & torch.isfinite(g)).all()):
            return invalid
        diff = (a - g).abs()
        relative = diff / (g.abs() + RELATIVE_EPS)
        maximum = max(maximum, relative.max().item())
        relative_sum, squared_sum = relative.sum().item(), diff.square().sum().item()
        if not all(math.isfinite(value) for value in (maximum, relative_sum, squared_sum)):
            return invalid
        relative_sums.append(relative_sum)
        squared_sums.append(squared_sum)
    try:
        relative_total, squared_total = math.fsum(relative_sums), math.fsum(squared_sums)
    except OverflowError:
        return invalid
    return {
        "finite": True,
        "MARE": maximum,
        "MERE": relative_total / max(truth.numel(), 1),
        "RMSE": math.sqrt(squared_total / max(truth.numel(), 1)),
    }


def _output_ulp(golden, dtype):
    anchor = abs(golden.to(dtype).item())
    if math.isinf(anchor):
        return math.inf
    info = torch.finfo(dtype)
    if anchor < info.tiny:
        return info.tiny * info.eps
    return math.ldexp(info.eps, math.frexp(anchor)[1] - 1)


def mixed_tolerance(actual, reference, dtype):
    if dtype not in DTYPE_TOLERANCES:
        raise ValueError(f"Unsupported actual output dtype: {dtype}")
    a, reference = _cpu_flat(actual).float(), _cpu_flat(reference)
    g = reference.float()
    if a.numel() != g.numel():
        return {"passed": False, "error": f"numel {a.numel()} != {g.numel()}"}
    tolerance, absolute_floor = DTYPE_TOLERANCES[dtype]
    finite = torch.isfinite(a) & torch.isfinite(g)
    diff = (a - g).abs()
    close = finite & (diff <= tolerance + tolerance * g.abs())
    close |= torch.isinf(a) & (a == reference.to(dtype).float())
    bad = int((~close).sum().item())
    matched = (a.numel() - bad) / a.numel() if a.numel() else 1.0
    maximum, limit, index = 0.0, absolute_floor, None
    if bool(finite.any()):
        index = int(torch.where(finite, diff, torch.full_like(diff, -1)).argmax().item())
        maximum = diff[index].item()
        limit = max(absolute_floor, 32 * _output_ulp(reference[index], dtype))
    return {
        "passed": matched >= 0.99 and maximum <= limit,
        "dtype": str(dtype).removeprefix("torch."),
        "atol": tolerance,
        "rtol": tolerance,
        "matched_ratio": matched,
        "bad": bad,
        "numel": a.numel(),
        "max_abs": maximum,
        "max_abs_limit": limit,
        "max_abs_index": index,
    }


def dual_baseline(actual, baseline, truth):
    b, f = error_metrics(baseline, truth), error_metrics(actual, truth)
    finite = b["finite"] and f["finite"]
    ratios = {name: f[name] / max(b[name], ERROR_FLOOR) if finite else math.inf for name in RATIO_LIMITS}
    return {
        "passed": finite and all(ratios[name] <= limit for name, limit in RATIO_LIMITS.items()),
        "baseline": b,
        "fused": f,
        "ratios": ratios,
        "limits": RATIO_LIMITS,
        "denominator_floor": ERROR_FLOOR,
    }


def evaluate(actual, raw_baseline, mode_baseline, truth, expected_counts):
    missing = [name for name in (*OUTPUTS, "counts") if actual.get(name) is None]
    if missing:
        return {"passed": False, "error": f"Missing outputs/gradients: {missing}"}
    counts = actual["counts"].detach().to(device="cpu", dtype=torch.int64)
    expected_counts = expected_counts.detach().to(device="cpu", dtype=torch.int64)
    counts_ok = counts.shape == expected_counts.shape and torch.equal(counts, expected_counts)
    report = {"passed": counts_ok, "counts_passed": counts_ok, "outputs": {}}
    for name in OUTPUTS:
        mixed = mixed_tolerance(actual[name], mode_baseline[name], actual[name].dtype)
        try:
            dual = dual_baseline(actual[name], raw_baseline[name], truth[name])
        except ValueError as error:
            dual = {"passed": False, "error": str(error)}
        report["outputs"][name] = {"mixed_tolerance": mixed, "dual_baseline": dual}
        report["passed"] &= mixed["passed"] and dual["passed"]
    return report


def write_report(path, value):
    def safe(item):
        if isinstance(item, dict):
            return {key: safe(value) for key, value in item.items()}
        if isinstance(item, (tuple, list)):
            return [safe(value) for value in item]
        if isinstance(item, float) and not math.isfinite(item):
            return str(item)
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

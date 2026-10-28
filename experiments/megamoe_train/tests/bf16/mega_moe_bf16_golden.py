#!/usr/bin/env python3
"""Generate and compare full-pipeline MegaMoe BF16 test data.

The golden models the kernel's visible rounding points:
GMM1 BF16 output, BF16 SwiGLU output, BF16 route weighting (either immediately
after SwiGLU or during the final unpermute), and GMM2 BF16 output.
Deterministic dense weights are generated in chunks so the precision stimulus
matches the W8A8 test without keeping every expert's weights resident at once.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def float32_to_bf16_bits(data: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(data, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float32(data: np.ndarray) -> np.ndarray:
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def bf16_round(data: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(float32_to_bf16_bits(data))


def file_suffix(case: dict) -> str:
    return f"{case['M']}_{case['K']}_{case['N']}_{case['EP']}.bin"


def data_path(data_dir: Path, rank: int, name: str, case: dict) -> Path:
    return data_dir / f"rank{rank}_{name}_{file_suffix(case)}"


def write_array(path: Path, data: np.ndarray, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab" if append else "wb") as output:
        output.write(np.ascontiguousarray(data).tobytes())


def _random_dense_bf16_weights(
    rng: np.random.Generator,
    expert_count: int,
    output_size: int,
    input_size: int,
) -> np.ndarray:
    safe_values = np.array(
        [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0],
        dtype=np.float32,
    )
    safe_value_bits = float32_to_bf16_bits(safe_values)
    indices = rng.integers(
        0,
        safe_value_bits.size,
        size=(expert_count, output_size, input_size),
        dtype=np.uint8,
    )
    return np.take(safe_value_bits, indices)


def _random_expert_ids(
    rng: np.random.Generator,
    token_count: int,
    topk: int,
    total_experts: int,
    routing_skew: float,
) -> np.ndarray:
    """Sample distinct expert IDs per token with bounded host peak memory.

    ``routing_skew == 0`` gives uniform random routing. A positive value adds
    an exponentially decaying expert log-weight and therefore creates a
    deterministic, seed-controlled hot-expert distribution.
    """
    expert_ids = np.empty((token_count, topk), dtype=np.int32)
    expert_axis = np.arange(total_experts, dtype=np.float64)
    if total_experts == 1:
        log_weights = np.zeros(1, dtype=np.float64)
    else:
        log_weights = -routing_skew * expert_axis / (total_experts - 1)

    # Gumbel top-k samples without replacement. Chunking prevents an
    # [global_m, total_experts] temporary from becoming excessively large.
    chunk_rows = 1024
    partition_index = total_experts - topk
    for row_begin in range(0, token_count, chunk_rows):
        row_end = min(row_begin + chunk_rows, token_count)
        scores = rng.gumbel(size=(row_end - row_begin, total_experts)) + log_weights
        selected = np.argpartition(scores, partition_index, axis=1)[:, partition_index:]
        selected_scores = np.take_along_axis(scores, selected, axis=1)
        selected_order = np.argsort(-selected_scores, axis=1)
        expert_ids[row_begin:row_end] = np.take_along_axis(selected, selected_order, axis=1).astype(
            np.int32, copy=False
        )
    return expert_ids


def _generate_expert_ids(
    rng: np.random.Generator,
    token_count: int,
    topk: int,
    total_experts: int,
    routing_mode: str,
    routing_skew: float,
) -> np.ndarray:
    if routing_mode == "balanced":
        route_index = np.arange(token_count * topk, dtype=np.int64)
        return (route_index % total_experts).reshape(token_count, topk).astype(np.int32)
    if routing_mode == "random":
        return _random_expert_ids(rng, token_count, topk, total_experts, routing_skew=0.0)
    if routing_mode == "imbalanced":
        return _random_expert_ids(rng, token_count, topk, total_experts, routing_skew=routing_skew)
    raise ValueError(f"unsupported routing mode: {routing_mode}")


def _print_routing_summary(
    expert_ids: np.ndarray,
    ep: int,
    local_experts: int,
    routing_mode: str,
) -> None:
    expert_rows = np.bincount(
        np.asarray(expert_ids, dtype=np.int64).reshape(-1),
        minlength=ep * local_experts,
    )
    rank_rows = expert_rows.reshape(ep, local_experts).sum(axis=1)
    rank_text = ",".join(str(int(value)) for value in rank_rows)
    print(
        f"[ROUTING] mode={routing_mode} rank_rows=[{rank_text}] "
        f"expert_min={int(expert_rows.min())} "
        f"expert_max={int(expert_rows.max())}",
        flush=True,
    )


def validate_recv_capacity(case: dict, expert_ids: np.ndarray) -> None:
    """Fail test-data generation before an undersized fused run can truncate."""
    ep = int(case["EP"])
    local_experts = int(case["expertPerRank"])
    max_recv_tokens = int(case["maxRecvTokenNum"])
    owner_ranks = np.asarray(expert_ids, dtype=np.int64).reshape(-1) // local_experts
    recv_rows = np.bincount(owner_ranks, minlength=ep)
    overflowing = np.flatnonzero(recv_rows > max_recv_tokens)
    if overflowing.size == 0:
        return

    details = ", ".join(f"rank {int(rank)} receives {int(recv_rows[rank])} routed rows" for rank in overflowing)
    raise ValueError(
        "MegaMoe BF16 routed-token capacity exceeded: "
        f"{details}, but maxRecvTokenNum is {max_recv_tokens}. "
        "Use a larger maxRecvTokenNum and correspondingly larger local "
        "SHMEM allocation before running the fused operator."
    )


def generate(
    case: dict,
    data_dir: Path,
    seed: int,
    generate_golden: bool = True,
    routing_mode: str = "balanced",
    routing_skew: float = 1.0,
    topk_weight_mode: str = "random",
    early_topk_weight: bool = False,
) -> None:
    m = case["M"]
    n = case["N"]
    k = case["K"]
    topk = case["TOPK"]
    local_experts = case["expertPerRank"]
    ep = case["EP"]
    total_experts = ep * local_experts
    global_m = ep * m
    if k % 256 != 0 or n % 512 != 0:
        raise ValueError("BF16 golden requires K % 256 == 0 and N % 512 == 0")
    if topk > total_experts:
        raise ValueError("TOPK cannot exceed the total expert count")

    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    x = bf16_round(rng.uniform(-0.1, 0.1, size=(global_m, k)).astype(np.float32))

    expert_ids = _generate_expert_ids(
        rng,
        global_m,
        topk,
        total_experts,
        routing_mode,
        routing_skew,
    )
    _print_routing_summary(expert_ids, ep, local_experts, routing_mode)
    if case.get("validateRecvCapacity", False):
        validate_recv_capacity(case, expert_ids)
    logits = rng.uniform(-1.0, 1.0, size=(global_m, topk)).astype(np.float32)
    if topk_weight_mode == "random":
        logits -= np.max(logits, axis=1, keepdims=True)
        topk_weights = np.exp(logits)
        topk_weights /= np.sum(topk_weights, axis=1, keepdims=True)
    elif topk_weight_mode == "uniform":
        topk_weights = np.full((global_m, topk), 1.0 / topk, dtype=np.float32)
    else:
        raise ValueError(f"unsupported topk_weight_mode: {topk_weight_mode}")
    topk_weights = bf16_round(topk_weights)

    for rank in range(ep):
        rank_slice = slice(rank * m, (rank + 1) * m)
        write_array(
            data_path(data_dir, rank, "input_x_bf16", case),
            float32_to_bf16_bits(x[rank_slice]),
        )
        write_array(
            data_path(data_dir, rank, "input_expert_ids_bf16", case),
            expert_ids[rank_slice],
        )
        write_array(
            data_path(data_dir, rank, "input_expert_weights_bf16", case),
            float32_to_bf16_bits(topk_weights[rank_slice]),
        )
        write_array(
            data_path(data_dir, rank, "weight1_bf16", case),
            np.empty(0, dtype=np.uint16),
        )
        write_array(
            data_path(data_dir, rank, "weight2_bf16", case),
            np.empty(0, dtype=np.uint16),
        )

    route_outputs = np.zeros((global_m, topk, k), dtype=np.uint16) if generate_golden else None
    expert_counts = np.zeros(total_experts, dtype=np.int32) if generate_golden else None
    swiglu_width = n // 2

    # BF16 weights occupy twice the storage of W8A8 weights.  Four experts
    # per chunk keeps the production-shape host peak bounded while retaining
    # the same dense random stimulus used by the W8A8 precision test.
    weight_chunk_size = 4
    for chunk_start in range(0, total_experts, weight_chunk_size):
        chunk_end = min(chunk_start + weight_chunk_size, total_experts)
        chunk_experts = chunk_end - chunk_start
        weight1_chunk = _random_dense_bf16_weights(
            rng,
            chunk_experts,
            n,
            k,
        )
        weight2_chunk = _random_dense_bf16_weights(
            rng,
            chunk_experts,
            k,
            swiglu_width,
        )

        for chunk_offset, expert in enumerate(range(chunk_start, chunk_end)):
            rank = expert // local_experts
            weight1 = weight1_chunk[chunk_offset]
            weight2 = weight2_chunk[chunk_offset]
            write_array(
                data_path(data_dir, rank, "weight1_bf16", case),
                weight1,
                append=True,
            )
            write_array(
                data_path(data_dir, rank, "weight2_bf16", case),
                weight2,
                append=True,
            )

            if not generate_golden:
                continue

            token_and_route = np.argwhere(expert_ids == expert)
            expert_counts[expert] = token_and_route.shape[0]
            if token_and_route.size == 0:
                continue

            token_indices = token_and_route[:, 0]
            weight1_fp32 = bf16_bits_to_float32(weight1)
            gmm1 = bf16_round(x[token_indices] @ weight1_fp32.T)
            del weight1_fp32
            gate, up = np.split(gmm1, 2, axis=1)
            swiglu = bf16_round(gate / (1.0 + np.exp(-gate)) * up)
            if early_topk_weight:
                route_indices = token_and_route[:, 1]
                swiglu = bf16_round(swiglu * topk_weights[token_indices, route_indices, None])
            del gmm1, gate, up
            weight2_fp32 = bf16_bits_to_float32(weight2)
            gmm2 = bf16_round(swiglu @ weight2_fp32.T)
            del weight2_fp32
            route_outputs[
                token_indices,
                token_and_route[:, 1],
                :,
            ] = float32_to_bf16_bits(gmm2)

        del weight1_chunk, weight2_chunk

    if not generate_golden:
        print(
            f"[INPUT] generated BF16 M={m} N={n} K={k} EP={ep} "
            f"experts={total_experts} topk_weight_mode={topk_weight_mode} "
            f"data_dir={data_dir}",
            flush=True,
        )
        return

    combined = np.zeros((global_m, k), dtype=np.float32)
    for route in range(topk):
        route_value = bf16_bits_to_float32(route_outputs[:, route, :])
        contribution = route_value if early_topk_weight else bf16_round(route_value * topk_weights[:, route, None])
        combined = contribution if route == 0 else bf16_round(combined + contribution)

    for rank in range(ep):
        rank_slice = slice(rank * m, (rank + 1) * m)
        expert_slice = slice(rank * local_experts, (rank + 1) * local_experts)
        write_array(
            data_path(data_dir, rank, "golden_y_bf16", case),
            float32_to_bf16_bits(combined[rank_slice]),
        )
        write_array(
            data_path(data_dir, rank, "golden_expert_token_nums_bf16", case),
            expert_counts[expert_slice],
        )

    print(
        f"[GOLDEN] generated BF16 M={m} N={n} K={k} EP={ep} "
        f"experts={total_experts} "
        f"topk_weight_mode={topk_weight_mode} "
        f"topk_weight_stage={'post_swiglu' if early_topk_weight else 'unpermute'} "
        f"data_dir={data_dir}",
        flush=True,
    )


def deviation_ratio(
    actual: np.ndarray,
    golden: np.ndarray,
    absolute_error_threshold: float,
) -> np.ndarray:
    """Return the W8A8-style protected per-element deviation."""
    absolute_diff = np.abs(actual - golden)
    denominator_floor = (1.0 / (1 << 14)) / absolute_error_threshold
    denominator = np.maximum(
        np.maximum(np.abs(actual), np.abs(golden)),
        denominator_floor,
    )
    relative_deviation = absolute_diff / (denominator + 1e-10)
    deviation = np.where(
        absolute_diff < absolute_error_threshold,
        absolute_diff,
        relative_deviation,
    )
    both_equal = actual == golden
    return np.where(
        np.isfinite(deviation),
        deviation,
        np.where(both_equal, 0.0, np.inf),
    )


def _diagnostic_sections(size: int) -> list[tuple[str, np.ndarray]]:
    if size <= 0:
        return []
    middle_start = max(0, size // 2 - 5)
    windows = (
        ("first", 0),
        ("middle", middle_start),
        ("last", max(0, size - 10)),
    )
    sections = []
    used_ranges = set()
    for label, start in windows:
        stop = min(size, start + 10)
        key = (start, stop)
        if key not in used_ranges:
            sections.append((label, np.arange(start, stop)))
            used_ranges.add(key)
    return sections


def _print_pair_diagnostics(
    rank: int,
    actual: np.ndarray,
    golden: np.ndarray,
    deviation: np.ndarray,
    incorrect: np.ndarray,
    k: int,
) -> None:
    print(
        f"[PRECISION][FUSED-GOLDEN] rank={rank} diagnostic values (flattened indices)",
        flush=True,
    )
    for label, indices in _diagnostic_sections(golden.size):
        print(f"  [{label.upper()} 10]", flush=True)
        for index in indices:
            tensor_index = divmod(int(index), k)
            absolute_diff = abs(float(actual[index] - golden[index]))
            print(
                f"  index={int(index)} tensor_index={tensor_index} "
                f"golden={golden[index]:.8g} actual={actual[index]:.8g} "
                f"absolute_diff={absolute_diff:.8g} "
                f"deviation={deviation[index]:.6%}",
                flush=True,
            )
    bad_indices = np.flatnonzero(incorrect)[:10]
    if bad_indices.size:
        print("  [FIRST 10 INCORRECT]", flush=True)
        for index in bad_indices:
            tensor_index = divmod(int(index), k)
            print(
                f"  index={int(index)} tensor_index={tensor_index} "
                f"golden={golden[index]:.8g} actual={actual[index]:.8g} "
                f"deviation={deviation[index]:.6%}",
                flush=True,
            )


def _compare_y_pair(
    rank: int,
    case: dict,
    actual_path: Path,
    golden_path: Path,
    deviation_threshold: float,
    absolute_error_threshold: float,
    max_error_ratio: float,
    print_diagnostics: bool,
    failures: list[str],
) -> None:
    missing_paths = [path for path in (actual_path, golden_path) if not path.is_file()]
    if missing_paths:
        failures.extend(f"FUSED-GOLDEN rank {rank}: missing {path}" for path in missing_paths)
        return

    actual = bf16_bits_to_float32(np.fromfile(actual_path, np.uint16))
    golden = bf16_bits_to_float32(np.fromfile(golden_path, np.uint16))
    if actual.size != golden.size:
        failures.append(f"FUSED-GOLDEN rank {rank}: y size {actual.size} != {golden.size}")
        return

    deviation = deviation_ratio(actual, golden, absolute_error_threshold)
    incorrect = deviation >= deviation_threshold
    error_ratio = float(np.count_nonzero(incorrect)) / max(1, incorrect.size)
    max_deviation = float(np.max(deviation)) if deviation.size else 0.0
    result = "PASS" if error_ratio <= max_error_ratio else "FAIL"
    print(
        f"[PRECISION][FUSED-GOLDEN] rank={rank} y={result} "
        f"error_ratio={error_ratio:.6%} "
        f"max_deviation={max_deviation:.6%} "
        f"deviation_threshold={deviation_threshold:.6%}",
        flush=True,
    )
    if deviation.size:
        max_index = int(np.argmax(deviation))
        tensor_index = divmod(max_index, case["K"])
        max_absolute_diff = float(abs(actual[max_index] - golden[max_index]))
        print(
            f"[PRECISION][FUSED-GOLDEN][MAX-DEVIATION] rank={rank} "
            f"index={max_index} tensor_index={tensor_index} "
            f"golden={golden[max_index]:.8g} "
            f"actual={actual[max_index]:.8g} "
            f"absolute_diff={max_absolute_diff:.8g} "
            f"deviation={deviation[max_index]:.6%}",
            flush=True,
        )
    if print_diagnostics or result == "FAIL":
        _print_pair_diagnostics(
            rank,
            actual,
            golden,
            deviation,
            incorrect,
            case["K"],
        )
    if result == "FAIL":
        failures.append(f"FUSED-GOLDEN rank {rank}: y error ratio {error_ratio:.6%} exceeds {max_error_ratio:.6%}")


def _compare_token_pair(
    rank: int,
    actual_path: Path,
    golden_path: Path,
    failures: list[str],
) -> None:
    missing_paths = [path for path in (actual_path, golden_path) if not path.is_file()]
    if missing_paths:
        failures.extend(f"FUSED-GOLDEN rank {rank}: missing {path}" for path in missing_paths)
        return

    actual_tokens = np.fromfile(actual_path, np.int32)
    golden_tokens = np.fromfile(golden_path, np.int32)
    tokens_equal = np.array_equal(actual_tokens, golden_tokens)
    print(
        f"[PRECISION][FUSED-GOLDEN] rank={rank} expert_token_nums={'PASS' if tokens_equal else 'FAIL'}",
        flush=True,
    )
    if not tokens_equal:
        failures.append(
            f"FUSED-GOLDEN rank {rank}: expert_token_nums "
            f"actual={actual_tokens.tolist()} "
            f"expected={golden_tokens.tolist()}"
        )


def compare(
    case: dict,
    data_dir: Path,
    absolute_tolerance: float,
    relative_tolerance: float,
    max_error_ratio: float,
    print_diagnostics: bool = False,
    ranks=None,
) -> None:
    if absolute_tolerance <= 0:
        raise ValueError("absolute_tolerance must be positive")
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if not 0.0 <= max_error_ratio <= 1.0:
        raise ValueError("max_error_ratio must be in [0, 1]")

    failures = []
    for rank in range(case["EP"]) if ranks is None else ranks:
        _compare_y_pair(
            rank,
            case,
            data_path(data_dir, rank, "output_y_bf16", case),
            data_path(data_dir, rank, "golden_y_bf16", case),
            relative_tolerance,
            absolute_tolerance,
            max_error_ratio,
            print_diagnostics,
            failures,
        )
        _compare_token_pair(
            rank,
            data_path(
                data_dir,
                rank,
                "output_expert_token_nums_bf16",
                case,
            ),
            data_path(
                data_dir,
                rank,
                "golden_expert_token_nums_bf16",
                case,
            ),
            failures,
        )

    if failures:
        for failure in failures:
            print(f"[ERROR] {failure}", file=sys.stderr)
        raise AssertionError("; ".join(failures))
    print(
        f"[PRECISION] MegaMoe BF16 M={case['M']} N={case['N']} K={case['K']} passed",
        flush=True,
    )

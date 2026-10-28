#!/usr/bin/env python3
"""MegaMoe BF16 training backward precision test.

Each EP rank runs the differentiable MegaMoeBF16Train module, builds a scalar loss
from the output and calls loss.backward().
grad_x / grad_weight1 / grad_weight2 are compared against a float32 CPU
reference that replays the same routing on the global data.

Usage:
    python test_mega_moe_backward.py 0,1 --seq 129 --hidden 256 --gmm-n 1024 \
        --topk 4 --expert-per-rank 2 --check-intermediates
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

TEST_DIR = Path(__file__).resolve().parent
TOPK = 8
EXPERTS_PER_RANK = 4
LOCAL_MEM_BYTES = 1024 * 1024 * 1024
DEFAULT_SEED = 20260825
ACLSHMEM_DATA_OP_UDMA = 0x08
ACLSHMEM_DATA_OP_SDMA = 0x02


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


def random_dense_bf16_weights(
    rng: np.random.Generator,
    expert_count: int,
    output_size: int,
    input_size: int,
) -> np.ndarray:
    safe_values = np.array([-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    indices = rng.integers(0, len(safe_values), size=(expert_count, output_size, input_size))
    dense = bf16_round(rng.uniform(-1.0, 1.0, size=(expert_count, output_size, input_size)))
    mask = rng.random((expert_count, output_size, input_size)) < 0.05
    # safe_values are exactly representable in BF16.  Assign the values, not
    # their uint16 bit patterns (for example, BF16 1.0 has bits 0x3f80=16256).
    dense[mask] = safe_values[indices[mask]]
    return dense


def generate_expert_ids(
    rng: np.random.Generator,
    token_count: int,
    topk: int,
    total_experts: int,
    routing_mode: str,
) -> np.ndarray:
    if routing_mode == "balanced":
        route_index = np.arange(token_count * topk, dtype=np.int64)
        return (route_index % total_experts).reshape(token_count, topk).astype(np.int32)
    if routing_mode == "random":
        return rng.integers(0, total_experts, size=(token_count, topk)).astype(np.int32)
    if routing_mode == "imbalanced":
        # Favour low-index experts so a few experts are hot and at least one
        # is empty with high probability.
        expert_axis = np.arange(total_experts, dtype=np.float64)
        log_weights = -2.0 * expert_axis / max(total_experts - 1, 1)
        flat = rng.gumbel(size=(token_count * topk, total_experts)) + log_weights
        return np.argmax(flat, axis=1).reshape(token_count, topk).astype(np.int32)
    raise ValueError(f"unsupported routing_mode: {routing_mode}")


def reference_backward(
    x: np.ndarray,
    expert_ids: np.ndarray,
    topk_weights: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    grad_y: np.ndarray,
    save_mode: str = "bf16",
    swiglu_limit: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Float32 reference backward on the GLOBAL data (all ranks' tokens)."""
    total_experts = w1.shape[0]
    grad_x = np.zeros_like(x, dtype=np.float32)
    grad_topk = np.zeros_like(topk_weights, dtype=np.float32)
    grad_w1 = np.zeros_like(w1, dtype=np.float32)
    grad_w2 = np.zeros_like(w2, dtype=np.float32)
    for expert in range(total_experts):
        rows = np.argwhere(expert_ids == expert)
        if rows.size == 0:
            continue
        tokens = rows[:, 0]
        slots = rows[:, 1]
        p = topk_weights[tokens, slots].astype(np.float32)
        xr = x[tokens].astype(np.float32)
        dO = grad_y[tokens].astype(np.float32)
        w1_e = w1[expert].astype(np.float32)
        w2_e = w2[expert].astype(np.float32)

        z = bf16_round(xr @ w1_e.T)
        raw_gate, raw_up = np.split(z, 2, axis=1)
        gate, up = raw_gate, raw_up
        if save_mode == "mxfp8":
            from saved_gate_up_reference import w8a8_reference

            _, gate, _, _ = w8a8_reference().mxfp8_quantize(bf16_round(gate), -1)
            _, up, _, _ = w8a8_reference().mxfp8_quantize(bf16_round(up), -1)
        gate_ind = np.ones_like(gate, dtype=np.float32)
        up_ind = np.ones_like(up, dtype=np.float32)
        if swiglu_limit > 0:
            limit = np.float32(swiglu_limit)
            gate_high = raw_gate > limit
            up_low = raw_up < -limit
            up_high = raw_up > limit
            gate = np.minimum(gate, limit)
            up = np.clip(up, -limit, limit)
            if save_mode == "mxfp8":
                gate = np.where(gate_high, limit, gate)
                up = np.where(up_low, -limit, np.where(up_high, limit, up))
            gate_ind = (~gate_high).astype(np.float32)
            up_ind = (~(up_low | up_high)).astype(np.float32)
        s = 1.0 / (1.0 + np.exp(-gate))
        h = gate * s * up

        dHp = dO @ w2_e
        grad_topk[tokens, slots] = np.sum(dHp * h, axis=1)
        dH = p[:, None] * dHp
        dUp = dH * (gate * s) * up_ind
        dGate = dH * up * (s * (1.0 + gate * (1.0 - s))) * gate_ind
        dZ = np.concatenate([dGate, dUp], axis=1)
        dXr = dZ @ w1_e
        np.add.at(grad_x, tokens, dXr)
        grad_w1[expert] += dZ.T @ xr
        grad_w2[expert] += dO.T @ (p[:, None] * h)
    return grad_x, grad_topk, grad_w1, grad_w2


def reference_packed_intermediates(
    x: np.ndarray,
    expert_ids: np.ndarray,
    topk_weights: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    grad_y: np.ndarray,
    rank: int,
    experts_per_rank: int,
    max_recv: int,
    save_mode: str = "bf16",
    swiglu_limit: float = 0.0,
) -> dict[str, np.ndarray | int]:
    """Reference for the expert-packed BF16 buffers in backward_workspace.

    Global tokens are rank-major. Dispatch collects source ranks in rank order,
    so np.argwhere() gives the expected packed route order for each expert.
    Every GM materialization point is rounded to BF16 like the device kernel.
    """
    f = w1.shape[1] // 2
    packed_do = []
    packed_gate = []
    packed_up = []
    packed_grad_z = []
    packed_dx = []
    packed_hp = []
    packed_dp_partials = []
    packed_routes = []
    counts = []

    for local_expert in range(experts_per_rank):
        expert = rank * experts_per_rank + local_expert
        routes = np.argwhere(expert_ids == expert)
        remaining = max_recv - sum(counts)
        routes = routes[: max(remaining, 0)]
        counts.append(int(routes.shape[0]))
        if routes.size == 0:
            continue

        tokens = routes[:, 0]
        slots = routes[:, 1]
        xr = x[tokens].astype(np.float32)
        p = topk_weights[tokens, slots].astype(np.float32)
        d_o = bf16_round(grad_y[tokens].astype(np.float32))

        z = bf16_round(xr @ w1[expert].astype(np.float32).T)
        raw_gate, raw_up = np.split(z, 2, axis=1)
        gate, up = raw_gate, raw_up
        if save_mode == "mxfp8":
            from saved_gate_up_reference import w8a8_reference

            _, gate, _, _ = w8a8_reference().mxfp8_quantize(gate, -1)
            _, up, _, _ = w8a8_reference().mxfp8_quantize(up, -1)
        # Saved planes contain raw/dequantized values before activation clamps.
        # Keep their reference separate from the clipped gradient operands.
        saved_gate, saved_up = gate, up
        gate_ind = np.ones_like(gate, dtype=np.float32)
        up_ind = np.ones_like(up, dtype=np.float32)
        if swiglu_limit > 0:
            limit = np.float32(swiglu_limit)
            gate_high = raw_gate > limit
            up_low = raw_up < -limit
            up_high = raw_up > limit
            gate = np.minimum(gate, limit)
            up = np.clip(up, -limit, limit)
            if save_mode == "mxfp8":
                gate = np.where(gate_high, limit, gate)
                up = np.where(up_low, -limit, np.where(up_high, limit, up))
            gate_ind = (~gate_high).astype(np.float32)
            up_ind = (~(up_low | up_high)).astype(np.float32)
        sigmoid = 1.0 / (1.0 + np.exp(-gate))
        # Match the forward epilogue's gate / (1 + exp(-gate)), keeping
        # SiLU and the Up product in FP32 until the BF16 H boundary.
        silu = gate / (1.0 + np.exp(-gate))
        h = bf16_round(silu * up)
        hp = bf16_round(p[:, None] * h)

        d_hp = bf16_round(d_o @ w2[expert].astype(np.float32))
        # Reconstructed H must be the same BF16 activation as forward H.
        # Keep dH's BF16 tensor boundary, but no standalone BF16 SiLU/dS.
        dp_products = bf16_round(d_hp * h)
        d_h = bf16_round(p[:, None] * d_hp)
        d_up = d_h * silu * up_ind
        d_gate = d_h * up * (sigmoid * (1.0 + gate * (1.0 - sigmoid))) * gate_ind
        grad_z = bf16_round(np.concatenate([d_gate, d_up], axis=1))

        packed_do.append(d_o)
        packed_gate.append(saved_gate)
        packed_up.append(saved_up)
        packed_grad_z.append(grad_z)
        # Saved modes replace each expert's retired dO plane with dX.
        packed_dx.append(bf16_round(grad_z @ w1[expert].astype(np.float32)))
        packed_hp.append(hp)
        # The optimized epilogue reduces each 256-wide F tile in FP32 and
        # stores one partial per packed row; P6 sums only these partials.
        packed_dp_partials.append(dp_products.reshape(dp_products.shape[0], -1, 256).sum(axis=2, dtype=np.float32))
        packed_routes.append(
            np.column_stack(
                [
                    np.full(routes.shape[0], expert, dtype=np.int64),
                    tokens.astype(np.int64),
                    slots.astype(np.int64),
                ]
            )
        )

    def concat(parts: list[np.ndarray], width: int) -> np.ndarray:
        if not parts:
            return np.empty((0, width), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    return {
        "valid_rows": sum(counts),
        "counts": np.asarray(counts, dtype=np.int32),
        "routes": (np.concatenate(packed_routes, axis=0) if packed_routes else np.empty((0, 3), dtype=np.int64)),
        "ptrDO": concat(packed_do, x.shape[1]),
        "ptrDX": concat(packed_dx, x.shape[1]),
        "gate": concat(packed_gate, f),
        "up": concat(packed_up, f),
        "gradZ": concat(packed_grad_z, f * 2),
        "Hp": concat(packed_hp, f),
        "dpPartials": concat(packed_dp_partials, (f + 255) // 256),
    }


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def training_saved_state_layout(
    m: int,
    k: int,
    topk: int,
    world_size: int,
    expert_per_rank: int,
    max_recv: int,
    n: int,
    save_mode: str,
    swiglu_limit: float = 0.0,
) -> dict[str, int | str | bool]:
    """Python mirror of TrainingSavedStateLayout for diagnostics only."""
    resolved = save_mode
    if resolved == "auto":
        resolved = "mxfp8" if swiglu_limit <= 0 else "bf16"
    f = n // 2
    # Packed X and routed weights belong to dispatch_output, not saved state.
    offset = 0
    offset += _align_up(world_size * expert_per_rank * 4, 512)
    offset += _align_up(m, 256) * topk * 4
    offset += _align_up(m, 256) * topk * 4
    offset += _align_up(world_size * expert_per_rank, 128) * 4
    offset += _align_up(world_size * _align_up(world_size * expert_per_rank, 128) * 4, 512)
    # Packed TopK weights are also outside training_saved_state.
    progress = offset
    legacy = progress + _align_up(256 * 4, 128)
    result: dict[str, int | str | bool] = {
        "mode": resolved,
        "progress": progress,
        "legacy_bytes": legacy,
    }
    if resolved == "none":
        result["total_bytes"] = legacy
        return result
    plane_bytes = _align_up(max_recv * f * (1 if resolved == "mxfp8" else 2), 512)
    gate = _align_up(legacy, 512)
    up = gate + plane_bytes
    result.update({"gate": gate, "up": up, "plane_bytes": plane_bytes})
    if resolved == "mxfp8":
        scale_bytes = _align_up(max_recv * ((f + 31) // 32), 512)
        gate_scale = up + plane_bytes
        up_scale = gate_scale + scale_bytes
        result.update(
            {
                "gate_scale": gate_scale,
                "up_scale": up_scale,
                "scale_bytes": scale_bytes,
                "total_bytes": up_scale + scale_bytes,
            }
        )
    else:
        result["total_bytes"] = up + plane_bytes
    return result


def format_saved_state_layout(rank: int, max_recv: int, layout: dict) -> str:
    """Report Python parser offsets; this is not a Kernel layout readback."""
    fields = [
        f"[SAVED_LAYOUT] rank={rank} source=python_test_layout unit=bytes",
        f"mode={layout['mode']} max_recv={max_recv}",
        "packed_x=dispatch_output routed_weights=dispatch_output",
        f"progress={layout['progress']}",
    ]
    if layout["mode"] == "none":
        fields.append("gate=not_saved up=not_saved")
    else:
        fields.append(f"gate={layout['gate']} up={layout['up']}")
        if layout["mode"] == "mxfp8":
            fields.append(f"gate_scale={layout['gate_scale']} up_scale={layout['up_scale']}")
    return " ".join(fields)


def decode_backward_workspace(
    workspace,
    training_saved_state,
    m: int,
    max_recv: int,
    k: int,
    n: int,
    topk: int,
    world_size: int,
    expert_per_rank: int,
    save_mode: str,
    swiglu_limit: float = 0.0,
) -> dict[str, np.ndarray]:
    """Decode backward scratch and its selected gate/up source."""
    raw = workspace.detach().cpu().contiguous().numpy().tobytes()
    saved_raw = training_saved_state.detach().cpu().contiguous().numpy().tobytes()
    f = n // 2
    offset = 0
    result = {}

    def take(name: str, rows: int, cols: int) -> None:
        nonlocal offset
        count = rows * cols
        end = offset + count * 2
        if end > len(raw):
            raise RuntimeError(f"{name} exceeds backward_workspace: end={end}, size={len(raw)}")
        bits = np.frombuffer(raw, dtype=np.uint16, count=count, offset=offset)
        result[name] = bf16_bits_to_float32(bits).reshape(rows, cols).copy()
        offset += _align_up(count * 2, 512)

    def take_fp32(name: str, rows: int, cols: int) -> None:
        nonlocal offset
        count = rows * cols
        end = offset + count * 4
        if end > len(raw):
            raise RuntimeError(f"{name} exceeds backward_workspace: end={end}, size={len(raw)}")
        result[name] = np.frombuffer(raw, dtype=np.float32, count=count, offset=offset).reshape(rows, cols).copy()
        offset += _align_up(count * 4, 512)

    layout = training_saved_state_layout(
        m,
        k,
        topk,
        world_size,
        expert_per_rank,
        max_recv,
        n,
        save_mode,
        swiglu_limit,
    )
    resolved = str(layout["mode"])
    # BackwardUsesGm(savedMode) aliases ptrDX to ptrDO. After kernel return
    # this plane holds packed dX, not the earlier Dispatch result dO.
    take("ptrDO" if resolved == "none" else "ptrDX", max_recv, k)
    if resolved == "none":
        take("gate", max_recv, f)
        take("up", max_recv, f)
    elif resolved == "bf16":
        for name in ("gate", "up"):
            count = max_recv * f
            bits = np.frombuffer(
                saved_raw,
                dtype=np.uint16,
                count=count,
                offset=int(layout[name]),
            )
            result[name] = bf16_bits_to_float32(bits).reshape(max_recv, f).copy()
    else:
        from saved_gate_up_reference import w8a8_reference

        groups = (f + 31) // 32
        decoder = w8a8_reference().decode_fp8_e4m3fn
        for name in ("gate", "up"):
            data = np.frombuffer(
                saved_raw,
                dtype=np.uint8,
                count=max_recv * f,
                offset=int(layout[name]),
            ).reshape(max_recv, f)
            scales = np.frombuffer(
                saved_raw,
                dtype=np.uint8,
                count=max_recv * groups,
                offset=int(layout[f"{name}_scale"]),
            ).reshape(max_recv, groups)
            scale_values = np.exp2(scales.astype(np.float32) - 127.0)
            result[name] = (
                (decoder(data.reshape(max_recv, groups, 32)) * scale_values[..., None])
                .reshape(max_recv, f)
                .astype(np.float32)
            )
    take("gradZ", max_recv, n)
    take("Hp", max_recv, f)
    take_fp32("dpPartials", max_recv, (f + 255) // 256)
    return result


def check_packed_intermediate(
    name: str,
    got: np.ndarray,
    ref: np.ndarray,
    routes: np.ndarray,
    atol: float,
    rtol: float,
) -> bool:
    abs_err = np.abs(got - ref)
    bad = (abs_err > atol) & (abs_err / (np.abs(ref) + 1e-6) > rtol)
    bad_count = int(bad.sum())
    max_abs = float(abs_err.max()) if abs_err.size else 0.0
    passed = bad_count == 0
    print(
        f"[INTERMEDIATE] {name}: bad={bad_count}/{ref.size} max_abs_err={max_abs:.6g} {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    if not passed:
        row, col = np.argwhere(bad)[0]
        route = routes[row].tolist() if row < len(routes) else ["?", "?", "?"]
        print(
            f"[INTERMEDIATE] {name} first_bad row={row} col={col} "
            f"route=[expert,token,slot]={route} got={got[row, col]:.8g} "
            f"ref={ref[row, col]:.8g}",
            flush=True,
        )
    return passed


def _align_up_any(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def decode_expanded_row_idx_gather(
    saved_state,
    m: int,
    topk: int,
    k: int,
    world_size: int,
    expert_per_rank: int,
    max_recv: int,
) -> np.ndarray:
    """Decode expandedRowIdxGather (int32 [m*topk]) from training_saved_state."""
    raw = saved_state.detach().cpu().contiguous().numpy().tobytes()
    ptr_a_offset = 0
    cumsum_offset = ptr_a_offset  # packed X lives in dispatch_output
    expanded_row_idx_offset = cumsum_offset + _align_up_any(world_size * expert_per_rank * 4, 512)
    expanded_row_idx_gather_offset = expanded_row_idx_offset + _align_up_any(m, 256) * topk * 4
    count = m * topk
    return (
        np.frombuffer(
            raw,
            dtype=np.int32,
            count=count,
            offset=expanded_row_idx_gather_offset,
        )
        .copy()
        .reshape(m, topk)
    )


def diagnose_grad_x(
    rank: int,
    m: int,
    topk: int,
    x_global: np.ndarray,
    expert_ids_global: np.ndarray,
    topk_weights_global: np.ndarray,
    grad_y_global: np.ndarray,
    w1_global: np.ndarray,
    w2_global: np.ndarray,
    got_gx: np.ndarray,
    packed_ref: dict,
    packed_got: dict,
    gather: np.ndarray,
    save_mode: str,
    swiglu_limit: float,
) -> None:
    """Compare device grad_x against a Python reconstruction from the device
    gradZ, using (a) the reference np.argwhere routing and (b) the device's
    expandedRowIdxGather routing.  Pinpoints whether the combine/unpermute
    mapping is wrong or the GEMM output is wrong."""
    grad_z = packed_got["gradZ"]
    routes = packed_ref["routes"]
    valid_rows = int(packed_ref["valid_rows"])

    # (a) reference routing: gradZ rows are packed per local expert, rank-major.
    recon_ref_route = np.zeros_like(x_global, dtype=np.float32)
    for row in range(valid_rows):
        expert_global, token_global, _slot = routes[row]
        recon_ref_route[token_global] += grad_z[row].astype(np.float32) @ w1_global[expert_global].astype(np.float32)
    recon_local = recon_ref_route[rank * m : (rank + 1) * m]

    # (b) device expandedRowIdxGather routing: gather[r] = packed row for
    # (token=r//topk, slot=r%topk).  Reuse reference routes to map a packed
    # row back to its (expert, token) so we can accumulate the device-mapped
    # rows with the same global accumulation.
    row_to_route = {}
    for row in range(valid_rows):
        expert_global, token_global, slot = routes[row]
        row_to_route[row] = (expert_global, token_global, slot)
    recon_dev_route = np.zeros_like(x_global, dtype=np.float32)
    for token_local in range(m):
        for slot in range(topk):
            packed_row = int(gather[token_local, slot])
            entry = row_to_route.get(packed_row)
            if entry is None:
                continue
            expert_global, token_global, _s = entry
            recon_dev_route[token_global] += grad_z[packed_row].astype(np.float32) @ w1_global[expert_global].astype(
                np.float32
            )
    recon_dev_local = recon_dev_route[rank * m : (rank + 1) * m]

    ref_gx, _, _, _ = reference_backward(
        x_global,
        expert_ids_global,
        topk_weights_global,
        w1_global,
        w2_global,
        grad_y_global,
        save_mode,
        swiglu_limit,
    )
    ref_gx = ref_gx[rank * m : (rank + 1) * m]

    def stat(name: str, arr: np.ndarray) -> None:
        print(
            f"[DIAG-gradx] rank{rank} {name}: norm={np.linalg.norm(arr):.6f} max={np.abs(arr).max():.6f}",
            flush=True,
        )

    stat("got(device)", got_gx)
    stat("recon_ref_route", recon_local)
    stat("recon_dev_route", recon_dev_local)
    stat("ref_fp32", ref_gx)
    for name, arr in (
        ("got_vs_recon_ref", got_gx - recon_local),
        ("got_vs_recon_dev", got_gx - recon_dev_local),
        ("recon_ref_vs_recon_dev", recon_local - recon_dev_local),
        ("got_vs_ref_fp32", got_gx - ref_gx),
    ):
        print(
            f"[DIAG-gradx] rank{rank} {name}: diff_norm={np.linalg.norm(arr):.6f} max_abs={np.abs(arr).max():.6f}",
            flush=True,
        )
    # distribution of row-wise norm of device vs recon
    row_norm_got = np.linalg.norm(got_gx, axis=1)
    row_norm_recon = np.linalg.norm(recon_local, axis=1)
    print(
        f"[DIAG-gradx] rank{rank} row_norm ratio recon/got: "
        f"mean={np.mean(row_norm_recon / (row_norm_got + 1e-9)):.4f} "
        f"min={np.min(row_norm_recon / (row_norm_got + 1e-9)):.4f} "
        f"max={np.max(row_norm_recon / (row_norm_got + 1e-9)):.4f}",
        flush=True,
    )


def check_grad(name: str, got: np.ndarray, ref: np.ndarray, atol: float, rtol: float) -> bool:
    got = got.astype(np.float32)
    ref = ref.astype(np.float32)
    abs_err = np.abs(got - ref)
    denom = np.abs(ref) + 1e-6
    rel_err = abs_err / denom
    bad = (abs_err > atol) & (rel_err > rtol)
    total = ref.size
    bad_count = int(bad.sum())
    max_abs = float(abs_err.max())
    scale = float(np.abs(ref).max()) if ref.size else 1.0
    passed = bad_count == 0 or (bad_count / total <= 0.05 and max_abs <= max(0.1, 2.0 * scale * 0.05))
    print(
        f"[CHECK] {name}: bad={bad_count}/{total} max_abs_err={max_abs:.6g} "
        f"scale={scale:.6g} {'PASS' if passed else 'FAIL'}",
        flush=True,
    )
    return passed


def _watchdog(holder: dict, watchdog_secs: float) -> None:
    """Kernel hang watchdog.  After watchdog_secs with no completion, print
    where the rank is stuck and dump the fixed-offset bwProgress buffer
    (256 x int32, indexed by blockIdx%256)."""
    deadline = time.monotonic() + watchdog_secs
    while time.monotonic() < deadline:
        time.sleep(1.0)
        if holder.get("done", False):
            return
    print(f"[WATCHDOG] rank={holder.get('rank')} TIMEOUT after {watchdog_secs:.0f}s", flush=True)
    ws = holder.get("training_saved_state")
    if ws is None:
        print("[WATCHDOG] no training_saved_state yet — stuck before/inside the forward", flush=True)
        return
    try:
        import torch

        progress_offset = holder.get("backward_progress_offset")
        if progress_offset is None:
            raise RuntimeError("backward progress offset is unavailable")
        with torch.npu.stream(torch.npu.Stream()):
            tail = ws.narrow(0, int(progress_offset), 1024).to(device="cpu")
        prog = np.frombuffer(tail.numpy().tobytes(), dtype=np.int32)
        nz = [(int(i % 256), int(v)) for i, v in enumerate(prog) if int(v) != 0]
        print(f"[WATCHDOG] bwProgress[blockIdx%256] nonzero = {nz}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[WATCHDOG] progress dump failed: {e}", flush=True)


def _stage(rank: int, name: str, t0: float) -> float:
    """Progress + per-stage wall time. Returns the new t0 for the next stage.

    Printed unconditionally with flush so a hung stage is visible from the
    terminal (multi-node runs previously looked dead during the long CPU
    reference computation before the first 'init shmem' line).
    """
    now = time.monotonic()
    print(f"[rank={rank}] [{now - t0:8.3f}s] {name}", flush=True)
    return now


def run_rank(rank: int, case: dict, config: dict) -> None:
    import torch
    import torch_npu

    from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

    # Under torchrun (multi-node) LOCAL_RANK is the per-node device index and
    # rank is the global rank; under mp.spawn they are identical.
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(case["EP"])
    os.environ["ASCEND_MEGAMOE_SHMEM_IP_PORT"] = config["shmem_ip_port"]

    t0 = time.monotonic()
    t0 = _stage(rank, "start: imports done, setting device", t0)
    device = torch.device(f"npu:{config['devices'][local_rank]}")
    torch.npu.set_device(device)

    m = case["M"]
    n = case["N"]
    k = case["K"]
    topk = case["TOPK"]
    e = case["expertPerRank"]
    ep = case["EP"]
    total_experts = ep * e
    global_m = ep * m

    # The kernel's routing is derived from topk_ids only; weights are the same
    # across ranks for the same (global) expert.  Regenerate the global tensors
    # on every rank with the seed, then slice the local portion.
    t0 = _stage(rank, f"generating global data (global_m={global_m}, E={total_experts}) ...", t0)
    rng_global = np.random.default_rng(config["seed"])
    rng_weights = np.random.default_rng(config["seed"] + 1)
    x_global = bf16_round(rng_global.uniform(-0.1, 0.1, size=(global_m, k)).astype(np.float32))
    expert_ids_global = generate_expert_ids(rng_global, global_m, topk, total_experts, config["routing_mode"])
    logits = rng_global.uniform(-1.0, 1.0, size=(global_m, topk)).astype(np.float32)
    logits -= np.max(logits, axis=1, keepdims=True)
    topk_weights_global = np.exp(logits)
    topk_weights_global /= np.sum(topk_weights_global, axis=1, keepdims=True)
    topk_weights_global = bf16_round(topk_weights_global)
    grad_y_global = bf16_round(rng_global.uniform(-0.1, 0.1, size=(global_m, k)).astype(np.float32))
    w1_global = random_dense_bf16_weights(rng_weights, total_experts, n, k)
    w2_global = random_dense_bf16_weights(rng_weights, total_experts, k, n // 2)
    t0 = _stage(rank, "global data generated", t0)

    rank_slice = slice(rank * m, (rank + 1) * m)
    x_np = x_global[rank_slice]
    ids_np = expert_ids_global[rank_slice]
    weights_np = topk_weights_global[rank_slice]
    grad_y_np = grad_y_global[rank_slice]
    w1_np = w1_global[rank * e : (rank + 1) * e]
    w2_np = w2_global[rank * e : (rank + 1) * e]

    # Reference on global data, then take this rank's slices.
    # --skip-reference: skip the CPU golden (minutes at multi-node shapes).
    # The kernel still runs end-to-end; grad comparison is skipped.
    if config.get("skip_reference"):
        t0 = _stage(rank, "skipping CPU reference (--skip-reference)", t0)
        ref_grad_x = ref_grad_topk = ref_grad_w1 = ref_grad_w2 = None
    else:
        t0 = _stage(rank, "computing CPU reference backward (this is the long stage) ...", t0)
        ref_grad_x, ref_grad_topk, ref_grad_w1, ref_grad_w2 = reference_backward(
            x_global,
            expert_ids_global,
            topk_weights_global,
            w1_global,
            w2_global,
            grad_y_global,
            config["resolved_save_mode"],
            config["swiglu_limit"],
        )
        ref_grad_x = ref_grad_x[rank_slice]
        ref_grad_topk = ref_grad_topk[rank_slice]
        ref_grad_w1 = ref_grad_w1[rank * e : (rank + 1) * e]
        ref_grad_w2 = ref_grad_w2[rank * e : (rank + 1) * e]
        t0 = _stage(rank, "CPU reference done", t0)

    def npu(t: np.ndarray, dtype: torch.dtype) -> torch.Tensor:
        return torch.from_numpy(t).to(dtype=dtype).to(device=device).contiguous()

    x = npu(x_np, torch.bfloat16).requires_grad_(True)
    topk_ids = npu(ids_np, torch.int32)
    topk_weights = npu(weights_np, torch.bfloat16).requires_grad_(True)
    w1 = npu(w1_np, torch.bfloat16).requires_grad_(True)
    w2 = npu(w2_np, torch.bfloat16).requires_grad_(True)
    grad_y = npu(grad_y_np, torch.bfloat16)
    t0 = _stage(rank, "inputs moved to NPU", t0)

    shmem = None
    op = None
    holder = {
        "rank": rank,
        "capture_backward_workspace": config["check_intermediates"],
    }
    watchdog = threading.Thread(target=_watchdog, args=(holder, config["watchdog_secs"]), daemon=True)
    watchdog.start()
    try:
        t0 = _stage(rank, f"init shmem (ep={ep}, {config['shmem_ip_port']}) ...", t0)
        os.environ["DEEPEP_SHMEM_ENDPOINT"] = config["shmem_ip_port"]
        op = MegaMoeBF16Train(None, LOCAL_MEM_BYTES, rank=rank, world_size=ep)
        shmem = op._runtime
        t0 = _stage(rank, "shmem ok (root Session established)", t0)
        max_recv = m * ep * min(e, topk)
        saved_layout = training_saved_state_layout(
            m,
            k,
            topk,
            ep,
            e,
            max_recv,
            n,
            config["save_gmm1_output"],
            config["swiglu_limit"],
        )
        holder["backward_progress_offset"] = saved_layout["progress"]
        print(format_saved_state_layout(rank, max_recv, saved_layout), flush=True)
        t0 = _stage(rank, f"forward (m={m}, k={k}, n={n}, topk={topk}, epr={e}) ...", t0)
        y, counts = op._forward_diagnostics(
            x,
            topk_ids,
            topk_weights,
            w1,
            w2,
            max_recv,
            swiglu_limit=config["swiglu_limit"],
            save_gmm1_output=config["save_gmm1_output"],
            diagnostic_holder=holder,
        )
        t0 = _stage(rank, "forward ok, backward ...", t0)
        loss = (y * grad_y).sum()
        loss.backward()
        torch_npu.npu.synchronize()
        t0 = _stage(rank, "backward ok, checking results ...", t0)

        intermediate_ok = True
        if config["check_intermediates"]:
            print(
                f"[INFO] rank={rank} y_shape={tuple(y.shape)} "
                f"counts={counts.detach().cpu().numpy().tolist()} "
                f"loss={loss.item():.6g}",
                flush=True,
            )
            backward_workspace = holder.get("backward_workspace")
            if backward_workspace is None:
                raise RuntimeError("backward_debug did not return backward_workspace")
            packed_ref = reference_packed_intermediates(
                x_global,
                expert_ids_global,
                topk_weights_global,
                w1_global,
                w2_global,
                grad_y_global,
                rank,
                e,
                max_recv,
                config["resolved_save_mode"],
                config["swiglu_limit"],
            )
            packed_got = decode_backward_workspace(
                backward_workspace,
                holder["training_saved_state"],
                m,
                max_recv,
                k,
                n,
                topk,
                ep,
                e,
                config["save_gmm1_output"],
                config["swiglu_limit"],
            )
            valid_rows = int(packed_ref["valid_rows"])
            routes = packed_ref["routes"]
            got_counts = counts.detach().cpu().numpy().astype(np.int32)
            counts_ok = np.array_equal(got_counts, packed_ref["counts"])
            print(
                f"[INTERMEDIATE] rank{rank} counts: got={got_counts.tolist()} "
                f"ref={packed_ref['counts'].tolist()} "
                f"{'PASS' if counts_ok else 'FAIL'}",
                flush=True,
            )
            intermediate_ok &= counts_ok
            # Only none retains dO here; saved modes have overwritten it with
            # dX. Check the live value rather than a retired tensor's bytes.
            live_plane = "ptrDO" if config["resolved_save_mode"] == "none" else "ptrDX"
            live_atol, live_rtol = (0.0, 0.0) if live_plane == "ptrDO" else (0.05, 0.15)
            intermediate_ok &= check_packed_intermediate(
                f"rank{rank} {live_plane}",
                packed_got[live_plane][:valid_rows],
                packed_ref[live_plane],
                routes,
                live_atol,
                live_rtol,
            )
            for name in ("gate", "up", "gradZ", "Hp", "dpPartials"):
                intermediate_ok &= check_packed_intermediate(
                    f"rank{rank} {name}",
                    packed_got[name][:valid_rows],
                    packed_ref[name],
                    routes,
                    0.05,
                    0.15,
                )

        got_gx = x.grad.detach().cpu().float().numpy()
        got_gtopk = topk_weights.grad.detach().cpu().float().numpy()
        got_gw1 = w1.grad.detach().cpu().float().numpy()
        got_gw2 = w2.grad.detach().cpu().float().numpy()

        if config["check_intermediates"]:
            saved_state = holder.get("training_saved_state")
            if saved_state is None:
                raise RuntimeError("training_saved_state missing from diagnostic holder")
            gather = decode_expanded_row_idx_gather(saved_state, m, topk, k, ep, e, max_recv)
            diagnose_grad_x(
                rank,
                m,
                topk,
                x_global,
                expert_ids_global,
                topk_weights_global,
                grad_y_global,
                w1_global,
                w2_global,
                got_gx,
                packed_ref,
                packed_got,
                gather,
                config["resolved_save_mode"],
                config["swiglu_limit"],
            )

        ok = intermediate_ok
        if config.get("skip_reference"):
            print(
                f"[INFO] rank={rank} --skip-reference: kernel fwd+bwd completed, "
                f"grad shapes: gx={got_gx.shape} gtopk={got_gtopk.shape} "
                f"gw1={got_gw1.shape} gw2={got_gw2.shape} "
                f"gw2_nonzero={np.count_nonzero(got_gw2)}",
                flush=True,
            )
        else:
            ok &= check_grad(f"rank{rank} grad_x", got_gx, ref_grad_x, 0.05, 0.15)
            ok &= check_grad(f"rank{rank} grad_topk_weights", got_gtopk, ref_grad_topk, 0.05, 0.15)
            ok &= check_grad(f"rank{rank} grad_w1", got_gw1, ref_grad_w1, 0.05, 0.15)
            ok &= check_grad(f"rank{rank} grad_w2", got_gw2, ref_grad_w2, 0.05, 0.15)

        shmem.barrier()
        print(f"[RESULT] rank={rank} {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            raise SystemExit(f"rank {rank} backward precision check failed")
    finally:
        holder["done"] = True
        if op is not None:
            op.destroy()
        if shmem is not None:
            shmem.close()


def parse_devices(value: str) -> list[int]:
    devices = [int(item) for item in value.split(",")]
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("devices must be comma-separated integers")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("devices must be unique")
    return devices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", type=parse_devices, nargs="?", default="0,1", help="for example: 0,1")
    # --m/--k/--n are the canonical short names; --seq/--hidden/--gmm-n are
    # torchrun-safe aliases (--m/--n are ambiguous prefixes of torchrun's own
    # --master_addr/--nnodes/etc, so launchers must use the long names).
    parser.add_argument("--m", "--seq", dest="m", type=int, default=64)
    parser.add_argument("--k", "--hidden", dest="k", type=int, default=512)
    parser.add_argument("--n", "--gmm-n", dest="n", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=TOPK)
    parser.add_argument("--expert-per-rank", type=int, default=EXPERTS_PER_RANK)
    parser.add_argument("--routing-mode", default="balanced", choices=["balanced", "random", "imbalanced"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--save-gmm1-output",
        default="bf16",
        choices=["none", "bf16", "mxfp8", "auto"],
        help="forward saved gate/up format consumed by backward",
    )
    parser.add_argument("--swiglu-limit", type=float, default=0.0)
    parser.add_argument("--shmem-ip-port", default="tcp://127.0.0.1:8899")
    parser.add_argument("--mode", default="precision", choices=["precision"])
    parser.add_argument(
        "--check-intermediates",
        action="store_true",
        help=("retain backward_workspace and compare ptrDO/gate/up/gradZ/Hp/dpPartials"),
    )
    parser.add_argument(
        "--skip-reference",
        action="store_true",
        help=(
            "skip the CPU reference backward (minutes at large/multi-node "
            "shapes) and run the kernel only: no grad comparison, but the "
            "full shmem/forward/backward pipeline still executes. Use for "
            "link bring-up and perf runs; add --check-intermediates for "
            "per-tensor diagnostics without the global reference"
        ),
    )
    parser.add_argument(
        "--watchdog-secs",
        type=float,
        default=120.0,
        help="on hang: after this many seconds dump the kernel bwProgress buffer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = args.devices
    case = {
        "M": args.m,
        "K": args.k,
        "N": args.n,
        "TOPK": args.topk,
        "expertPerRank": args.expert_per_rank,
        "EP": len(devices),
    }
    config = {
        "devices": devices,
        "shmem_ip_port": args.shmem_ip_port,
        "seed": args.seed,
        "routing_mode": args.routing_mode,
        "check_intermediates": args.check_intermediates,
        "skip_reference": args.skip_reference,
        "watchdog_secs": args.watchdog_secs,
        "save_gmm1_output": args.save_gmm1_output,
        "swiglu_limit": args.swiglu_limit,
        "resolved_save_mode": (
            "mxfp8"
            if args.save_gmm1_output == "auto" and args.swiglu_limit <= 0
            else "bf16"
            if args.save_gmm1_output == "auto"
            else args.save_gmm1_output
        ),
    }
    if case["K"] % 256 != 0 or case["N"] % 512 != 0:
        print("[ERROR] requires K % 256 == 0 and N % 512 == 0", file=sys.stderr)
        return 1
    if args.topk > case["EP"] * case["expertPerRank"]:
        print(f"[ERROR] TOPK {args.topk} exceeds total experts", file=sys.stderr)
        return 1

    # Mirror the forward test: the SHMEM transport needs the device list and a
    # session id derived from the (tcp://) shmem URL to init aclshmemx.
    # If the caller already restricted ASCEND_RT_VISIBLE_DEVICES (e.g. to idle
    # physical cards), keep that mapping; --devices then refers to logical indices.
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", ",".join(str(device) for device in devices))
    os.environ["SHMEM_UID_SESSION_ID"] = args.shmem_ip_port.removeprefix("tcp://")

    # torchrun mode (single- or multi-node): RANK/WORLD_SIZE/LOCAL_RANK are
    # preset by the launcher, devices is the per-node local list, and EP comes
    # from WORLD_SIZE so a 2x8 deployment passes devices=0,...,7 on each node
    # while the symmetric window spans all 16 ranks.
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        case["EP"] = world
        if args.topk > world * args.expert_per_rank:
            print(f"[ERROR] TOPK {args.topk} exceeds total experts", file=sys.stderr)
            return 1
        run_rank(rank, case, config)
        if rank == 0:
            print("[SUCCESS] MegaMoe BF16 backward precision test passed", flush=True)
        return 0

    import torch.multiprocessing as mp

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    mp.spawn(run_rank, args=(case, config), nprocs=case["EP"], join=True)
    print("[SUCCESS] MegaMoe BF16 backward precision test passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

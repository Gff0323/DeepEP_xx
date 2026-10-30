#!/usr/bin/env python3
"""MegaMoe BF16 training backward precision test.

Each EP rank runs the differentiable MegaMoeBF16Train module, builds a scalar loss
from the output and calls loss.backward().
grad_x / grad_weight1 / grad_weight2 are compared against a float32 CPU
reference that replays the same routing on the global data.

Usage:
    python test_mega_moe_backward.py 0,1 --seq 129 --hidden 256 --gmm-n 1024 \
        --topk 4 --expert-per-rank 2
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
    """Report a stalled rank without accessing device memory."""
    deadline = time.monotonic() + watchdog_secs
    while time.monotonic() < deadline:
        time.sleep(1.0)
        if holder.get("done", False):
            return
    print(f"[WATCHDOG] rank={holder.get('rank')} TIMEOUT after {watchdog_secs:.0f}s", flush=True)


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
    from device_support import create_ep_group

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
    ep_group = None
    holder = {
        "rank": rank,
    }
    watchdog = threading.Thread(target=_watchdog, args=(holder, config["watchdog_secs"]), daemon=True)
    watchdog.start()
    try:
        t0 = _stage(rank, f"init shmem (ep={ep}, {config['shmem_ip_port']}) ...", t0)
        os.environ["DEEPEP_SHMEM_ENDPOINT"] = config["shmem_ip_port"]
        ep_group = create_ep_group(rank, ep, config["shmem_ip_port"])
        op = MegaMoeBF16Train(ep_group, LOCAL_MEM_BYTES)
        shmem = op._runtime
        t0 = _stage(rank, "shmem ok (root Session established)", t0)
        max_recv = m * ep * min(e, topk)
        t0 = _stage(rank, f"forward (m={m}, k={k}, n={n}, topk={topk}, epr={e}) ...", t0)
        y, counts = op(
            x,
            topk_ids,
            topk_weights,
            w1,
            w2,
            max_recv,
            swiglu_limit=config["swiglu_limit"],
            save_gmm1_output=config["save_gmm1_output"],
        )
        t0 = _stage(rank, "forward ok, backward ...", t0)
        loss = (y * grad_y).sum()
        loss.backward()
        torch_npu.npu.synchronize()
        t0 = _stage(rank, "backward ok, checking results ...", t0)

        got_gx = x.grad.detach().cpu().float().numpy()
        got_gtopk = topk_weights.grad.detach().cpu().float().numpy()
        got_gw1 = w1.grad.detach().cpu().float().numpy()
        got_gw2 = w2.grad.detach().cpu().float().numpy()

        got_counts = counts.detach().cpu().numpy().astype(np.int64)
        expected_counts = np.bincount(expert_ids_global.reshape(-1), minlength=total_experts)[rank * e : (rank + 1) * e]
        ok = bool(np.array_equal(got_counts, expected_counts))
        print(f"[CHECK] rank{rank} expert_counts: {'PASS' if ok else 'FAIL'}", flush=True)
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
            op.close()
        if ep_group is not None:
            ep_group.close()


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
        "--skip-reference",
        action="store_true",
        help=(
            "skip the CPU reference backward (minutes at large/multi-node "
            "shapes) and run the kernel only: no grad comparison, but the "
            "full shmem/forward/backward pipeline still executes. Use for "
            "link bring-up and perf runs."
        ),
    )
    parser.add_argument(
        "--watchdog-secs",
        type=float,
        default=120.0,
        help="report a stalled rank after this many seconds",
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

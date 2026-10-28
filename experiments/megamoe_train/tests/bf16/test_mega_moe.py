#!/usr/bin/env python3
"""MegaMoe BF16 precision test with W8A8-aligned NPU profiling."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import sys
import time


# ==================== 物理机配置：单机 / 多机共用同一个入口 ====================
SERVER_NUM = 1  # 机器总数：单机 1、双机 2、四机 4；所有机器保持一致。
SERVER_INDEX = 0  # 主机为 0；其他机器依次为 1、2、3…，直到 SERVER_NUM-1；单机为 0。
MASTER_IP = "127.0.0.1"  # 多机时，所有机器都填主机（SERVER_INDEX=0）可访问的实际 IP。
SHMEM_PORT = 8899  # SHMEM bootstrap；也可用 --shmem-ip-port 覆盖。
CONTROL_PORT = 8900  # 多机测试协调端口，不能与 SHMEM_PORT 相同。
TIMEOUT_SECONDS = 1800  # 多机协调等待上限，包含 CPU golden 等待时间。
# 设备列表、shape、precision/performance、循环数仍由原来的 sh 命令参数指定。
# ============================================================================


TEST_DIR = Path(__file__).resolve().parent
DEVICE_LARGE_PAGE_SIZE = 2 * 1024 * 1024
BUFFER_ALIGNMENT = 512
ACLSHMEM_DATA_OP_UDMA = 0x08
ACLSHMEM_DATA_OP_SDMA = 0x02
TOPK = 16
EXPERTS_PER_RANK = 4
BASE_LOCAL_MEM_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_SEED = 831
DEFAULT_WARMUP = 0
DEFAULT_LOOPS = 110
PERFORMANCE_TAIL_LOOPS = 50
DEFAULT_ABSOLUTE_TOLERANCE = 0.01
DEFAULT_RELATIVE_TOLERANCE = 0.01
DEFAULT_MAX_ERROR_RATIO = 0.05


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def recv_capacity(m: int, ep: int, routing_mode: str) -> int:
    if routing_mode == "balanced":
        # Reserve the sum of per-expert upper bounds for non-divisible counts.
        rows_per_expert = (m * TOPK + EXPERTS_PER_RANK - 1) // EXPERTS_PER_RANK
        return align_up(rows_per_expert * EXPERTS_PER_RANK, 128)
    if routing_mode in ("random", "imbalanced"):
        return align_up(
            m * ep * min(EXPERTS_PER_RANK, TOPK),
            128,
        )
    raise ValueError(f"unsupported routing mode: {routing_mode}")


def local_mem_bytes(k: int, *, m: int, ep: int) -> int:
    # Match the BF16 PeermemInfo layout: control/counts, ptrA0,
    # routed TopK weights, and ptrD. Larger hidden sizes exceed the old 1 GiB.
    # Ordinary BF16 always reserves 64 KiB of rank-sync slots and starts
    # payload at 72 KiB.
    required_bytes = 72 * 1024
    required_bytes += ep * align_up(ep * EXPERTS_PER_RANK, 128) * 4
    required_bytes += 2 * m * TOPK * k * 2
    required_bytes += align_up(m * TOPK * 2, BUFFER_ALIGNMENT)
    return align_up(max(BASE_LOCAL_MEM_BYTES, required_bytes), DEVICE_LARGE_PAGE_SIZE)


def read_cases(
    csv_path: Path,
    ep: int,
    routing_mode: str = "balanced",
) -> list[dict]:
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        expected_fields = ["M", "N", "K"]
        if reader.fieldnames != expected_fields:
            raise ValueError(f"{csv_path} must contain only M,N,K; found {reader.fieldnames}")
        cases = []
        for line_number, row in enumerate(reader, start=2):
            try:
                shape = {name: int(row[name]) for name in expected_fields}
            except (TypeError, ValueError) as error:
                raise ValueError(f"{csv_path}:{line_number} contains a non-integer shape") from error
            if any(value <= 0 for value in shape.values()):
                raise ValueError(f"{csv_path}:{line_number} requires positive M,N,K")
            if shape["K"] % 256 != 0 or shape["N"] % 512 != 0:
                raise ValueError(f"{csv_path}:{line_number} requires K % 256 == 0 and N % 512 == 0")
            max_recv_tokens = recv_capacity(shape["M"], ep, routing_mode)
            cases.append(
                {
                    **shape,
                    "TOPK": TOPK,
                    "expertPerRank": EXPERTS_PER_RANK,
                    "EP": ep,
                    "maxRecvTokenNum": max_recv_tokens,
                    "validateRecvCapacity": True,
                    "localMemBytes": local_mem_bytes(
                        shape["K"],
                        m=shape["M"],
                        ep=ep,
                    ),
                }
            )
    if not cases:
        raise ValueError(f"{csv_path} does not contain any cases")
    return cases


def parse_devices(value: str) -> list[int]:
    try:
        devices = [int(item) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("devices must be comma-separated integers") from error
    if not devices or any(device < 0 for device in devices):
        raise argparse.ArgumentTypeError("device IDs must be non-negative")
    if len(set(devices)) != len(devices):
        raise argparse.ArgumentTypeError("device IDs must be unique")
    return devices


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("devices", type=parse_devices, help="for example: 0,1")
    parser.add_argument("--topk", type=int, default=TOPK, help="路由 TopK（默认保持 16）")
    parser.add_argument("--expert-per-rank", type=int, default=EXPERTS_PER_RANK, help="每卡专家数（默认保持 4）")
    parser.add_argument(
        "--save-gmm1-output",
        choices=("auto", "none", "bf16", "mxfp8"),
        default="bf16",
        help="saved gate/up mode (default: bf16, saving enabled)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=TEST_DIR / "mega_moe_cases.csv",
    )
    parser.add_argument(
        "--mode",
        choices=("both", "precision", "performance"),
        default="both",
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument(
        "--data-cache-dir",
        type=Path,
        default=TEST_DIR / "output" / "data_cache",
        help="persistent input/golden cache; shared by repeated runs (default: output/data_cache)",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="regenerate inputs/golden even when the persistent cache is valid",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--routing-mode",
        choices=("balanced", "random", "imbalanced"),
        default="balanced",
        help=(
            "expert-ID generation: balanced keeps the performance baseline; "
            "random samples experts uniformly; imbalanced uses a seeded "
            "skewed distribution (default: balanced)"
        ),
    )
    parser.add_argument(
        "--routing-skew",
        type=float,
        default=1.0,
        help=(
            "positive log-weight range used by --routing-mode imbalanced; "
            "larger values produce hotter low-ID experts (default: 1.0)"
        ),
    )
    parser.add_argument(
        "--topk-weight-mode",
        choices=("random", "uniform"),
        default="random",
        help=(
            "route-weight generation: random uses normalized random BF16 "
            "weights; uniform uses the same BF16 value for every route to "
            "diagnose weight dispatch independently (default: random)"
        ),
    )
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--loops", type=int)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=DEFAULT_ABSOLUTE_TOLERANCE,
        help=("use absolute error instead of relative deviation below this threshold (default: 0.01)"),
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=DEFAULT_RELATIVE_TOLERANCE,
        help=("mark an element incorrect when protected deviation reaches this threshold (default: 0.01)"),
    )
    parser.add_argument(
        "--max-error-ratio",
        type=float,
        default=DEFAULT_MAX_ERROR_RATIO,
    )
    parser.add_argument(
        "--print-diagnostics",
        action="store_true",
        help=("always print first/middle/last output windows, including when precision passes"),
    )
    parser.add_argument(
        "--shmem-ip-port",
        default=os.environ.get(
            "ASCEND_MEGAMOE_SHMEM_IP_PORT",
            f"tcp://{MASTER_IP}:{SHMEM_PORT}",
        ),
    )
    args = parser.parse_args(argv)
    if args.topk < 1 or args.expert_per_rank < 1:
        parser.error("--topk 和 --expert-per-rank 必须为正整数")
    return args


def file_suffix(case: dict) -> str:
    return f"{case['M']}_{case['K']}_{case['N']}_{case['EP']}.bin"


def rank_file(data_dir: Path, rank: int, name: str, case: dict) -> Path:
    return data_dir / f"rank{rank}_{name}_{file_suffix(case)}"


def shape_label(case: dict) -> str:
    return f"M={case['M']} N={case['N']} K={case['K']} EP={case['EP']}"


def shape_directory_name(case: dict) -> str:
    return f"M{case['M']}_N{case['N']}_K{case['K']}_EP{case['EP']}"


def load_numpy(path: Path, dtype, shape):
    import numpy as np

    array = np.fromfile(path, dtype=dtype)
    expected = 1
    for dimension in shape:
        expected *= dimension
    if array.size != expected:
        raise RuntimeError(f"{path} contains {array.size} elements, expected {expected}")
    return array.reshape(shape).copy()


def check_saved_gate_up(
    rank, case, data_dir, state, counts, gate_offset, up_offset, gate_scale_offset=None, up_scale_offset=None
):
    """Sample raw GMM1 outputs after timing, without downloading full planes."""
    import numpy as np
    import torch
    from mega_moe_bf16_golden import bf16_bits_to_float32, bf16_round

    k, n, capacity = case["K"], case["N"], case["maxRecvTokenNum"]
    f = n // 2
    compressed = gate_scale_offset is not None
    blocks = sorted({0, (f // 2) // 32, f // 32 - 1})
    columns = [group * 32 + i for group in blocks for i in range(32)] if compressed else sorted({0, f // 2, f - 1})
    weight_columns = columns + [f + column for column in columns]
    weights = np.memmap(
        rank_file(data_dir, rank, "weight1_bf16", case), mode="r", dtype=np.uint16, shape=(case["expertPerRank"], n, k)
    )

    def read_bf16(byte_offset, elements):
        bits = state.narrow(0, byte_offset, elements * 2).cpu().view(torch.uint16).numpy()
        return bf16_bits_to_float32(bits)

    row_base = 0
    checked_rows = 0
    for expert, count in enumerate(counts.detach().cpu().tolist()):
        rows = min(int(count), max(0, capacity - row_base))
        if rows:
            selected = sorted({0, rows // 2, rows - 1})
            x = np.stack([read_bf16((row_base + row) * k * 2, k) for row in selected])
            w = bf16_bits_to_float32(weights[expert, weight_columns, :])
            expected = bf16_round(x @ w.T)
            if compressed:
                from saved_gate_up_reference import compare_mxfp8_blocks

                for plane, (offset, scale_offset) in enumerate(
                    ((gate_offset, gate_scale_offset), (up_offset, up_scale_offset))
                ):
                    raw = np.stack(
                        [state.narrow(0, offset + (row_base + row) * f, f).cpu().numpy()[columns] for row in selected]
                    ).reshape(len(selected), len(blocks), 32)
                    scales = np.stack(
                        [
                            state.narrow(0, scale_offset + (row_base + row) * (f // 32), f // 32).cpu().numpy()[blocks]
                            for row in selected
                        ]
                    )
                    target = expected[:, plane * len(columns) : (plane + 1) * len(columns)].reshape(
                        len(selected), len(blocks), 32
                    )
                    # The reference helper treats the final axis as one complete group.
                    compare_mxfp8_blocks(
                        raw,
                        scales,
                        target,
                        context=(
                            f"rank={rank} expert={expert} plane={'gate' if plane == 0 else 'up'} "
                            f"packed_rows={[row_base + row for row in selected]} "
                            f"group_start_columns={[group * 32 for group in blocks]}"
                        ),
                    )
                checked_rows += len(selected)
                row_base += rows
                continue
            actual = np.stack(
                [
                    np.concatenate(
                        [
                            read_bf16(gate_offset + (row_base + row) * f * 2, f)[columns],
                            read_bf16(up_offset + (row_base + row) * f * 2, f)[columns],
                        ]
                    )
                    for row in selected
                ]
            )
            if not np.allclose(
                actual, expected, rtol=DEFAULT_RELATIVE_TOLERANCE, atol=DEFAULT_ABSOLUTE_TOLERANCE, equal_nan=False
            ):
                raise RuntimeError(
                    f"rank {rank} expert {expert}: saved raw gate/up mismatch; "
                    f"max_abs_error={np.max(np.abs(actual - expected))}"
                )
            checked_rows += len(selected)
        row_base += rows
    print(
        f"[SAVED_GATE_UP] rank={rank} sampled_rows={checked_rows} "
        f"format={'MXFP8_E4M3_E8M0' if compressed else 'BF16'} "
        f"columns_per_plane={len(columns)} PASS",
        flush=True,
    )


def load_inputs(rank: int, case: dict, data_dir: Path, device):
    import numpy as np
    import torch

    m = case["M"]
    n = case["N"]
    k = case["K"]
    topk = case["TOPK"]
    experts = case["expertPerRank"]

    def npu_tensor(name, dtype, shape, torch_dtype=None):
        tensor = torch.from_numpy(load_numpy(rank_file(data_dir, rank, name, case), dtype, shape))
        if torch_dtype is not None:
            tensor = tensor.view(torch_dtype)
        return tensor.to(device=device, non_blocking=False).contiguous()

    return (
        npu_tensor("input_x_bf16", np.uint16, (m, k), torch.bfloat16),
        npu_tensor("input_expert_ids_bf16", np.int32, (m, topk)),
        npu_tensor(
            "input_expert_weights_bf16",
            np.uint16,
            (m, topk),
            torch.bfloat16,
        ),
        npu_tensor(
            "weight1_bf16",
            np.uint16,
            (experts, n, k),
            torch.bfloat16,
        ),
        npu_tensor(
            "weight2_bf16",
            np.uint16,
            (experts, k, n // 2),
            torch.bfloat16,
        ),
    )


def create_npu_profiler(torch_npu, loops: int, output_dir: Path):
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=0,
            active=loops,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir)),
        profile_memory=True,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    )


def packed_weight_reference(ids_by_rank, weights_by_rank, order_by_rank, rank: int, experts_per_rank: int):
    """Use saved source permutations; do not assume a stable device sort."""
    import numpy as np

    pieces = []
    for expert in range(rank * experts_per_rank, (rank + 1) * experts_per_rank):
        for ids, weights, order in zip(ids_by_rank, weights_by_rank, order_by_rank):
            sorted_ids = ids.reshape(-1)[order]
            pieces.append(weights.reshape(-1)[order][sorted_ids == expert])
    return np.concatenate(pieces).astype(np.uint16, copy=False)


def check_saved_packed_weights(case: dict, data_dir: Path, ranks=None) -> None:
    import numpy as np

    ids, weights, orders = [], [], []
    routes = case["M"] * case["TOPK"]
    for src in range(case["EP"]):
        ids.append(load_numpy(rank_file(data_dir, src, "input_expert_ids_bf16", case), np.int32, (routes,)))
        weights.append(load_numpy(rank_file(data_dir, src, "input_expert_weights_bf16", case), np.uint16, (routes,)))
        order = load_numpy(rank_file(data_dir, src, "output_saved_row_idx", case), np.int32, (routes,))
        if not np.array_equal(np.sort(order), np.arange(routes)):
            raise RuntimeError(f"rank {src}: saved expandedRowIdx is not a permutation")
        sorted_ids = ids[-1][order]
        if np.any(sorted_ids[1:] < sorted_ids[:-1]):
            raise RuntimeError(f"rank {src}: saved expandedRowIdx is not expert-sorted")
        orders.append(order)
    for rank in range(case["EP"]) if ranks is None else ranks:
        expected = packed_weight_reference(ids, weights, orders, rank, case["expertPerRank"])
        if expected.size > case["maxRecvTokenNum"]:
            raise RuntimeError(f"rank {rank}: saved-weight reference exceeds receive capacity")
        actual = load_numpy(
            rank_file(data_dir, rank, "output_saved_packed_weights_bf16", case), np.uint16, (case["maxRecvTokenNum"],)
        )[: expected.size]
        bad = np.flatnonzero(actual != expected)
        if bad.size:
            row = int(bad[0])
            raise RuntimeError(
                f"rank {rank}: saved packed weights mismatch at row {row}: "
                f"got bits=0x{int(actual[row]):04x}, expected=0x{int(expected[row]):04x}"
            )
        print(f"[SAVED_STATE] rank={rank} packed_topk_weights PASS rows={expected.size}", flush=True)


def run_rank(local_rank: int, case: dict, config: dict) -> None:
    rank = config.get("rank_start", 0) + local_rank
    import numpy as np
    import torch
    import torch_npu

    from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(case["EP"])
    os.environ["ASCEND_MEGAMOE_SHMEM_IP_PORT"] = config["shmem_ip_port"]

    # Runtime visibility is sorted; retain the user-provided physical rank order.
    logical_device = sorted(config["devices"]).index(config["devices"][local_rank])
    device = torch.device(f"npu:{logical_device}")
    torch.npu.set_device(device)
    data_dir = Path(config["data_dir"])
    result_dir = data_dir / "python_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    inputs = load_inputs(rank, case, data_dir, device)

    shmem = None
    op = None
    try:
        os.environ["DEEPEP_SHMEM_ENDPOINT"] = config["shmem_ip_port"]
        op = MegaMoeBF16Train(None, case["localMemBytes"], rank=rank, world_size=case["EP"])
        shmem = op._runtime
        save_mode = config.get("save_gmm1_output", "bf16")
        print(f"[SAVED_GATE_UP] rank={rank} requested_mode={save_mode}", flush=True)

        for _ in range(config["warmup"]):
            op._compute_reference(*inputs, case["maxRecvTokenNum"], save_mode)
        torch_npu.npu.synchronize()

        y = None
        expert_token_nums = None
        training_saved_state = None
        for repeat_id in range(config["repeat"]):
            profile_context = None
            if config["enable_profiler"]:
                profile_context = create_npu_profiler(
                    torch_npu,
                    config["loops"],
                    result_dir / "profiling" / config["profile_session"] / f"rank_{rank}" / f"repeat_{repeat_id}",
                )
                profile_context.__enter__()
            try:
                for loop_idx in range(config["loops"]):
                    result = op._compute_reference(*inputs, case["maxRecvTokenNum"], save_mode)
                    (y, expert_token_nums, training_saved_state) = result
                    if profile_context is not None:
                        # The NPU launch is asynchronous.  Complete the final
                        # launch before leaving the profiler's active window;
                        # otherwise op_summary may contain a truncated final
                        # Task Duration(us) record.
                        if loop_idx + 1 == config["loops"]:
                            torch_npu.npu.synchronize()
                        profile_context.step()
            finally:
                torch_npu.npu.synchronize()
                if profile_context is not None:
                    profile_context.__exit__(None, None, None)

            if profile_context is not None:
                print(
                    f"[PERF] MegaMoe BF16 profiler collection completed "
                    f"{shape_label(case)} rank={rank} repeat={repeat_id} "
                    f"loops={config['loops']}",
                    flush=True,
                )

        y.detach().cpu().contiguous().view(torch.uint16).numpy().tofile(
            rank_file(data_dir, rank, "output_y_bf16", case)
        )
        expert_token_nums.detach().cpu().contiguous().numpy().astype(
            np.int32,
            copy=False,
        ).tofile(rank_file(data_dir, rank, "output_expert_token_nums_bf16", case))

        def align_up(value, alignment):
            return (value + alignment - 1) // alignment * alignment

        per_rank_stride = align_up(case["EP"] * case["expertPerRank"], 128)
        legacy_saved_state_bytes = (
            align_up(case["maxRecvTokenNum"] * case["K"] * 2, 512)
            + align_up(case["EP"] * case["expertPerRank"] * 4, 512)
            + align_up(case["M"], 256) * case["TOPK"] * 4 * 2
            + align_up(case["EP"] * case["expertPerRank"], 128) * 4
            + align_up(case["EP"] * per_rank_stride * 4, 512)
            + align_up(case["maxRecvTokenNum"] * 2, 512)
            + align_up(256 * 4, 128)
        )
        gate_offset = align_up(legacy_saved_state_bytes, 512)
        # Raw compute() has limit=0, so auto selects MXFP8.
        compressed = save_mode in ("auto", "mxfp8")
        plane_bytes = align_up(case["maxRecvTokenNum"] * (case["N"] // 2) * (1 if compressed else 2), 512)
        up_offset = gate_offset + plane_bytes
        scale_bytes = align_up(case["maxRecvTokenNum"] * (case["N"] // 64), 512) if compressed else 0
        gate_scale_offset = up_offset + plane_bytes if compressed else None
        up_scale_offset = gate_scale_offset + scale_bytes if compressed else None
        expected_training_saved_state_bytes = (
            legacy_saved_state_bytes if save_mode == "none" else up_offset + plane_bytes + 2 * scale_bytes
        )
        if (
            training_saved_state.dtype != torch.uint8
            or training_saved_state.numel() != expected_training_saved_state_bytes
        ):
            raise RuntimeError(
                f"rank {rank}: training_saved_state must be a uint8 "
                f"tensor with {expected_training_saved_state_bytes} bytes, "
                f"got dtype={training_saved_state.dtype} "
                f"bytes={training_saved_state.numel()}"
            )
        if config["mode"] != "performance":
            # Small state segments only; do not copy the full packed-X allocation.
            weight_offset = legacy_saved_state_bytes - 1024 - align_up(case["maxRecvTokenNum"] * 2, 512)
            row_idx_offset = align_up(case["maxRecvTokenNum"] * case["K"] * 2, 512) + align_up(
                case["EP"] * case["expertPerRank"] * 4, 512
            )
            training_saved_state.narrow(0, weight_offset, case["maxRecvTokenNum"] * 2).cpu().view(
                torch.uint16
            ).numpy().tofile(rank_file(data_dir, rank, "output_saved_packed_weights_bf16", case))
            training_saved_state.narrow(0, row_idx_offset, case["M"] * case["TOPK"] * 4).cpu().view(
                torch.int32
            ).numpy().tofile(rank_file(data_dir, rank, "output_saved_row_idx", case))
            if save_mode != "none":
                check_saved_gate_up(
                    rank,
                    case,
                    data_dir,
                    training_saved_state,
                    expert_token_nums,
                    gate_offset,
                    up_offset,
                    gate_scale_offset,
                    up_scale_offset,
                )
            else:
                print(
                    f"[SAVED_GATE_UP] rank={rank} mode=none saved_bytes={training_saved_state.numel()} SKIP", flush=True
                )
        with (result_dir / f"rank_{rank}.json").open(
            "w",
            encoding="utf-8",
        ) as output:
            json.dump(
                {
                    "rank": rank,
                    "device": config["devices"][local_rank],
                },
                output,
                indent=2,
            )
        shmem.barrier()
    finally:
        if op is not None:
            op.destroy()
        if shmem is not None:
            shmem.close()


def write_performance_summary(case: dict, config: dict, data_dir: Path) -> list[dict]:
    def profiler_durations(rank: int) -> list[float]:
        exact_durations = []
        fallback_durations = []
        profile_root = result_dir / "profiling" / config["profile_session"] / f"rank_{rank}"
        if not profile_root.is_dir():
            return exact_durations
        # 两种文件都记录 NPU kernel 时间；不能使用 PyTorch operator_details。
        for pattern, type_key, name_key, duration_key in (
            ("op_summary*.csv", "OP Type", "OP Name", "Task Duration(us)"),
            ("kernel_details.csv", "Type", "Name", "Duration(us)"),
        ):
            exact_durations, fallback_durations = [], []
            for summary_path in sorted(profile_root.rglob(pattern)):
                with summary_path.open(newline="", encoding="utf-8-sig") as summary_file:
                    for row in csv.DictReader(summary_file):
                        op_type = row.get(type_key, "")
                        op_text = " ".join((op_type, row.get(name_key, ""), row.get("Task Type", "")))
                        normalized = op_text.lower().replace("_", "")
                        if "megamoe" not in normalized:
                            continue
                        try:
                            duration = float(row[duration_key])
                        except (KeyError, TypeError, ValueError):
                            continue
                        normalized_type = op_type.lower().replace("_", "")
                        if normalized_type == "megamoebf16":
                            exact_durations.append(duration)
                        else:
                            fallback_durations.append(duration)
            if exact_durations or fallback_durations:
                break
        durations = exact_durations or fallback_durations
        return durations[-PERFORMANCE_TAIL_LOOPS:]

    result_dir = data_dir / "python_results"
    rows = []
    for rank in config.get("local_ranks", range(case["EP"])):
        with (result_dir / f"rank_{rank}.json").open(encoding="utf-8") as input_file:
            row = json.load(input_file)
        task_durations = profiler_durations(rank)
        if not task_durations:
            raise RuntimeError(
                f"rank {rank}: MegaMoe BF16 profiler Task Duration(us) rows "
                "were not found under "
                f"{result_dir / 'profiling' / config['profile_session']}"
            )
        row.update(
            {
                "metric_source": "npu_profiler_task_duration",
                "tail_count": len(task_durations),
                "mean_us": statistics.fmean(task_durations),
                "median_us": statistics.median(task_durations),
                "min_us": min(task_durations),
                "max_us": max(task_durations),
            }
        )
        rows.append(row)

    output_path = result_dir / "performance.csv"
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "M",
                "N",
                "K",
                "rank",
                "device",
                "warmup",
                "loops",
                "repeat",
                "metric_source",
                "tail_count",
                "mean_us",
                "median_us",
                "min_us",
                "max_us",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    case["M"],
                    case["N"],
                    case["K"],
                    row["rank"],
                    row["device"],
                    config["warmup"],
                    config["loops"],
                    config["repeat"],
                    row["metric_source"],
                    row["tail_count"],
                    row["mean_us"],
                    row["median_us"],
                    row["min_us"],
                    row["max_us"],
                ]
            )

    rank_medians = [row["median_us"] for row in rows]
    rank_mins = [row["min_us"] for row in rows]
    print(
        f"[PERF] {shape_label(case)} "
        f"scope={'local' if len(rows) != case['EP'] else 'global'} "
        f"tail={PERFORMANCE_TAIL_LOOPS} "
        f"min_of_rank_medians_us={min(rank_medians):.3f} "
        f"max_of_rank_medians_us={max(rank_medians):.3f} "
        f"median_of_rank_medians_us={statistics.median(rank_medians):.3f} "
        f"min_of_rank_mins_us={min(rank_mins):.3f}",
        flush=True,
    )
    print(f"[INFO] Performance CSV: {output_path}", flush=True)
    return rows


def run_case(args, case: dict, data_dir: Path, coordinator=None) -> None:
    rank_start = getattr(args, "server_index", 0) * len(args.devices)
    local_ranks = list(range(rank_start, rank_start + len(args.devices)))

    def phase(name, action):
        return action() if coordinator is None else coordinator.phase(name, action)

    if case["localMemBytes"] % DEVICE_LARGE_PAGE_SIZE != 0:
        raise ValueError("localMemBytes must be aligned to 2 MiB")
    performance_enabled = args.mode in ("both", "performance")
    warmup = (DEFAULT_WARMUP if performance_enabled else 0) if args.warmup is None else args.warmup
    loops = (DEFAULT_LOOPS if performance_enabled else 1) if args.loops is None else args.loops
    if warmup < 0 or loops <= 0 or args.repeat <= 0:
        raise ValueError("warmup must be >= 0; loops/repeat must be > 0")
    if args.routing_skew <= 0:
        raise ValueError("routing-skew must be > 0")

    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Running MegaMoe BF16 {shape_label(case)}", flush=True)
    variant = "bf16"
    requested_token_count_exchange = os.environ.get("MEGAMOE_BF16_TOKEN_COUNT_EXCHANGE", "auto")
    token_count_exchange = requested_token_count_exchange if variant == "bf16" else "dense(fixed)"
    print(
        f"[CAPACITY] variant={variant} routing_mode={args.routing_mode} "
        f"token_count_exchange={token_count_exchange} "
        f"max_recv_tokens={case['maxRecvTokenNum']} "
        f"shmem_bytes={case['localMemBytes']} "
        f"shmem_gib={case['localMemBytes'] / 1024**3:.3f}",
        flush=True,
    )
    from bf16_data_cache import prepare_data

    phase(
        "generate",
        lambda: prepare_data(
            case,
            data_dir,
            args.seed,
            cache_root=args.data_cache_dir,
            refresh=args.refresh_data,
            generate_golden=args.mode != "performance",
            routing_mode=args.routing_mode,
            routing_skew=args.routing_skew,
            topk_weight_mode=args.topk_weight_mode,
            early_topk_weight=True,
        ),
    )

    import torch.multiprocessing as mp

    config = {
        "save_gmm1_output": args.save_gmm1_output,
        "rank_start": rank_start,
        "local_ranks": local_ranks,
        "mode": args.mode,
        "devices": args.devices,
        "data_dir": str(data_dir),
        "warmup": warmup,
        "loops": loops,
        "repeat": args.repeat,
        "shmem_ip_port": args.shmem_ip_port,
        "enable_profiler": performance_enabled,
        "profile_session": time.strftime("%Y%m%d_%H%M%S"),
    }
    phase("compute", lambda: mp.spawn(run_rank, args=(case, config), nprocs=len(args.devices), join=True))
    if coordinator is not None and args.mode != "performance":
        coordinator.exchange_orders(case, data_dir, local_ranks)

    def validate_local():
        performance_rows = []
        if args.mode != "performance":
            check_saved_packed_weights(case, data_dir, local_ranks)

        if performance_enabled:
            performance_rows = write_performance_summary(case, config, data_dir)
        if args.mode != "performance":
            from mega_moe_bf16_golden import compare

            compare(
                case,
                data_dir,
                args.absolute_tolerance,
                args.relative_tolerance,
                args.max_error_ratio,
                print_diagnostics=args.print_diagnostics,
                ranks=local_ranks,
            )
        return performance_rows

    performance_rows = phase("validate", validate_local)
    if coordinator is not None and performance_enabled:
        coordinator.report_performance(case, performance_rows, data_dir)
    print(f"[SUCCESS] MegaMoe BF16 {shape_label(case)} passed", flush=True)


def main() -> None:
    global TOPK, EXPERTS_PER_RANK
    args = parse_args()
    TOPK, EXPERTS_PER_RANK = args.topk, args.expert_per_rank
    if SERVER_NUM < 1 or not 0 <= SERVER_INDEX < SERVER_NUM:
        raise ValueError("SERVER_NUM must be positive; SERVER_INDEX must be in [0, SERVER_NUM)")
    args.server_num, args.server_index = SERVER_NUM, SERVER_INDEX
    if SERVER_NUM > 1:
        from physical_server_support import run_multinode

        run_multinode(args, sys.modules[__name__], CONTROL_PORT, TIMEOUT_SECONDS)
        return
    cases = read_cases(
        args.csv_path,
        len(args.devices),
        args.routing_mode,
    )
    data_root = args.data_dir
    if data_root is None:
        run_name = f"run_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
        data_root = TEST_DIR / "output" / run_name
    data_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(TEST_DIR))
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(str(device) for device in sorted(args.devices))
    os.environ["SHMEM_UID_SESSION_ID"] = args.shmem_ip_port.removeprefix("tcp://")

    for case in cases:
        run_case(args, case, data_root / shape_directory_name(case))
    print(f"[SUCCESS] BF16 test data retained in: {data_root}", flush=True)


if __name__ == "__main__":
    main()

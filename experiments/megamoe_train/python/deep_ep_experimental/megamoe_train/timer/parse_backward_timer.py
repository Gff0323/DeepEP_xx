#!/usr/bin/env python3
# Copyright (c) 2026, Lu Lu
# Modified by guofanfeng 2026
"""Parse MegaMoe BF16 backward AscendTimerV2 raw per-rank dumps.

Ported from the training operator's ``test/megamoe_bf16_test/parse_backward_timer.py``
(ascend_deepep_hkq); the stage semantics are the ones documented there.

Backward writes the shared AscendTimerV2 layout.  Stages:

  Fixed slots (whole-stage bars):
    KERNEL               = 0   whole kernel (timerAll_)
    AIV_GATHER           = 1   P0-A gather grad_y -> peermem ptrA0 (all AIV)
    AIV_XSYNC_GATHER     = 2   cross-rank barrier after the gather
    AIV_RESTORE_TABLE    = 3   P0-B restore saved token-count table + SyncAll
    AIC_SPIN_DZ          = 5   AIC waits last dZ tiles + both AIVs done (P1 exit)
    AIC_SPIN_HP          = 7   AIC waits Hp/dpProducts published (P4 entry)
    AIV_XSYNC_UNPERMUTE  = 8   cross-rank barrier before P5 reads ptrD
    AIV_UNPERMUTE        = 9   P5 grad_x = sum over topk slots of ptrD
    AIV_XSYNC_DP         = 10  cross-rank barrier: ptrD readers drained
    AIV_DP_REDUCE        = 11  P6a dp = sum_F(dHp*H), route to source ranks
    AIV_XSYNC_TOPK       = 12  cross-rank barrier: remote dp writes visible
    AIV_TOPK_UNPERMUTE   = 13  P6b grad_topk from routed dp
    AIV_TABLE_RESET      = 14  re-zero the symmetric token table
    AIV_RECOMPUTE_PASS   = 16  AIV bookkeeping walk of the recompute phase
    AIV_P1_WAIT          = 17  每条 AIV lane 在 P1 内 WaitForCube 等待的累计
                               （Accumulate：end-start = 各 tile 等待之和；首个
                               tile 的等待含 P0-C 尾巴 + AIC 入口 SPIN 的 ramp，
                               故 AIV_SWIGLU_GRAD_e = 本槽等待 + epilogue 计算）
    AIC_P2_TAIL          = 18  AIC waits AIV0 consumed every dX tile (P2 exit)
    AIC_INIT             = 19  AIC wakeup-to-first-stage window
    AIV_INIT             = 20  AIV wakeup-to-first-stage window

  Dynamic per-expert slots (one block per expert, forward-style):
    AIV_DISPATCH_e     P0-B dispatch of this expert's dO band (AIV0)
    AIC_RECOMPUTE_e    P0-C Z = Xpacked @ W1.T (AIC)
    AIC_DO_WAIT_e      P1 entry: wait for this expert's dO band (AIC)
    AIC_GMM2GRAD_e     P1 dH = dO @ W2 (AIC)
    AIV_SWIGLU_GRAD_e  P1 epilogue: Hp / dHp*H / dZ (AIV0 and AIV1)
    AIC_GMM1GRAD_e     P2 dX = dZ @ W1 (AIC)
    AIV_COMBINE_e      P2 send dX tiles back to source ranks (AIV0)
    AIC_DW1_e          P3 dW1 = dZ^T @ Xpacked (AIC)
    AIC_DW2_e          P4 dW2 = dO^T @ Hp (AIC)

Usage:
    # 汇总表（默认）：逐 expert 槽按"每核求和 -> 跨核取 max"聚成每阶段一行；
    # --r/--k/--f 给五个 GEMM 阶段算 TFLOPS 并对照 roofline（410.4）
    python -m deep_ep_experimental.megamoe_train.timer backward \
        --dir <profile_dir> --rank-start 0 --rank-end 0 --summary --r 32768 --k 7168 --f 8192
    # 逐 expert 明细（旧行为）
    python -m deep_ep_experimental.megamoe_train.timer backward \
        --dir <profile_dir> --summary --per-expert ...

汇总口径：
    parsed CSV 每行是一个核（core_type=AIC 立方核 / AIV 向量核），各核独立记录
    自己在每个槽位上的 [start, end] 周期对。逐 expert 槽（AIC_RECOMPUTE_0..7）
    的聚合必须是"每核先把自己的各 expert 时长加总（= 该核在此阶段的总耗时），
    再跨核取 max（最慢核 = 该阶段的墙钟贡献）"——直接把所有核的数加总是错的，
    那会把 32 个核并行分摊的 tile 重复计数。AIC_max / AIV_max 是按核型分开的
    最慢核，all_max = 两者取大；各阶段 all_max 之和可大于 KERNEL（跨核型流水
    重叠，属正常）。
"""

import argparse
import csv
import sys
from array import array
from pathlib import Path

from . import protocol as p

FIXED_NAMES = p.BACKWARD_FIXED_NAMES
BACKWARD_DYNAMIC_BASE = p.BACKWARD_DYNAMIC_BASE
DYNAMIC_NAMES = p.BACKWARD_DYNAMIC_NAMES
SHARED_FIXED_COUNT = len(p.FIXED_NAMES)
SHARED_DYNAMIC_COUNT = len(p.DYNAMIC_NAMES)
MAX_DYNAMIC_ITER = p.MAX_DYNAMIC_ITER
CORE_COUNT = p.N_CORE_COUNT
COUNTERS_PER_CORE = p.COUNTERS_PER_CORE
DYNAMIC_ITER_PER_CORE = p.DYNAMIC_ITER_PER_CORE
BASE_COUNTERS = p.N_TIMING_COUNTER
TOTAL_SIZE = p.TOTAL_BUFFER_SIZE
CYCLE_TO_US = p.CYCLE_TO_US

# 五个反向 GEMM 阶段的 FLOPs 系数（单位 R*F*K）。
GEMM_STAGE_COEF = {
    "AIC_RECOMPUTE": 4.0,   # P0-C 重算 Z = X@W1^T（2RKN = 4RFK）
    "AIC_GMM2GRAD": 2.0,    # P1 dHp = dO@W2
    "AIC_GMM1GRAD": 4.0,    # P2 dX = dZ@W1（2RNK = 4RFK）
    "AIC_DW1": 4.0,         # P3 dW1 = dZ^T@X（2NRK = 4RFK）
    "AIC_DW2": 2.0,         # P4 dW2 = dO^T@Hp
}
ROOF_TFLOPS = 410.4  # 432 峰值 × 0.95
FIXED_NAMES_SET = set(FIXED_NAMES)


def load_raw(path):
    """Read a flat int64 ``.bin`` dump, or the legacy CSV-of-integers form."""
    path = Path(path)
    values = array("q")
    if path.suffix == ".bin":
        if path.stat().st_size < TOTAL_SIZE * 8:
            raise RuntimeError(
                f"{path} has {path.stat().st_size} bytes; expected at least {TOTAL_SIZE * 8}"
            )
        with path.open("rb") as stream:
            values.fromfile(stream, TOTAL_SIZE)
        if sys.byteorder != "little":
            values.byteswap()
    else:
        with path.open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                values.extend(int(item) for item in row if item.strip())
    if len(values) < TOTAL_SIZE:
        raise RuntimeError(
            f"{path} contains {len(values)} values; expected at least {TOTAL_SIZE}"
        )
    return values[:TOTAL_SIZE]


def core_base(core_id):
    group_id, subgroup = divmod(core_id, 3)
    return group_id * COUNTERS_PER_CORE * 3 + subgroup * COUNTERS_PER_CORE


def event_pair(values, core_id, logical_index):
    index = core_base(core_id) + logical_index * 2
    start, end = values[index], values[index + 1]
    duration = (end - start) / CYCLE_TO_US if end > start else 0.0
    return start, end, duration


def parse_rank(raw_path, parsed_path, groups):
    values = load_raw(raw_path)
    dynamic_counts = values[BASE_COUNTERS:]
    max_iters = [
        max(
            dynamic_counts[
                core_id * DYNAMIC_ITER_PER_CORE
                + BACKWARD_DYNAMIC_BASE
                + type_index
            ]
            for core_id in range(CORE_COUNT)
        )
        for type_index in range(len(DYNAMIC_NAMES))
    ]

    events = [(name, index) for index, name in enumerate(FIXED_NAMES)]
    for type_index, name in enumerate(DYNAMIC_NAMES):
        for iteration in range(min(max_iters[type_index], MAX_DYNAMIC_ITER)):
            logical_index = (
                SHARED_FIXED_COUNT
                + (BACKWARD_DYNAMIC_BASE + type_index) * MAX_DYNAMIC_ITER
                + iteration
            )
            events.append((f"{name}_{iteration}", logical_index))

    with parsed_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        header = ["core_id", "group_id", "core_type", "sub_id"]
        for name, _ in events:
            header.extend(
                [
                    f"{name}_start_cycle",
                    f"{name}_end_cycle",
                    f"{name}_duration(us)",
                ]
            )
        header.extend(f"{name}_IterCount" for name in DYNAMIC_NAMES)
        writer.writerow(header)

        for core_id in range(3 * groups):
            group_id, subgroup = divmod(core_id, 3)
            core_type = "AIC" if subgroup == 0 else "AIV"
            sub_id = 0 if subgroup == 0 else subgroup - 1
            row = [core_id, group_id, core_type, sub_id]
            for _, logical_index in events:
                start, end, duration = event_pair(values, core_id, logical_index)
                row.extend([start, end, f"{duration:.2f}"])
            dynamic_base = core_id * DYNAMIC_ITER_PER_CORE
            row.extend(
                dynamic_counts[dynamic_base + BACKWARD_DYNAMIC_BASE + index]
                for index in range(len(DYNAMIC_NAMES))
            )
            writer.writerow(row)


def _stage_columns(fieldnames):
    """把 duration 列整理成 (stage, [columns], n_iter) 列表。

    固定槽名里也有带数字尾巴的（AIC_SPIN_AIV0/AIV1），必须先对 FIXED_NAMES
    精确匹配；剩下的 `NAME_<iter>` 才是逐 expert 迭代列，聚到 base 名下。
    """
    suffix = "_duration(us)"
    fixed = []
    dynamic = {}
    for column in fieldnames:
        if not column.endswith(suffix):
            continue
        name = column[: -len(suffix)]
        if name in FIXED_NAMES_SET:
            fixed.append((name, [column], 1))
            continue
        base, sep, it = name.rpartition("_")
        if sep and it.isdigit():
            dynamic.setdefault(base, {})[int(it)] = column
        else:
            fixed.append((name, [column], 1))  # 未知列按单列兜底
    ordered_dynamic = sorted(
        dynamic.items(),
        key=lambda kv: (DYNAMIC_NAMES.index(kv[0])
                        if kv[0] in DYNAMIC_NAMES else len(DYNAMIC_NAMES)),
    )
    return fixed + [
        (base, [iters[i] for i in sorted(iters)], len(iters))
        for base, iters in ordered_dynamic
    ]


def _aggregate(rows, columns):
    """每核对该阶段各 expert 列求和，再按核型取 max。

    跨核不能求和（各核并行分摊 tile，加总会重复计数）；最慢核的时长才是
    该阶段的墙钟贡献。
    """
    aic = 0.0
    aiv = 0.0
    for row in rows:
        total = sum(float(row[c]) for c in columns)
        if row["core_type"] == "AIC":
            aic = max(aic, total)
        else:
            aiv = max(aiv, total)
    return aic, aiv


def summarize(parsed_dir, rank_start, rank_end, per_expert=False, r=0, k=0, f=0):
    """打印分阶段汇总表（us）。

    默认：逐 expert 槽聚合为每阶段一行（每核求和 -> 跨核取 max）；
    per_expert=True 时改打逐 expert 明细（旧行为）。
    r/k/f > 0 时给五个 GEMM 阶段算 TFLOPS（coef*R*F*K / all_max）并对照 roofline。
    """
    for rank in range(rank_start, rank_end + 1):
        parsed_path = parsed_dir / f"rank_{rank}_megamoe_bf16_backward_timer_parsed.csv"
        if not parsed_path.exists():
            print(f"[SKIP] {parsed_path} missing", flush=True)
            continue
        with parsed_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        stages = _stage_columns(fieldnames)
        if per_expert:
            stages = [
                (f"{name}_{i}" if n_iter > 1 else name, [col], 1)
                for name, columns, n_iter in stages
                for i, col in enumerate(columns)
            ]

        kernel_us = 0.0
        swiglu_all = 0.0
        stage_stats = []
        for name, columns, n_iter in stages:
            aic, aiv = _aggregate(rows, columns)
            stage_stats.append((name, n_iter, aic, aiv))
            if name == "BW_KERNEL_TIMING":
                kernel_us = max(kernel_us, aic, aiv)
            if name == "AIV_SWIGLU_GRAD":
                swiglu_all = max(aic, aiv)

        title = ("per-expert detail" if per_expert
                 else "per-core sum over experts, then max over cores")
        print(f"\n===== rank {rank} backward stage summary (us; {title}) =====")
        print(f"{'stage':<24}{'n_e':>4}{'AIC_max':>11}{'AIV_max':>11}"
              f"{'all_max':>11}{'%KERNEL':>9}{'owner':>7}  note")
        for name, n_iter, aic, aiv in stage_stats:
            all_max = max(aic, aiv)
            if all_max <= 0.0:
                continue
            pct = 100.0 * all_max / kernel_us if kernel_us > 0 else 0.0
            owner = "AIC" if aic >= aiv else "AIV"
            note = ""
            if not per_expert:
                if name == "AIV_P1_WAIT" and swiglu_all > 0:
                    note = (f"{100.0 * all_max / swiglu_all:.0f}% of "
                            "AIV_SWIGLU_GRAD（P1 Vector 等待占比）")
                coef = GEMM_STAGE_COEF.get(name)
                if coef and r > 0 and k > 0 and f > 0:
                    tflops = coef * r * f * k / (all_max * 1e-6) / 1e12
                    note = (f"≈{tflops:.0f} TFLOPS "
                            f"({100.0 * tflops / ROOF_TFLOPS:.0f}% of "
                            f"{ROOF_TFLOPS:.0f})")
            print(f"{name:<24}{n_iter:>4}{aic:>11.1f}{aiv:>11.1f}"
                  f"{all_max:>11.1f}{pct:>8.1f}%{owner:>7}  {note}")
        if not per_expert:
            print("(AIC_* 与 AIV_* 阶段跨核型流水重叠，各阶段 all_max 之和 > "
                  "KERNEL 属正常；五个 AIC GEMM 阶段的 TFLOPS 对照 roofline 即"
                  "\u201cGMM 是否到顶\u201d)")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rank-start", type=int, default=0)
    parser.add_argument("--rank-end", type=int, required=True)
    parser.add_argument(
        "--groups",
        type=int,
        default=32,
        help="number of AIC groups (block_dim / GetCoreNumAic) to dump",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print a compact per-stage table after parsing",
    )
    parser.add_argument(
        "--per-expert",
        action="store_true",
        help="with --summary: one row per expert iteration (old behavior) "
             "instead of per-stage sums",
    )
    parser.add_argument(
        "--r", type=int, default=0,
        help="routed rows per rank (bench 均衡路由下 = m*topk)；与 --k/--f 同给时"
             "为五个 GEMM 阶段计算 TFLOPS 并对照 roofline（410.4）",
    )
    parser.add_argument("--k", type=int, default=0, help="hidden size（--f 同给时生效）")
    parser.add_argument("--f", type=int, default=0, help="ffn 中间维（= n/2）")
    parser.add_argument(
        "--raw-pattern",
        default="rank_{rank}_megamoe_bf16_backward_timer.bin",
        help="raw dump name; {rank} is substituted",
    )
    parser.add_argument(
        "--parsed-pattern",
        default="rank_{rank}_megamoe_bf16_backward_timer_parsed.csv",
        help="parsed CSV name; {rank} is substituted",
    )
    args = parser.parse_args(argv)
    output_dir = args.output_dir or args.dir / "timer"
    output_dir.mkdir(parents=True, exist_ok=True)
    parsed_ranks = []
    for rank in range(args.rank_start, args.rank_end + 1):
        raw_path = args.dir / args.raw_pattern.format(rank=rank)
        if not raw_path.exists():
            print(f"[SKIP] {raw_path} missing", flush=True)
            continue
        parsed_path = output_dir / args.parsed_pattern.format(rank=rank)
        parse_rank(raw_path, parsed_path, args.groups)
        parsed_ranks.append(rank)
        print(f"[TIMER] parsed {raw_path} -> {parsed_path}")
    if args.summary:
        for rank in parsed_ranks:
            # summarize() reads the default name, so pass the parsed dir and
            # rewrite the file name when a custom pattern was used.
            summarize_args = dict(per_expert=args.per_expert, r=args.r, k=args.k, f=args.f)
            if args.parsed_pattern == "rank_{rank}_megamoe_bf16_backward_timer_parsed.csv":
                summarize(output_dir, rank, rank, **summarize_args)
            else:
                _summarize_one(output_dir / args.parsed_pattern.format(rank=rank), rank, **summarize_args)
    return 0


def _summarize_one(parsed_path, rank, **kwargs):
    """Same table as summarize(), for a non-default parsed file name."""
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        shutil.copy(parsed_path, tmp_dir / f"rank_{rank}_megamoe_bf16_backward_timer_parsed.csv")
        summarize(tmp_dir, rank, rank, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())

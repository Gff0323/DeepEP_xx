#!/usr/bin/env python3
"""Internal physical-server coordination for test_mega_moe.py.

The existing test_mega_moe.sh / test_mega_moe.py remain the only user entry points.
The TCPStore control port exchanges configuration and status;
the operator still communicates through SHMEM on its separate bootstrap port.
"""

from __future__ import annotations

import ast
import csv
from datetime import timedelta
import hashlib
import ipaddress
import json
import os
import statistics
import time


def validate_topology(
    server_num, server_index, devices, master_ip, shmem_port, control_port, timeout, experts_per_rank=4, topk=16
):
    if server_num < 1 or not 0 <= server_index < server_num:
        raise ValueError("SERVER_NUM must be positive; SERVER_INDEX must be in [0, SERVER_NUM)")
    if not devices or len(devices) != len(set(devices)) or min(devices) < 0:
        raise ValueError("DEVICES must contain distinct non-negative local device IDs")
    address = ipaddress.ip_address(master_ip)
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise ValueError("MASTER_IP must be a concrete IPv4 address")
    if server_num > 1 and address.is_loopback:
        raise ValueError("Multi-server MASTER_IP cannot be 127.x.x.x; use server 0's reachable IP")
    if not (0 < shmem_port < 65536 and 0 < control_port < 65536):
        raise ValueError("SHMEM_PORT and CONTROL_PORT must be in 1..65535")
    if shmem_port == control_port:
        raise ValueError("SHMEM_PORT and CONTROL_PORT must differ")
    if timeout <= 0:
        raise ValueError("TIMEOUT_SECONDS must be positive")
    ep = server_num * len(devices)
    if ep < 2 or ep > 768 or ep * experts_per_rank > 1024:
        raise ValueError("Global EP exceeds the BF16 routing limits")
    if ep * experts_per_rank < topk:
        raise ValueError("Global expert count must be >= TOPK (currently 16)")
    return list(range(server_index * len(devices), (server_index + 1) * len(devices)))


class Coordinator:
    def __init__(self, store, server_num, server_index, prefix="job", runner_module=None):
        if runner_module is None:
            import test_mega_moe as runner_module
        self.runner = runner_module
        self.store = store
        self.server_num = server_num
        self.server_index = server_index
        self.prefix = prefix

    def gather(self, name, payload):
        if self.server_num == 1:
            return [payload]
        key = f"{self.prefix}/{name}"
        self.store.set(f"{key}/value/{self.server_index}", payload)
        values = [self.store.get(f"{key}/value/{i}") for i in range(self.server_num)]
        # Server 0 owns the TCPStore. Before it can leave a final phase, all
        # clients must have finished reading; otherwise store teardown races get().
        self.store.set(f"{key}/read/{self.server_index}", b"1")
        if self.server_index == 0:
            self.store.wait([f"{key}/read/{i}" for i in range(self.server_num)])
        return values

    def phase(self, name, action):
        result = None
        error = None
        try:
            result = action()
        except Exception as exc:
            error = exc
        status = {"server": self.server_index, "error": None if error is None else f"{type(error).__name__}: {error}"}
        statuses = [json.loads(value) for value in self.gather(f"status/{name}", json.dumps(status).encode())]
        failures = [item for item in statuses if item["error"] is not None]
        if failures:
            raise RuntimeError(f"Physical BF16 phase={name} failed: {failures}") from error
        print(f"[MULTINODE] server={self.server_index} phase={name} all_servers=PASS", flush=True)
        return result

    def check_manifest(self, manifest):
        manifests = [
            json.loads(value) for value in self.gather("manifest", json.dumps(manifest, sort_keys=True).encode())
        ]
        if any(item != manifests[0] for item in manifests):
            raise ValueError(f"Server configurations differ; no kernel launched: {manifests}")

    def report_performance(self, case, local_rows, data_dir):
        rows = [
            row for value in self.gather("performance", json.dumps(local_rows).encode()) for row in json.loads(value)
        ]

        def report():
            if sorted(row["rank"] for row in rows) != list(range(case["EP"])):
                raise ValueError("Global performance rows do not cover each rank exactly once")
            rows.sort(key=lambda row: row["rank"])
            with (data_dir / "python_results" / "performance_global.csv").open(
                "w", newline="", encoding="utf-8"
            ) as output:
                writer = csv.DictWriter(output, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            medians = [row["median_us"] for row in rows]
            print(
                f"[PERF][GLOBAL] {self.runner.shape_label(case)} ranks={len(rows)} "
                f"median_of_rank_medians_us={statistics.median(medians):.3f} "
                f"max_of_rank_medians_us={max(medians):.3f}",
                flush=True,
            )

        self.phase("global_performance", report)


def source_digest(path):
    # SERVER_INDEX and MASTER_IP are deliberately edited independently on each
    # host. Compare executable test logic, excluding the editable topology block.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    config_names = {"SERVER_NUM", "SERVER_INDEX", "MASTER_IP", "SHMEM_PORT", "CONTROL_PORT", "TIMEOUT_SECONDS"}
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in config_names for target in node.targets)
        )
    ]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def make_manifest(args, cases, runner):
    import numpy as np

    # Local paths/device IDs and SERVER_INDEX may differ. The stimulus and
    # operator call sequence must be identical before entering SHMEM collectives.
    return {
        "protocol": 1,
        "server_num": args.server_num,
        "devices_per_server": len(args.devices),
        "cases": cases,
        "seed": args.seed,
        "mode": args.mode,
        "save_gmm1_output": args.save_gmm1_output,
        "warmup": args.warmup,
        "loops": args.loops,
        "repeat": args.repeat,
        "routing_mode": args.routing_mode,
        "routing_skew": args.routing_skew,
        "topk_weight_mode": args.topk_weight_mode,
        "shmem_ip_port": args.shmem_ip_port,
        "tolerances": [args.absolute_tolerance, args.relative_tolerance, args.max_error_ratio],
        "numpy": np.__version__,
        "test_sources": {
            name: source_digest(runner.TEST_DIR / name)
            for name in (
                "test_mega_moe.py",
                "mega_moe_bf16_golden.py",
                "bf16_data_cache.py",
                "physical_server_support.py",
            )
        },
    }


def run_multinode(args, runner, control_port, timeout):
    from urllib.parse import urlsplit

    server_num, server_index = args.server_num, args.server_index
    address = urlsplit(args.shmem_ip_port)
    if address.scheme != "tcp" or address.hostname is None or address.port is None:
        raise ValueError("--shmem-ip-port must be tcp://master_ip:PORT")
    master_ip, shmem_port = address.hostname, address.port
    local_ranks = validate_topology(
        server_num,
        server_index,
        args.devices,
        master_ip,
        shmem_port,
        control_port,
        timeout,
        runner.EXPERTS_PER_RANK,
        runner.TOPK,
    )
    if (
        (args.warmup is not None and args.warmup < 0)
        or (args.loops is not None and args.loops <= 0)
        or args.repeat <= 0
        or args.routing_skew <= 0
    ):
        raise ValueError("Require warmup >= 0, loops/repeat > 0 and routing-skew > 0")
    ep = server_num * len(args.devices)
    cases = runner.read_cases(args.csv_path, ep, args.routing_mode)
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = ",".join(map(str, args.devices))
    os.environ["SHMEM_UID_SESSION_ID"] = args.shmem_ip_port.removeprefix("tcp://")
    data_root = args.data_dir or runner.TEST_DIR / "output" / (
        f"physical_{time.strftime('%Y%m%d_%H%M%S')}_server{server_index}_{os.getpid()}"
    )
    # --data-dir is a root: each physical server gets a separate directory even
    # if the caller happens to point both hosts at the same shared filesystem.
    data_root = data_root / f"server_{server_index}"
    data_root.mkdir(parents=True, exist_ok=True)
    print(
        f"[TOPOLOGY] server={server_index}/{server_num} local_devices={args.devices} "
        f"global_ranks={local_ranks} EP={ep} experts={ep * runner.EXPERTS_PER_RANK} "
        f"SHMEM={args.shmem_ip_port} control={master_ip}:{control_port}",
        flush=True,
    )
    print(f"[INFO] Test data: {data_root}", flush=True)

    store = None
    if server_num > 1:
        import torch.distributed as dist

        # On the physical master, PyTorch 2.7.1's libuv backend failed a
        # localhost 256 KiB round trip; the legacy backend passed on both hosts.
        # Select it explicitly for the control data.
        store = dist.TCPStore(
            master_ip, control_port, server_num, server_index == 0, timeout=timedelta(seconds=timeout), use_libuv=False
        )
    coordinator = Coordinator(store, server_num, server_index, runner_module=runner)
    coordinator.check_manifest(make_manifest(args, cases, runner))
    for index, case in enumerate(cases):
        case_coordinator = Coordinator(store, server_num, server_index, f"case/{index}", runner_module=runner)
        runner.run_case(args, case, data_root / runner.shape_directory_name(case), case_coordinator)
    coordinator.phase("finished", lambda: None)
    print(f"[SUCCESS][GLOBAL] ordinary BF16 EP={ep}, all servers passed; data={data_root}", flush=True)

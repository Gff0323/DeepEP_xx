"""CPU checks for physical-server orchestration; no CANN or NPU required."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

import mega_moe_bf16_golden as golden
import test_mega_moe as runner
from physical_server_support import Coordinator, run_multinode, source_digest, validate_topology


class MemoryStore:
    """Blocking Store interface for deterministic two-parent protocol tests."""

    def __init__(self):
        self.values = {}
        self.condition = threading.Condition()

    def set(self, key, value):
        with self.condition:
            self.values[key] = value
            self.condition.notify_all()

    def get(self, key):
        with self.condition:
            if not self.condition.wait_for(lambda: key in self.values, timeout=5):
                raise TimeoutError(key)
            return self.values[key]

    def wait(self, keys):
        for key in keys:
            self.get(key)


def both(actions):
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(action) for action in actions]
        return [future.result(timeout=10) for future in futures]


class PhysicalLauncherTests(unittest.TestCase):
    def test_multinode_dispatch_runs_both_parent_lifecycles(self):
        store = MemoryStore()
        torch = types.ModuleType("torch")
        dist = types.ModuleType("torch.distributed")
        dist.TCPStore = lambda *args, **kwargs: store
        torch.distributed = dist
        seen = []

        def run_case(args, case, directory, coordinator):
            seen.append((args.server_index, case["EP"], directory.name))
            coordinator.phase("compute", lambda: None)

        facade = types.SimpleNamespace(
            TEST_DIR=runner.TEST_DIR,
            EXPERTS_PER_RANK=runner.EXPERTS_PER_RANK,
            TOPK=runner.TOPK,
            read_cases=runner.read_cases,
            shape_directory_name=runner.shape_directory_name,
            run_case=run_case,
        )
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict("sys.modules", {"torch": torch, "torch.distributed": dist}),
            patch.dict("os.environ", {}, clear=True),
        ):

            def run_node(node):
                args = runner.parse_args(
                    [
                        "0,1,2,3,4,5,6,7",
                        "--mode",
                        "precision",
                        "--data-dir",
                        tmp,
                        "--shmem-ip-port",
                        "tcp://192.0.2.1:8899",
                    ]
                )
                args.server_num, args.server_index = 2, node
                run_multinode(args, facade, 8900, 10)

            both([lambda: run_node(0), lambda: run_node(1)])
        self.assertEqual(sorted((node, ep) for node, ep, _ in seen), [(0, 16), (1, 16)])

    def test_unified_main_selects_single_or_multinode(self):
        for nodes, node in ((1, 0), (2, 1)):
            with tempfile.TemporaryDirectory() as tmp:
                args = runner.parse_args(["0,1,2,3,4,5,6,7", "--mode", "precision", "--data-dir", tmp])
                with (
                    patch.object(runner, "SERVER_NUM", nodes),
                    patch.object(runner, "SERVER_INDEX", node),
                    patch.object(runner, "parse_args", return_value=args),
                    patch.object(runner, "run_case") as run_case,
                    patch("physical_server_support.run_multinode") as multi,
                    patch.dict("os.environ", {}),
                ):
                    runner.main()
                    if nodes == 1:
                        multi.assert_not_called()
                        self.assertEqual(run_case.call_args.args[1]["EP"], 8)
                    else:
                        run_case.assert_not_called()
                        self.assertIs(multi.call_args.args[1], runner)
                        self.assertEqual(multi.call_args.args[0].server_index, 1)

    def test_editable_host_config_does_not_trip_code_consistency_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / str(index) for index in range(3)]
            for index, path in enumerate(paths):
                path.write_text(
                    f'SERVER_INDEX = {index}\nMASTER_IP = "192.0.2.{index}"\n'
                    f"def compute():\n    return {1 if index < 2 else 2}\n",
                    encoding="utf-8",
                )
            self.assertEqual(source_digest(paths[0]), source_digest(paths[1]))
            self.assertNotEqual(source_digest(paths[0]), source_digest(paths[2]))

    def test_master_ip_defaults_and_explicit_override(self):
        with patch.object(runner, "MASTER_IP", "192.0.2.1"), patch.dict("os.environ", {}, clear=True):
            self.assertEqual(runner.parse_args(["0,1,2,3"]).shmem_ip_port, "tcp://192.0.2.1:8899")
            self.assertEqual(
                runner.parse_args(["0,1,2,3", "--shmem-ip-port", "tcp://192.0.2.2:9999"]).shmem_ip_port,
                "tcp://192.0.2.2:9999",
            )

    def test_independent_hosts_generate_identical_inputs_and_golden(self):
        case = {
            "M": 1,
            "K": 256,
            "N": 512,
            "EP": 4,
            "TOPK": 4,
            "expertPerRank": 1,
            "maxRecvTokenNum": 4,
            "validateRecvCapacity": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / str(node) for node in (0, 1)]
            for directory in dirs:
                golden.generate(case, directory, 831, early_topk_weight=True)
            names = sorted(path.name for path in dirs[0].iterdir())
            self.assertEqual(names, sorted(path.name for path in dirs[1].iterdir()))
            for name in names:
                self.assertEqual((dirs[0] / name).read_bytes(), (dirs[1] / name).read_bytes(), name)

    def test_rank_ranges_and_invalid_topologies(self):
        for node in (0, 1):
            self.assertEqual(
                validate_topology(2, node, list(range(8)), "192.0.2.1", 8899, 8900, 10),
                list(range(node * 8, node * 8 + 8)),
            )
        for nodes, node, ip, control in [
            (2, 0, "127.0.0.1", 8900),
            (1, 1, "127.0.0.1", 8900),
            (2, 0, "192.0.2.1", 8899),
        ]:
            with self.assertRaises(ValueError):
                validate_topology(nodes, node, list(range(8)), ip, 8899, control, 10)

    def test_worker_maps_physical_order_and_global_input_rank(self):
        class StopBeforeNpuWork(Exception):
            pass

        torch = types.ModuleType("torch")
        torch.device = lambda value: value
        torch.npu = types.SimpleNamespace(set_device=lambda value: None)
        extension = types.ModuleType("deep_ep_experimental.megamoe_train")
        for name in ("MegaMoeBF16Train",):
            setattr(extension, name, object)
        config = {"rank_start": 8, "shmem_ip_port": "tcp://192.0.2.1:8899", "data_dir": "."}
        with (
            patch.dict(
                "sys.modules",
                {
                    "torch": torch,
                    "torch_npu": types.ModuleType("torch_npu"),
                    "deep_ep_experimental.megamoe_train": extension,
                },
            ),
            patch.dict("os.environ", {}),
        ):
            for devices, expected_devices in (
                (list(range(8)), {0: 0, 5: 5, 7: 7}),
                ([0, 1, 2, 3, 4, 7, 6, 5], {0: 0, 5: 7, 7: 5}),
            ):
                config["devices"] = devices
                for local_rank in (0, 5, 7):
                    with patch.object(runner, "load_inputs", side_effect=StopBeforeNpuWork) as load:
                        with self.assertRaises(StopBeforeNpuWork):
                            runner.run_rank(local_rank, {"EP": 16}, config)
                        self.assertEqual(load.call_args.args[0], 8 + local_rank)
                        self.assertEqual(load.call_args.args[3], f"npu:{expected_devices[local_rank]}")
                        import os

                        self.assertEqual(os.environ["RANK"], str(8 + local_rank))
                        self.assertEqual(os.environ["LOCAL_RANK"], str(local_rank))

    def test_config_mismatch_rejected_on_both_servers(self):
        store = MemoryStore()

        def attempt(node):
            with self.assertRaisesRegex(ValueError, "configurations differ"):
                Coordinator(store, 2, node).check_manifest({"EP": 16, "loops": node + 1})

        both([lambda: attempt(0), lambda: attempt(1)])

    def test_one_server_failure_is_reported_on_both(self):
        store = MemoryStore()

        def attempt(node):
            def action():
                if node == 1:
                    raise ValueError("golden generation failed")

            with self.assertRaisesRegex(RuntimeError, "golden generation failed"):
                Coordinator(store, 2, node).phase("generate", action)

        both([lambda: attempt(0), lambda: attempt(1)])

    def test_local_validation_does_not_require_remote_outputs(self):
        torch = types.ModuleType("torch")
        mp = types.ModuleType("torch.multiprocessing")
        torch.multiprocessing = mp
        for node, ep in [(0, 8), (1, 16)]:
            with tempfile.TemporaryDirectory() as tmp:
                args = runner.parse_args(["0,1,2,3,4,5,6,7", "--mode", "precision"])
                args.server_index = node
                case = runner.read_cases(runner.TEST_DIR / "mega_moe_cases.csv", ep)[0]
                with (
                    patch.dict("sys.modules", {"torch": torch, "torch.multiprocessing": mp}),
                    patch.object(mp, "spawn", create=True) as spawn,
                    patch("bf16_data_cache.prepare_data"),
                    patch.object(golden, "compare") as compare,
                ):
                    runner.run_case(args, case, Path(tmp))
                    self.assertEqual(spawn.call_args.kwargs["nprocs"], 8)
                    config = spawn.call_args.kwargs["args"][1]
                    self.assertEqual(config["rank_start"], node * 8)
                    expected = list(range(node * 8, node * 8 + 8))
                    self.assertEqual(compare.call_args.kwargs["ranks"], expected)

    def test_global_performance_includes_both_servers(self):
        store = MemoryStore()
        case = {"M": 128, "K": 512, "N": 1024, "EP": 4}
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [Path(tmp) / str(node) for node in (0, 1)]
            for directory in dirs:
                (directory / "python_results").mkdir(parents=True)
            both(
                [
                    lambda node=node: Coordinator(store, 2, node).report_performance(
                        case,
                        [{"rank": rank, "median_us": float(rank + 1)} for rank in range(node * 2, node * 2 + 2)],
                        dirs[node],
                    )
                    for node in (0, 1)
                ]
            )
            self.assertEqual(
                (dirs[0] / "python_results/performance_global.csv").read_bytes(),
                (dirs[1] / "python_results/performance_global.csv").read_bytes(),
            )
            self.assertEqual(len((dirs[0] / "python_results/performance_global.csv").read_text().splitlines()), 5)


if __name__ == "__main__":
    unittest.main()

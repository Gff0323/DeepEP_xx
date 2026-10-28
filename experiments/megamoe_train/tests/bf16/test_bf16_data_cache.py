"""CPU cache lifecycle and launcher checks; no CANN/NPU required."""

from contextlib import redirect_stdout, redirect_stderr
import errno
import io
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import bf16_data_cache as cache
import mega_moe_bf16_golden as golden
import test_mega_moe as runner


CASE = dict(
    M=2,
    N=8,
    K=16,
    EP=2,
    TOPK=2,
    expertPerRank=1,
    maxRecvTokenNum=4,
    validateRecvCapacity=True,
    localMemBytes=2 * 1024**2,
)


def fake_files(case, directory, include_golden, value):
    for name, size in cache._file_sizes(case, include_golden, golden).items():
        (directory / name).write_bytes(bytes([value]) * size)


def concurrent_prepare(root, index, start):
    root = Path(root)

    def generate(case, directory, seed, **options):
        with (root / "generations").open("ab") as output:
            output.write(b"1")
        time.sleep(0.4)
        fake_files(case, directory, options["generate_golden"], 1)

    start.wait(10)
    with patch.object(golden, "generate", side_effect=generate):
        cache.prepare_data(CASE, root / f"run{index}", 831, cache_root=root / "cache")


class DataCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="bf16-cache-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []
        self.log = io.StringIO()
        self.stdout = redirect_stdout(self.log)
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.generator = patch.object(golden, "generate", side_effect=self.generate)
        self.generator.start()
        self.addCleanup(self.generator.stop)

    def generate(self, case, directory, seed, **options):
        self.calls.append(options)
        fake_files(case, directory, options["generate_golden"], len(self.calls))

    def prepare(self, name="run", case=None, seed=831, **kwargs):
        return cache.prepare_data(
            CASE if case is None else case, self.root / name, seed, cache_root=self.root / "cache", **kwargs
        )

    def entry(self, name="run"):
        return Path(json.loads((self.root / name / "input_cache.json").read_text())["cache"])

    def test_hit_across_runs_and_output_is_not_cached(self):
        self.assertFalse(self.prepare("first"))
        self.assertTrue(self.prepare("second"))
        self.assertEqual(len(self.calls), 1)
        for name in cache._file_sizes(CASE, True, golden):
            self.assertTrue(os.path.samefile(self.root / "first" / name, self.root / "second" / name))
        output = self.root / "second" / "output_y_bf16.bin"
        output.write_bytes(b"keep this result")
        self.assertTrue(self.prepare("second"))
        self.assertEqual(output.read_bytes(), b"keep this result")
        self.assertIn("[DATA_CACHE] HIT", self.log.getvalue())

    def test_refresh_keeps_previous_run_hardlinks_unchanged(self):
        self.prepare("first")
        name = next(iter(cache._file_sizes(CASE, True, golden)))
        original = (self.root / "first" / name).read_bytes()
        self.assertFalse(self.prepare("second", refresh=True))
        self.assertEqual((self.root / "first" / name).read_bytes(), original)
        self.assertNotEqual((self.root / "second" / name).read_bytes(), original)
        self.assertTrue(self.prepare("third"))

    def test_every_stimulus_setting_changes_key(self):
        self.prepare()
        baseline = self.entry()
        for field in ("M", "N", "K", "EP", "TOPK", "expertPerRank", "maxRecvTokenNum"):
            case = dict(CASE, **{field: CASE[field] * 2})
            with self.subTest(field=field):
                self.assertFalse(self.prepare(case=case))
                self.assertNotEqual(self.entry(), baseline)
        self.assertFalse(self.prepare(case=dict(CASE, validateRecvCapacity=False)))
        for kwargs in (
            dict(seed=832),
            dict(routing_mode="random"),
            dict(routing_skew=2.0),
            dict(topk_weight_mode="uniform"),
            dict(early_topk_weight=False),
        ):
            with self.subTest(kwargs=kwargs):
                self.assertFalse(self.prepare(**kwargs))
                self.assertNotEqual(self.entry(), baseline)

    def test_allocation_only_changes_reuse_cache(self):
        self.prepare()
        self.assertTrue(self.prepare(case=dict(CASE, localMemBytes=3 * 1024**3)))
        self.assertEqual(len(self.calls), 1)

    def test_generator_version_changes_invalidate(self):
        self.prepare()
        with patch.object(cache, "CACHE_VERSION", cache.CACHE_VERSION + 1):
            self.assertFalse(self.prepare())
        with patch.object(golden.np, "__version__", "different-version"):
            self.assertFalse(self.prepare())
        source = self.root / "new_generator.py"
        source.write_text("# changed golden implementation\n")
        with patch.object(golden, "__file__", str(source)):
            self.assertFalse(self.prepare())

    def test_performance_cache_upgrades_then_precision_cache_serves_performance(self):
        self.prepare(generate_golden=False)
        self.assertTrue(self.prepare(generate_golden=False))
        self.assertFalse(self.prepare(generate_golden=True))
        self.assertTrue(self.prepare(generate_golden=False))
        self.assertTrue(self.prepare(generate_golden=True))
        self.assertEqual([call["generate_golden"] for call in self.calls], [False, True])

    def test_missing_truncated_changed_files_and_bad_manifest(self):
        self.prepare()
        name = next(iter(cache._file_sizes(CASE, True, golden)))
        target = self.entry() / "data" / name
        target.unlink()
        self.assertFalse(self.prepare())
        target.write_bytes(b"truncated")
        self.assertFalse(self.prepare())
        stat = target.stat()
        os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000))
        self.assertFalse(self.prepare())
        for text in ("incomplete json", "[]", "{}"):
            (self.entry() / "manifest.json").write_text(text)
            self.assertFalse(self.prepare())

    def test_interrupted_generation_cannot_be_hit(self):
        self.prepare("old")
        entry = self.entry("old")

        def interrupted(case, directory, seed, **options):
            name = next(iter(cache._file_sizes(case, True, golden)))
            (directory / name).write_bytes(b"incomplete")
            raise RuntimeError("interrupted golden")

        with patch.object(golden, "generate", side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                self.prepare(refresh=True)
        self.assertFalse((entry / "manifest.json").exists())
        self.assertFalse(self.prepare())
        self.assertTrue(self.prepare())

    def test_incomplete_success_does_not_publish_manifest(self):
        with patch.object(golden, "generate", return_value=None):
            with self.assertRaises(FileNotFoundError):
                self.prepare()
        self.assertEqual(list((self.root / "cache").glob("*/manifest.json")), [])
        self.assertFalse(self.prepare())

    def test_cross_filesystem_copy_fallback(self):
        with patch.object(cache.os, "link", side_effect=OSError(errno.EXDEV, "different filesystem")):
            self.prepare("first")
            self.assertTrue(self.prepare("second"))
        for name in cache._file_sizes(CASE, True, golden):
            self.assertEqual((self.root / "first" / name).read_bytes(), (self.root / "second" / name).read_bytes())
        self.assertEqual(len(self.calls), 1)
        self.assertIn("Hard links unavailable", self.log.getvalue())

    def test_launcher_still_computes_and_compares_on_hits(self):
        torch = types.ModuleType("torch")
        mp = types.ModuleType("torch.multiprocessing")
        torch.multiprocessing = mp
        mp.spawn = unittest.mock.Mock()
        for variant in ("MEGA_MOE_BF16",):
            args = runner.parse_args(["0,1", "--mode", "precision", "--data-cache-dir", str(self.root / "cache")])
            with (
                patch.dict("sys.modules", {"torch": torch, "torch.multiprocessing": mp}),
                patch.dict(os.environ, {variant: "1"}),
                patch.object(runner, "check_saved_packed_weights") as saved,
                patch.object(golden, "compare") as compare,
            ):
                runner.run_case(args, CASE, self.root / variant)
                saved.assert_called_once()
                compare.assert_called_once()
        self.assertEqual(mp.spawn.call_count, 1)
        self.assertEqual(len(self.calls), 1)

    def test_real_generator_bytes_and_comparison_preserved(self):
        self.generator.stop()
        case = dict(CASE, N=512, K=256)
        fresh = self.root / "uncached"
        golden.generate(case, fresh, 831, early_topk_weight=True)
        self.assertFalse(self.prepare("first", case=case))
        self.assertTrue(self.prepare("second", case=case))
        for name in cache._file_sizes(case, True, golden):
            self.assertEqual((fresh / name).read_bytes(), (self.root / "second" / name).read_bytes())
        directory = self.root / "second"
        for rank in range(case["EP"]):
            for suffix in ("y_bf16", "expert_token_nums_bf16"):
                shutil.copyfile(
                    golden.data_path(directory, rank, "golden_" + suffix, case),
                    golden.data_path(directory, rank, "output_" + suffix, case),
                )
        golden.compare(case, directory, 0.01, 0.01, 0.05)
        bad = golden.data_path(directory, 0, "output_y_bf16", case)
        bad.write_bytes(b"\xc8\x42" * (case["M"] * case["K"]))
        with redirect_stderr(io.StringIO()), self.assertRaises(AssertionError):
            golden.compare(case, directory, 0.01, 0.01, 0.05)

    def test_two_processes_generate_only_once(self):
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        processes = [context.Process(target=concurrent_prepare, args=(str(self.root), i, start)) for i in range(2)]
        try:
            for process in processes:
                process.start()
            start.set()
            for process in processes:
                process.join(30)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual((self.root / "generations").read_bytes(), b"1")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                    process.join(5)


if __name__ == "__main__":
    unittest.main()

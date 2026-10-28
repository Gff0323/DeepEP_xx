"""CPU checks for W8A8-compatible saved-state compression (no NPU required)."""

from pathlib import Path
import re
import unittest
import numpy as np
from saved_gate_up_reference import w8a8_reference, compare_mxfp8_blocks
from test_workspace_layout import training_saved_state_size


def method(source, name):
    start = source.index(name + "(")
    brace = source.index("{", start)
    end, depth = brace + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[brace:end]


class SavedMxFp8Tests(unittest.TestCase):
    def test_w8a8_scalar_types_preserved(self):
        root = Path(__file__).resolve().parents[2]
        current = (root / "csrc/kernels/bf16/include/block_saved_gate_up_mxfp8.h").read_text(encoding="utf-8")

        def declaration(text, name):
            pattern = r"^\s*(?:static\s+)?constexpr\s+(\w+)\s+" + name + r"\s*=\s*([^;]+);"
            match = re.search(pattern, text, re.MULTILINE)
            self.assertIsNotNone(match, name)
            dtype, value = match.groups()
            return dtype, int(re.sub(r"[uUlL]+$", "", value.strip()), 0)

        expected = {
            "MAX_EXP_FOR_BF16": ("uint16_t", 0x7F80),
            "MAX_EXP_FOR_FP8": ("uint16_t", 0xFF),
            "BF16_EXP_BIAS": ("uint16_t", 0x7F00),
            "SHR_NUM_FOR_BF16": ("int16_t", 7),
            "NAN_CUSTOMIZATION": ("uint16_t", 0x7F81),
            "SPECIAL_EXP_THRESHOLD": ("uint16_t", 0x40),
            "OUT_ELE_NUM_ONE_BLK": ("int64_t", 64),
        }
        for name, value in expected.items():
            self.assertEqual(declaration(current, name), value)

    def test_mode_specific_allocations(self):
        args = (4096, 7168, 16, 8, 4, 65536, 8192)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=True), 1493833728)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=False), 2013927424)

    def test_groups_scales_and_reference_checker(self):
        ref = w8a8_reference()
        rng = np.random.default_rng(20260910)
        for power in (-60, -10, 0, 10, 60):
            data = ref.bf16_round(rng.normal(size=(3, 6, 32)).astype(np.float32) * np.float32(2.0**power))
            raw, decoded, scales, scale_values = ref.mxfp8_quantize(data, -1)
            np.testing.assert_array_equal(decoded, ref.decode_fp8_e4m3fn(raw) * scale_values)
            compare_mxfp8_blocks(raw, scales.squeeze(-1), data)
        # Normal-domain corruptions must be detected rather than accepted as quantization noise.
        data = ref.bf16_round(rng.normal(size=(3, 6, 32)).astype(np.float32))
        raw, _, scales, _ = ref.mxfp8_quantize(data, -1)
        damaged = raw.copy()
        damaged[0, 0, :] ^= np.uint8(0x80)
        with self.assertRaises(AssertionError):
            compare_mxfp8_blocks(damaged, scales.squeeze(-1), data)
        damaged = scales.squeeze(-1).copy()
        damaged[0, 0] += 3
        with self.assertRaises(AssertionError):
            compare_mxfp8_blocks(raw, damaged, data)

    def test_zero_rounding_and_saturation(self):
        ref = w8a8_reference()
        zero = np.zeros((2, 3, 32), dtype=np.float32)
        raw, decoded, scales, _ = ref.mxfp8_quantize(zero, -1)
        self.assertTrue(np.all(raw == 0) and np.all(decoded == 0) and np.all(scales == 0))
        np.testing.assert_array_equal(
            ref.encode_fp8_e4m3fn(np.array([1.0625, 1.1875, 1e6], dtype=np.float32)), [0x38, 0x3A, 0x7E]
        )

    def test_block_saturation_and_nan_diagnostics(self):
        ref = w8a8_reference()
        # These finite BF16 values normalize above E4M3FN's 448 maximum.
        # Saturated output must remain finite; accepting NaN would hide the bug.
        data = np.tile(np.array([1.875, -1.875], dtype=np.float32), (2, 3, 16))
        raw, _, scales, _ = ref.mxfp8_quantize(data, -1)
        self.assertEqual(int(raw[0, 0, 0]), 0x7E)
        self.assertEqual(int(raw[0, 0, 1]), 0xFE)
        compare_mxfp8_blocks(raw, scales.squeeze(-1), data)
        for code in (0x7F, 0xFF):
            damaged = raw.copy()
            damaged[1, 2, 7] = code
            with self.assertRaises(AssertionError) as failure:
                compare_mxfp8_blocks(damaged, scales.squeeze(-1), data, context="rank=4 plane=up")
            message = str(failure.exception)
            self.assertIn("rank=4 plane=up", message)
            self.assertIn("NaN FP8 encoding", message)
            self.assertIn("sample_index=(1, 2, 7)", message)
            self.assertIn(f"raw=0x{code:02x}", message)
            self.assertIn("expected_group_hex=", message)

    def test_reference_and_scale_failures_are_distinct(self):
        ref = w8a8_reference()
        data = np.ones((1, 2, 32), dtype=np.float32)
        raw, _, scales, _ = ref.mxfp8_quantize(data, -1)
        scales = scales.squeeze(-1)
        for value in (np.nan, np.inf):
            invalid = data.copy()
            invalid[0, 1, 3] = value
            with self.assertRaisesRegex(AssertionError, "non-finite CPU GMM1 reference"):
                compare_mxfp8_blocks(raw, scales, invalid)
        damaged = scales.copy()
        damaged[0, 1] = 255
        with self.assertRaisesRegex(AssertionError, "non-finite E8M0 scale"):
            compare_mxfp8_blocks(raw, damaged, data)
        damaged[0, 1] = 254
        with self.assertRaisesRegex(AssertionError, "non-finite dequantized value"):
            compare_mxfp8_blocks(raw, damaged, data)


if __name__ == "__main__":
    unittest.main()

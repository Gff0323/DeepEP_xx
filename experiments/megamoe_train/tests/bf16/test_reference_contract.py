# Copyright (c) 2026, Lu Lu
# Modified by xixing 2026

"""CPU regressions for the ATK runner's contract reference, without NPU calls."""

import math
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    if error.name != "torch":
        raise
    torch = None


def cpu_grouped_matmul(tokens, weights, counts):
    # Replace only the device GEMM boundary; exercise the real activation path.
    assert tokens.dtype == weights.dtype == torch.bfloat16
    parts, offset = [], 0
    for expert, size in enumerate(counts.tolist()):
        parts.append((tokens[offset : offset + size].float() @ weights[expert].float()).bfloat16())
        offset += size
    return torch.cat(parts)


def cpu_grouped_wgrad(tokens, grad, counts):
    assert tokens.dtype == grad.dtype == torch.bfloat16
    parts, offset = [], 0
    for size in counts.tolist():
        a, b = tokens[offset : offset + size], grad[offset : offset + size]
        parts.append((a.float().T @ b.float()).bfloat16())
        offset += size
    return torch.stack(parts)


@unittest.skipIf(torch is None, "CPU Torch is required; no NPU or ATK is needed")
class ReferenceContract(unittest.TestCase):
    def setUp(self):
        import npu_baseline

        self.baseline = npu_baseline
        self.enterContext(patch.object(npu_baseline, "_grouped_matmul", cpu_grouped_matmul))
        self.enterContext(patch.object(npu_baseline, "_grouped_wgrad", cpu_grouped_wgrad))

    def inputs(self, probability, ffn=1):
        w1 = torch.zeros((1, 2 * ffn, 1), dtype=torch.bfloat16)
        w1[0, 0, 0], w1[0, ffn, 0] = 1, 1.3125
        w2 = torch.zeros((1, 1, ffn), dtype=torch.bfloat16)
        w2[0, 0, 0] = 1
        return {
            "x": torch.ones((1, 1), dtype=torch.bfloat16),
            "ids": torch.zeros((1, 1), dtype=torch.int32),
            "p": torch.tensor([[probability]], dtype=torch.bfloat16),
            "w1": w1,
            "w2": w2,
            "dy": torch.tensor([[0.75]], dtype=torch.bfloat16),
        }

    def test_fp32_h_rounds_only_after_route_weight(self):
        for reference in ("bigop", "megamoe"):
            with self.subTest(reference=reference):
                trace = {}
                result = self.baseline.run_baseline(
                    self.inputs(0.00099945068359375), "cpu", 0, reference=reference, trace=trace
                )
                self.assertEqual(trace["h"].dtype, torch.float32)
                self.assertAlmostEqual(trace["h"].item(), 1.3125 / (1 + math.exp(-1)), delta=1e-7)
                self.assertEqual(trace["hp"].dtype, torch.bfloat16)
                self.assertEqual(result["routes"][0]["y"].item(), 0.000957489013671875)

    def test_fp32_dh_reaches_activation_derivative(self):
        for reference in ("bigop", "megamoe"):
            with self.subTest(reference=reference):
                trace = {}
                self.baseline.run_baseline(self.inputs(0.10009765625), "cpu", 0, reference=reference, trace=trace)
                self.assertEqual(trace["dz"].dtype, torch.bfloat16)
                self.assertEqual(trace["dz"][0, 0].item(), 0.09130859375)
                self.assertEqual(trace["dz"][0, 1].item(), 0.054931640625)

    def test_bigop_and_contract_share_numeric_boundaries(self):
        bigop, contract = {}, {}
        inputs = self.inputs(0.10009765625)
        self.baseline.run_baseline(inputs, "cpu", 0, reference="bigop", trace=bigop)
        self.baseline.run_baseline(inputs, "cpu", 0, reference="megamoe", trace=contract)
        for name in ("h", "hp", "dhp", "dp", "dz"):
            torch.testing.assert_close(bigop[name], contract[name], rtol=0, atol=0)
        self.assertEqual(bigop["h"].dtype, torch.float32)

    def test_saved_bf16_reuses_unmodified_forward_gate_up(self):
        captured = []

        def activation(z, *args):
            captured.append(z)
            return original(z, *args)

        original = self.baseline._swiglu_fp32
        with patch.object(self.baseline, "_swiglu_fp32", side_effect=activation):
            self.baseline.run_baseline(self.inputs(0.5), "cpu", 0, swiglu_limit=0.1, reference="bigop")
        self.assertEqual(len(captured), 2)
        self.assertIs(captured[0], captured[1])
        torch.testing.assert_close(captured[0], torch.tensor([[1, 1.3125]], dtype=torch.bfloat16))

    def test_non_bf16_clipping_limit_stays_fp32(self):
        trace = {}
        self.baseline.run_baseline(self.inputs(0.5), "cpu", 0, swiglu_limit=0.1, reference="bigop", trace=trace)
        expected_h = 0.1 / (1 + math.exp(-0.1)) * 0.1
        self.assertAlmostEqual(trace["h"].item(), expected_h, delta=1e-9)
        self.assertEqual(trace["dz"].count_nonzero().item(), 0)

    def test_clipping_retains_gradient_on_inclusive_boundaries(self):
        z = torch.tensor([[1.5, 1.5], [1.5078125, 1.5], [1.5, -1.5], [1.5, -1.5078125]], dtype=torch.bfloat16)
        dz = self.baseline._swiglu_backward_fp32(torch.ones((4, 1)), z, 1.5)
        self.assertTrue(bool((dz[[0, 2], 0] != 0).all()))
        self.assertTrue(bool((dz[:3, 1] != 0).all()))
        self.assertEqual(dz[1, 0].item(), 0)
        self.assertEqual(dz[3, 1].item(), 0)

    def test_dp_multiplies_and_reduces_in_fp32(self):
        inputs = self.inputs(1.0, ffn=2)
        inputs["w1"][0, 1, 0], inputs["w1"][0, 3, 0], inputs["w2"][0, 0, 1] = 1, 1.3125, -0.99609375
        trace = {}
        self.baseline.run_baseline(inputs, "cpu", 0, reference="bigop", trace=trace)
        h = 1.3125 / (1 + math.exp(-1))
        self.assertEqual(trace["dp"].dtype, torch.bfloat16)
        torch.testing.assert_close(trace["dhp"], torch.tensor([[0.75, -0.74609375]], dtype=torch.bfloat16))
        self.assertEqual(trace["dp"].item(), torch.tensor((0.75 - 0.74609375) * h).bfloat16().item())

    def test_route_combine_keeps_fp32_cancellation_and_slot_order(self):
        ids = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        for reference in ("bigop", "megamoe"):
            with self.subTest(reference=reference):
                buffer = torch.tensor([[[256, -256, 0.125], [1, -1, -0.5], [-256, 256, 1.0]]])
                result = self.baseline.unpermute_routes(buffer, ids, "cpu", reference=reference)
                self.assertEqual(result["y"].item(), 1)
                self.assertEqual(result["dx"].item(), -1)
                torch.testing.assert_close(result["dtopk_weights"], buffer[:, :, -1].bfloat16())
                buffer[0, :, 0] = torch.tensor([2**24, 1, -(2**24)])
                result = self.baseline.unpermute_routes(buffer, ids, "cpu", reference=reference)
                self.assertEqual(result["y"].item(), 0)

    def test_final_bf16_cast_rounds_halfway_values_to_even(self):
        buffer = torch.tensor([[[1, -1, 0], [2**-8, -(2**-8), 0]], [[1.0078125, -1.0078125, 0], [2**-8, -(2**-8), 0]]])
        ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        result = self.baseline.unpermute_routes(buffer, ids, "cpu", reference="bigop")
        torch.testing.assert_close(result["y"], torch.tensor([[1], [1.015625]], dtype=torch.bfloat16))
        torch.testing.assert_close(result["dx"], -result["y"])

    def test_bf16_reference_covers_row_boundaries_and_empty_experts(self):
        for rows in (1, 127, 128, 129):
            with self.subTest(rows=rows):
                inputs = self.inputs(0.00099945068359375)
                for name in ("x", "ids", "p", "dy"):
                    inputs[name] = inputs[name].repeat(rows, 1)
                for name in ("w1", "w2"):
                    inputs[name] = torch.cat((inputs[name], torch.zeros_like(inputs[name])))
                result = self.baseline.run_baseline(inputs, "cpu", 0, reference="bigop")
                self.assertEqual(result["counts"].tolist(), [rows, 0])
                self.assertEqual(tuple(result["routes"][0]["y"].shape), (rows, 1))
                self.assertTrue(bool((result["routes"][0]["y"] == 0.000957489013671875).all()))
                for name in ("dw1", "dw2"):
                    self.assertEqual(result[name].dtype, torch.bfloat16)
                    self.assertEqual(result[name][1].count_nonzero().item(), 0)

    def test_cpu_truth_keeps_fp64_mathematical_values(self):
        from cpu_reference import run_reference

        inputs = self.inputs(0.10009765625)
        result = run_reference(inputs, 0, chunk_rows=1)
        p = inputs["p"].item()
        s = 1 / (1 + math.exp(-1))
        h = s * 1.3125
        dg, du = (0.75 * p * 1.3125) * (s * (2 - s)), 0.75 * p * s
        expected = {
            "y": h * p,
            "dx": dg + 1.3125 * du,
            "dtopk_weights": 0.75 * h,
            "dw2": 0.75 * h * p,
        }
        for name, value in expected.items():
            self.assertEqual(result[name].dtype, torch.float64)
            self.assertAlmostEqual(result[name].item(), value, delta=1e-14)
        torch.testing.assert_close(result["dw1"].flatten(), torch.tensor([dg, du], dtype=torch.float64))

    def test_saved_state_replay_uses_same_fp32_contract(self):
        import mxfp8_diagnostics

        self.enterContext(patch.object(mxfp8_diagnostics, "_grouped_matmul", cpu_grouped_matmul))
        self.enterContext(patch.object(mxfp8_diagnostics, "_grouped_wgrad", cpu_grouped_wgrad))
        inputs, trace = self.inputs(0.10009765625, ffn=32), {}
        expected = self.baseline.run_baseline(inputs, "cpu", 0, "mxfp8", trace=trace)
        actual, stages = mxfp8_diagnostics.replay_saved_state(inputs, trace, "cpu", {"gate", "up"}, 0)
        self.assertEqual(stages["h"].dtype, torch.float32)
        self.assertEqual(stages["hp"].dtype, torch.bfloat16)
        for name in ("h", "hp", "dz", "dp"):
            torch.testing.assert_close(stages[name], trace[name], rtol=0, atol=0)
        for name in ("dw1", "dw2"):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_clipping_gradient_masks_remain_effective(self):
        gate_up = torch.tensor([[2, -2]], dtype=torch.bfloat16)
        dz = self.baseline._swiglu_backward_fp32(torch.tensor([[0.075]]), gate_up, 1.5)
        self.assertEqual(dz.dtype, torch.bfloat16)
        self.assertEqual(torch.count_nonzero(dz).item(), 0)


if __name__ == "__main__":
    unittest.main()

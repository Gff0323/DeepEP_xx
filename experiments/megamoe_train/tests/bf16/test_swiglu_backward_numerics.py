#!/usr/bin/env python3
"""CPU checks against forward activations and smooth-subgraph differences.

These check the numerical contract/reference, not execution of AscendC code.
BF16 tensor Casts use the usual identity-gradient convention; finite
differences are taken inside the FP32 SwiGLU subgraph, not across BF16 steps.
"""

import unittest
import numpy as np
from swiglu_reference import bf16_round, reference_packed_intermediates


def packed(gate, up, dhp, p, limit=0.0):
    gate, up, dhp, p = [bf16_round(v).reshape(-1) for v in (gate, up, dhp, p)]
    rows = len(gate)
    x = np.column_stack((gate, up)).astype(np.float32)
    # Sparse matrices expose the epilogue without an uncertain GEMM sum.
    w1 = np.zeros((2, 512, 2), dtype=np.float32)
    w1[:, :256, 0] = 1
    w1[:, 256:, 1] = 1
    w2 = np.zeros((2, 2, 256), dtype=np.float32)
    w2[:, 0, :] = 1
    dy = np.column_stack((dhp, np.zeros(rows, dtype=np.float32)))
    ids = np.zeros((rows, 1), dtype=np.int32)  # Expert 1 deliberately empty.
    return reference_packed_intermediates(
        x, ids, p[:, None], w1, w2, dy, rank=0, experts_per_rank=2, max_recv=rows, save_mode="bf16", swiglu_limit=limit
    )


class SwiGLUBackwardNumerics(unittest.TestCase):
    def test_rebuilt_h_matches_forward_counterexample(self):
        got = packed([1.0], [1.3125], [1.0], [1.0])
        self.assertEqual(float(got["Hp"][0, 0]), 0.9609375)
        # The former separate-SiLU path produced 0.95703125 instead.
        self.assertEqual(float(got["dpPartials"][0, 0]), 256 * 0.9609375)

    def test_gradients_against_smooth_subgraph_finite_differences(self):
        # Independent scalar derivative values rounded to BF16. The former
        # small-op intermediates gave 0.205078125 / 0.4921875 respectively.
        du_case = packed([1.0], [1.3125], [0.28125], [1.0])
        dg_case = packed([1.0], [1.3125], [0.40625], [1.0])
        self.assertEqual(float(du_case["gradZ"][0, 256]), 0.2060546875)
        self.assertEqual(float(dg_case["gradZ"][0, 0]), 0.494140625)
        # Includes positive/negative values, clamp boundaries, tails/full tile,
        # and an empty expert. No BF16 step is differentiated numerically.
        base_g = np.array([-2.0, -0.5, 0.0, 1.0, 1.5, 2.0], dtype=np.float32)
        base_u = np.array([0.75, -1.0, 1.3125, 1.3125, -1.5, 2.0], dtype=np.float32)
        for rows in (1, 65, 127, 128, 129):
            for limit in (0.0, 1.5):
                with self.subTest(rows=rows, limit=limit):
                    g, u = np.resize(base_g, rows), np.resize(base_u, rows)
                    p = np.resize(np.array([0.10009765625, 0.5, 1.0], np.float32), rows)
                    dhp = np.resize(np.array([0.75, -0.875, 0.3125], np.float32), rows)
                    got = packed(g, u, dhp, p, limit)
                    dh = bf16_round(p * dhp).astype(np.float64)
                    gc = np.minimum(g, limit) if limit else g
                    uc = np.clip(u, -limit, limit) if limit else u
                    gc, uc = gc.astype(np.float64), uc.astype(np.float64)

                    def smooth(gg, uu):
                        return dh * (gg / (1.0 + np.exp(-gg))) * uu

                    eps = 1e-5
                    dg = (smooth(gc + eps, uc) - smooth(gc - eps, uc)) / (2 * eps)
                    du = (smooth(gc, uc + eps) - smooth(gc, uc - eps)) / (2 * eps)
                    # Clamp endpoints pass gradients; outside is exactly zero.
                    if limit:
                        dg[g > limit] = 0
                        du[(u < -limit) | (u > limit)] = 0
                    actual_g, actual_u = np.split(got["gradZ"], 2, axis=1)
                    np.testing.assert_allclose(actual_g[:, 0], bf16_round(dg), rtol=0.008, atol=1e-7)
                    np.testing.assert_allclose(actual_u[:, 0], bf16_round(du), rtol=0.008, atol=1e-7)
                    if limit:
                        self.assertTrue(np.all(actual_g[g > limit] == 0))
                        self.assertTrue(np.all(actual_u[(u < -limit) | (u > limit)] == 0))

    def test_dh_bf16_boundary_is_retained(self):
        g, u, p, dhp = [1.0], [1.3125], [0.10009765625], [0.75]
        got = packed(g, u, dhp, p)
        dh = bf16_round(bf16_round(p) * bf16_round(dhp))
        silu = np.float32(1.0) / (np.float32(1.0) + np.exp(np.float32(-1.0)))
        self.assertEqual(float(got["gradZ"][0, 256]), float(bf16_round(dh * silu)[0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)

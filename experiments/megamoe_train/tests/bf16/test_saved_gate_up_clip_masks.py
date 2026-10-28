"""CPU format/coverage checks; device arithmetic is checked by the train smoke."""

import unittest
import numpy as np
from saved_gate_up_reference import pack_clip_masks, w8a8_reference
from test_workspace_layout import training_saved_state_size
from test_saved_gate_up_modes import load_bridge, Context, Operator, Tensor


class SavedClipMaskTests(unittest.TestCase):
    def test_boundary_direction_and_bit_order(self):
        g = np.zeros((2, 64), dtype=np.float32)
        u = g.copy()
        g[0, [0, 31, 32, 63]] = 10.0625
        g[1, :4] = [-100, 9.9375, 10, 10.0625]
        u[0, [0, 32]] = -10.0625
        u[0, [31, 63]] = 10.0625
        u[1, :6] = [-10.0625, -10, -9.9375, 9.9375, 10, 10.0625]
        gh, ul, uh = pack_clip_masks(g, u, 10)
        np.testing.assert_array_equal(gh, [[0x80000001, 0x80000001], [8, 0]])
        np.testing.assert_array_equal(ul, [[1, 1], [1, 0]])
        np.testing.assert_array_equal(uh, [[0x80000000, 0x80000000], [32, 0]])
        self.assertFalse(np.any(ul & uh))

    def test_fp32_limit_is_not_rounded_to_bf16(self):
        g = np.array([[1, 1.0078125] * 16], np.float32)
        gh, ul, uh = pack_clip_masks(g, -g, 1.001)
        np.testing.assert_array_equal(gh, [[0xAAAAAAAA]])
        np.testing.assert_array_equal(ul, gh)
        self.assertFalse(np.any(uh))
        # Rounding this limit up to BF16 would incorrectly unmark 1.0078125.
        gh, _, _ = pack_clip_masks(g, g, 1.006)
        np.testing.assert_array_equal(gh, [[0xAAAAAAAA]])

    def test_quantization_does_not_define_clip_branch(self):
        g = np.zeros((1, 32), np.float32)
        g[0, :2] = [10.0625, 16]
        _, decoded, _, _ = w8a8_reference().mxfp8_quantize(g, -1)
        self.assertEqual(decoded[0, 0], 10)
        gh, _, _ = pack_clip_masks(g, g, 10)
        decoded_gh, _, _ = pack_clip_masks(decoded, decoded, 10)
        self.assertEqual(int(gh[0, 0] & 1), 1)
        self.assertEqual(int(decoded_gh[0, 0] & 1), 0)

    def test_tail_bits_and_zero_replacement(self):
        for rows in (1, 127, 128, 129):
            for columns in (32, 64, 128, 256, 512):
                raw = np.full((rows, columns), 2, np.float32)
                storage = np.full((3, rows, (columns + 31) // 32), 0xFFFFFFFF, np.uint32)
                storage[:] = pack_clip_masks(raw, -raw, 1)
                self.assertTrue(np.all(storage[0] == 0xFFFFFFFF))
                storage[:] = pack_clip_masks(np.zeros_like(raw), np.zeros_like(raw), 1)
                self.assertFalse(np.any(storage))
        gh, _, _ = pack_clip_masks(np.ones((1, 33)) * 2, np.zeros((1, 33)), 1)
        np.testing.assert_array_equal(gh, [[0xFFFFFFFF, 1]])

    def test_tile_dma_owns_complete_disjoint_words(self):
        # Model byte DMA writes from 32B UB rows into compact GM at expert
        # bases which are deliberately not multiples of the 128-row tile.
        for width in (256, 512, 4096):
            counts = (0, 1, 127, 129, 0, 3)
            capacity = sum(counts) + 3
            gm = np.full((3, capacity, width // 32), 0xDEADBEEF, np.uint32)
            rng = np.random.default_rng(width)
            g = rng.integers(-2, 3, (sum(counts), width)).astype(np.float32)
            u = rng.integers(-2, 3, g.shape).astype(np.float32)
            row_base = 0
            for count in counts:
                for r in range(0, count, 128):
                    nr = min(128, count - r)
                    for col in range(0, width, 256):
                        masks = pack_clip_masks(
                            g[row_base + r : row_base + r + nr, col : col + 256],
                            u[row_base + r : row_base + r + nr, col : col + 256],
                            1,
                        )
                        for plane, words in enumerate(masks):
                            padded = np.full((nr, 8), 0xDEADBEEF, np.uint32)
                            padded[:, : words.shape[1]] = words
                            gm[plane, row_base + r : row_base + r + nr, col // 32 : col // 32 + words.shape[1]] = (
                                padded[:, : words.shape[1]]
                            )
                row_base += count
            np.testing.assert_array_equal(gm[:, : sum(counts)], np.stack(pack_clip_masks(g, u, 1)))
            self.assertTrue(np.all(gm[:, sum(counts) :] == 0xDEADBEEF))

    def test_format_allocation_and_per_call_metadata(self):
        args = (4096, 7168, 16, 8, 4, 65536, 8192)
        self.assertEqual(training_saved_state_size(*args, save_mxfp8=True, save_clip_masks=True), 1594497024)
        self.assertEqual(training_saved_state_size(*args, save_clip_masks=True), 2013927424)
        self.assertEqual(training_saved_state_size(*args, save_gate_up=False, save_clip_masks=True), 940185600)
        bridge, op = load_bridge(), Operator()
        contexts = []
        for mode, limit in (("mxfp8", 1.001), ("mxfp8", 0), ("bf16", 1), ("auto", 1), ("none", 1)):
            ctx, holder = Context(), {}
            bridge.forward(ctx, op, *(Tensor() for _ in range(5)), 65536, limit, mode, holder, None)
            self.assertEqual(holder["saved_gate_up_metadata"], ctx.saved_gate_up_metadata)
            contexts.append(ctx)
        self.assertEqual([x.saved_gate_up_metadata["layout_version"] for x in contexts], [2, 1, 1, 1, 1])
        self.assertEqual(contexts[0].saved_gate_up_metadata["clip_mask_kind"], "three_bitplanes_v1")
        self.assertEqual(contexts[0].saved_gate_up_metadata["normalized_limit"], float(np.float32(1.001)))


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
M_TILE = 128
N_TILE = 256


def ready_first_m_tile(ordinal: int, m_tiles: int, blocks_per_producer_round: int) -> int:
    tiles_per_ready_block = 2
    if blocks_per_producer_round <= 1 or m_tiles % tiles_per_ready_block != 0:
        return ordinal
    ready_blocks = m_tiles // tiles_per_ready_block
    if ready_blocks % blocks_per_producer_round != 0:
        return ordinal
    producer_partitions = ready_blocks // blocks_per_producer_round
    block_ordinal = ordinal // tiles_per_ready_block
    tile_in_block = ordinal % tiles_per_ready_block
    producer_round = block_ordinal // producer_partitions
    producer_partition = block_ordinal % producer_partitions
    return (producer_partition * blocks_per_producer_round + producer_round) * tiles_per_ready_block + tile_in_block


def adaptive_coord(
    task: int, m_tiles: int, n_tiles: int, width: int, ready_first: bool, blocks_per_producer_round: int = 2
) -> tuple[int, int]:
    tasks_per_full_stripe = m_tiles * width
    stripe = task // tasks_per_full_stripe
    task_in_stripe = task % tasks_per_full_stripe
    stripe_start = stripe * width
    actual_width = min(width, n_tiles - stripe_start)
    m_ordinal = task_in_stripe // actual_width
    m_tile = ready_first_m_tile(m_ordinal, m_tiles, blocks_per_producer_round) if ready_first else m_ordinal
    return m_tile, stripe_start + task_in_stripe % actual_width


class Gmm1ScheduleTest(unittest.TestCase):
    def test_ready_first_visits_each_m_tile_once(self) -> None:
        for blocks_per_round in (1, 2, 3, 4, 8):
            for m_tiles in range(1, 130):
                order = [ready_first_m_tile(i, m_tiles, blocks_per_round) for i in range(m_tiles)]
                self.assertEqual(sorted(order), list(range(m_tiles)))

    def test_ready_first_matches_producer_rounds(self) -> None:
        self.assertEqual(
            [ready_first_m_tile(i, 16, 2) for i in range(16)],
            [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15],
        )
        self.assertEqual(
            [ready_first_m_tile(i, 18, 3) for i in range(18)],
            [0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 16, 17],
        )
        self.assertEqual(
            [ready_first_m_tile(i, 16, 1) for i in range(16)],
            list(range(16)),
        )

    def test_adaptive_mapping_covers_every_tile(self) -> None:
        for m_tiles in (1, 2, 3, 7, 8, 15, 16, 31, 32, 65, 128):
            for n_tiles in (1, 2, 3, 4, 7, 16, 17):
                for width in (1, 2, 3, 4, 6):
                    if width > n_tiles:
                        continue
                    expected = {(m_tile, n_tile) for m_tile in range(m_tiles) for n_tile in range(n_tiles)}
                    for ready_first in (False, True):
                        for blocks_per_round in (1, 2, 3, 4):
                            actual = {
                                adaptive_coord(
                                    task,
                                    m_tiles,
                                    n_tiles,
                                    width,
                                    ready_first,
                                    blocks_per_round,
                                )
                                for task in range(m_tiles * n_tiles)
                            }
                            self.assertEqual(actual, expected)

    def test_build_interface_defaults_to_adaptive_auto(self) -> None:
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        for flag in (
            "DEEPEP_BF16_GMM1_ADAPTIVE_NSTRIPE=1",
            "DEEPEP_BF16_GMM1_READY_FIRST=1",
            "DEEPEP_BF16_GMM1_TILE_READY=1",
            "DEEPEP_BF16_GMM1_AUTO_SELECT=1",
        ):
            self.assertIn(flag, cmake)

    def test_adaptive_auto_falls_back_to_legacy_mapping(self) -> None:
        impl = (ROOT / "csrc" / "kernels" / "bf16" / "include" / "mega_moe_bf16_catlass_impl.h").read_text(
            encoding="utf-8"
        )
        self.assertIn("const bool useAdaptiveSchedule = useTileReady;", impl)
        self.assertIn(
            "nStripeWidth > 0 && useAdaptiveSchedule",
            impl,
        )


if __name__ == "__main__":
    unittest.main()

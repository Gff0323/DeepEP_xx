from __future__ import annotations

import numpy as np

from mega_moe_bf16_golden import _generate_expert_ids


def _assert_distinct_per_token(expert_ids: np.ndarray) -> None:
    ordered = np.sort(expert_ids, axis=1)
    assert np.all(ordered[:, 1:] != ordered[:, :-1])


def test_balanced_routing_preserves_the_existing_pattern() -> None:
    expert_ids = _generate_expert_ids(
        np.random.default_rng(831),
        token_count=512,
        topk=16,
        total_experts=32,
        routing_mode="balanced",
        routing_skew=6.0,
    )
    expected = (np.arange(512 * 16, dtype=np.int64) % 32).reshape(512, 16).astype(np.int32)
    assert np.array_equal(expert_ids, expected)


def test_imbalanced_routing_is_reproducible_distinct_and_rank_skewed() -> None:
    first = _generate_expert_ids(
        np.random.default_rng(831),
        token_count=4096,
        topk=16,
        total_experts=128,
        routing_mode="imbalanced",
        routing_skew=6.0,
    )
    second = _generate_expert_ids(
        np.random.default_rng(831),
        token_count=4096,
        topk=16,
        total_experts=128,
        routing_mode="imbalanced",
        routing_skew=6.0,
    )
    assert np.array_equal(first, second)
    _assert_distinct_per_token(first)

    rank_rows = np.bincount(first.reshape(-1) // 16, minlength=8)
    assert rank_rows.max() > 2 * rank_rows.min()

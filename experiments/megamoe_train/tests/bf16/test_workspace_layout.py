from __future__ import annotations


ALIGNMENT = 512
BF16_BYTES = 2
A5_UB_BYTES = 248 * 1024
BF16_TILE_M = 128
BF16_TILE_N = 256
BACKWARD_DP_PARTIAL_TILE_N = 256
TOPK_ROUTE_BATCH_ROWS = 1024
MAX_EP_RANKS = 1024
MAX_LOCAL_EXPERTS = 32
MAX_TOTAL_EXPERTS = 2048


def align_bytes(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def ordinary_bf16_peer_sync_offsets(_ep: int) -> tuple[int, int]:
    return 64 * 1024, 72 * 1024


def forward_dispatch_ub_size(
    ep: int,
    experts_per_rank: int,
    k: int,
    dispatch_rows: int = 2,
) -> tuple[int, int, int]:
    sync_bytes = align_bytes(ep * experts_per_rank, 128) * 4
    prev_sum_bytes = align_bytes(experts_per_rank * 4, 32)
    cumsum_bytes = ep * align_bytes(experts_per_rank * 4, 32)
    copy_elements = align_bytes(dispatch_rows * k, 32)
    copy_bytes = 2 * copy_elements * BF16_BYTES
    reset_flag_bytes = align_bytes((experts_per_rank * 16 * 4 * 2 + 31) // 32, 32)
    total = sync_bytes + prev_sum_bytes + cumsum_bytes + copy_bytes + 4 * 1024 + reset_flag_bytes
    return sync_bytes, prev_sum_bytes, total


def test_ordinary_bf16_ep_and_expert_boundaries_fit_control_layouts() -> None:
    assert 64 * 32 == MAX_TOTAL_EXPERTS
    assert MAX_EP_RANKS * 2 == MAX_TOTAL_EXPERTS
    assert MAX_LOCAL_EXPERTS == 32

    sync_bytes, prev_sum_bytes, total_ub = forward_dispatch_ub_size(64, 32, 7168)
    assert sync_bytes == 8 * 1024
    assert prev_sum_bytes == 128
    assert total_ub <= A5_UB_BYTES

    sync_count_offset, data_offset = ordinary_bf16_peer_sync_offsets(1024)
    assert 1024 * 64 <= sync_count_offset
    assert sync_count_offset == 64 * 1024
    assert data_offset == 72 * 1024
    assert 72 * 64 <= data_offset - sync_count_offset
    assert ordinary_bf16_peer_sync_offsets(2) == (64 * 1024, 72 * 1024)
    assert ordinary_bf16_peer_sync_offsets(768) == (64 * 1024, 72 * 1024)

    for experts_per_rank in range(1, MAX_LOCAL_EXPERTS + 1):
        ep = min(MAX_EP_RANKS, MAX_TOTAL_EXPERTS // experts_per_rank)
        _, prev_sum_bytes, total_ub = forward_dispatch_ub_size(ep, experts_per_rank, 7168)
        used_bytes = experts_per_rank * 4
        assert prev_sum_bytes >= used_bytes
        assert prev_sum_bytes - used_bytes < 32
        assert prev_sum_bytes % 32 == 0
        assert total_ub <= A5_UB_BYTES


def training_saved_state_size(
    m: int,
    k: int,
    topk: int,
    ep: int,
    experts_per_rank: int,
    max_recv_tokens: int,
    n: int,
    save_mxfp8: bool = False,
    save_gate_up: bool = True,
    save_clip_masks: bool = False,
    embed_dispatch_output: bool = True,
) -> int:
    size = align_bytes(max_recv_tokens * k * BF16_BYTES) if embed_dispatch_output else 0
    size += align_bytes(ep * experts_per_rank * 4)
    size += ((m + 255) // 256 * 256) * topk * 4 * 2
    size += ((ep * experts_per_rank + 127) // 128 * 128) * 4
    per_rank_stride = align_bytes(ep * experts_per_rank, 128)
    size += align_bytes(ep * per_rank_stride * 4)
    if embed_dispatch_output:
        size += align_bytes(max_recv_tokens * BF16_BYTES)
    size += align_bytes(256 * 4, 128)
    if not save_gate_up:
        return size
    data_bytes = 1 if save_mxfp8 else BF16_BYTES
    scale_bytes = 2 * align_bytes(max_recv_tokens * ((n // 2 + 31) // 32)) if save_mxfp8 else 0
    mask_bytes = 3 * align_bytes(max_recv_tokens * ((n // 2 + 31) // 32) * 4) if save_mxfp8 and save_clip_masks else 0
    return align_bytes(size) + 2 * align_bytes(max_recv_tokens * (n // 2) * data_bytes) + scale_bytes + mask_bytes


def dispatch_output_size(k: int, max_recv_tokens: int) -> int:
    return align_bytes(max_recv_tokens * k * BF16_BYTES) + align_bytes(max_recv_tokens * BF16_BYTES)


def forward_transient_workspace_size(
    n: int,
    experts_per_rank: int,
    max_recv_tokens: int,
) -> int:
    size = align_bytes(max_recv_tokens * (n // 2) * BF16_BYTES)
    size += 2 * align_bytes(experts_per_rank * 16 * 4, 128)
    return size


def workspace_sizes(
    m: int,
    k: int,
    n: int,
    topk: int,
    ep: int,
    experts_per_rank: int,
    max_output_tokens: int | None = None,
    peer_data_offset: int = 72 * 1024,
) -> tuple[int, int]:
    if max_output_tokens is None:
        max_output_tokens = m * topk * ep
    expanded_rows = (m + 255) // 256 * 256

    local = align_bytes(max_output_tokens * k * BF16_BYTES)
    local += align_bytes(max_output_tokens * (n // 2) * BF16_BYTES)
    local += align_bytes(ep * experts_per_rank * 4)
    local += expanded_rows * topk * 4 * 2
    local += ((ep * experts_per_rank + 127) // 128 * 128) * 4
    local += 2 * align_bytes(experts_per_rank * 16 * 4, 128)

    symmetric = peer_data_offset
    symmetric += ep * ((ep * experts_per_rank + 127) // 128 * 128) * 4
    symmetric += 2 * m * topk * k * BF16_BYTES
    return local, symmetric


def udma_workspace_sizes(
    m: int,
    k: int,
    n: int,
    topk: int,
    ep: int,
    experts_per_rank: int,
    max_recv_tokens: int | None = None,
) -> tuple[int, int]:
    if max_recv_tokens is None:
        max_recv_tokens = align_bytes(m * topk, BF16_TILE_M)
    _, symmetric = workspace_sizes(
        m,
        k,
        n,
        topk,
        ep,
        experts_per_rank,
        max_recv_tokens,
        peer_data_offset=60 * 1024,
    )
    shared_activation = align_bytes(max_recv_tokens * k * BF16_BYTES)
    local = align_bytes(max_recv_tokens * (n // 2) * BF16_BYTES)
    local += 2 * align_bytes(experts_per_rank * 16 * 4, 128)
    dispatch_signals = align_bytes(ep * experts_per_rank * 8)
    routed_topk_weights = align_bytes(m * topk * BF16_BYTES)
    combine_ready = align_bytes(experts_per_rank * ((max_recv_tokens + BF16_TILE_M - 1) // BF16_TILE_M) * 4)
    return (
        local + combine_ready,
        symmetric + dispatch_signals + ALIGNMENT + shared_activation + routed_topk_weights,
    )


def backward_workspace_size(
    max_output_tokens: int,
    k: int,
    n: int,
    experts_per_rank: int,
    save_mode: str = "none",
) -> int:
    f = n // 2
    dp_partial_tiles = (f + BACKWARD_DP_PARTIAL_TILE_N - 1) // BACKWARD_DP_PARTIAL_TILE_N
    size = align_bytes(max_output_tokens * k * BF16_BYTES)
    if save_mode == "none":
        size += align_bytes(max_output_tokens * f * BF16_BYTES)
        size += align_bytes(max_output_tokens * f * BF16_BYTES)
    elif save_mode not in ("bf16", "mxfp8"):
        raise ValueError(f"unsupported save_mode: {save_mode}")
    size += align_bytes(max_output_tokens * n * BF16_BYTES)
    size += align_bytes(max_output_tokens * f * BF16_BYTES)
    size += align_bytes(max_output_tokens * dp_partial_tiles * 4)
    size += align_bytes(experts_per_rank * 16 * 4, 128)
    size += align_bytes(4, 32)
    size += align_bytes(4, 32)
    size += align_bytes(experts_per_rank * 16 * 4, 128)  # GM dHp readiness
    size += 2 * align_bytes(experts_per_rank * 16 * 4, 128)  # dZ/dX readiness
    if save_mode != "none":
        size += align_bytes(experts_per_rank * 16 * 4, 128)  # all dO readers retired
    return size


def test_current_megamoe_shape_bf16_capacity() -> None:
    expected = {
        2: (81_816_576, 62_989_312),
        4: (163_605_504, 62_990_336),
        8: (327_183_360, 62_992_384),
    }
    for ep, sizes in expected.items():
        assert workspace_sizes(512, 5120, 3072, 6, ep, 16) == sizes


def test_bf16_udma_balanced_capacity_reuses_routing_workspace() -> None:
    expected = {
        2: (9_440_768, 94_441_472),
        4: (9_440_768, 94_442_496),
        8: (9_440_768, 94_445_056),
    }
    for ep, sizes in expected.items():
        assert udma_workspace_sizes(512, 5120, 3072, 6, ep, 16) == sizes


def test_bf16_udma_ep8_worst_case_capacity_fits_dynamic_peer_memory() -> None:
    max_recv_tokens = 4096 * 8 * 16
    _, peer_bytes = udma_workspace_sizes(4096, 3584, 6144, 16, 8, 16, max_recv_tokens)
    local_mem_bytes = 1024 * 1024 * 1024 + align_bytes(max_recv_tokens * 3584 * BF16_BYTES)
    assert peer_bytes == 4_697_818_624
    assert local_mem_bytes == 4_831_838_208
    assert peer_bytes < local_mem_bytes


def test_current_large_shape_one_gib_base_and_dynamic_udma_capacity() -> None:
    gib = 1024 * 1024 * 1024
    for ep in (2, 4, 8):
        max_recv_tokens = 4096 * ep * 16
        _, base_peer_bytes = workspace_sizes(4096, 3584, 6144, 16, ep, 16, max_recv_tokens)
        _, udma_peer_bytes = udma_workspace_sizes(4096, 3584, 6144, 16, ep, 16, max_recv_tokens)
        ptr_a_bytes = align_bytes(max_recv_tokens * 3584 * BF16_BYTES)
        assert base_peer_bytes < gib
        assert udma_peer_bytes < gib + ptr_a_bytes


def test_bf16_udma_token_row_is_contiguous_and_above_4k() -> None:
    row_bytes = 5120 * BF16_BYTES
    assert row_bytes == 10 * 1024
    assert row_bytes >= 4 * 1024


def test_bf16_udma_dispatch_offsets_are_expert_major_on_receiver() -> None:
    counts = [
        [[2, 1], [3, 2]],
        [[1, 4], [2, 3]],
    ]
    ep = len(counts)
    experts = len(counts[0][0])
    receiver_rows = [
        [None] * sum(counts[src][dst][expert] for expert in range(experts) for src in range(ep)) for dst in range(ep)
    ]

    for src in range(ep):
        src_offset = 0
        for dst in range(ep):
            for expert in range(experts):
                rows = counts[src][dst][expert]
                dst_offset = sum(
                    counts[prior_src][dst][prior_expert] for prior_expert in range(expert) for prior_src in range(ep)
                ) + sum(counts[prior_src][dst][expert] for prior_src in range(src))
                receiver_rows[dst][dst_offset : dst_offset + rows] = [(expert, src, row) for row in range(rows)]
                src_offset += rows

    for dst in range(ep):
        expected = [
            (expert, src, row)
            for expert in range(experts)
            for src in range(ep)
            for row in range(counts[src][dst][expert])
        ]
        assert receiver_rows[dst] == expected


def test_symmetric_buffer_counts_bf16_payload_bytes() -> None:
    _, symmetric = workspace_sizes(512, 5120, 3072, 6, 2, 16)
    bf16_payload = 2 * 512 * 6 * 5120 * BF16_BYTES
    assert symmetric >= bf16_payload
    assert symmetric - bf16_payload == 72 * 1024 + 2 * 128 * 4


def test_existing_weight_shapes_are_column_major_b_matrices() -> None:
    n, k = 3072, 5120
    output_column, reduction_index = 17, 23
    public_weight1_offset = output_column * k + reduction_index
    logical_b1_column_major_offset = reduction_index + output_column * k
    assert public_weight1_offset == logical_b1_column_major_offset

    hidden_output, swiglu_index = 29, 31
    swiglu_width = n // 2
    public_weight2_offset = hidden_output * swiglu_width + swiglu_index
    logical_b2_column_major_offset = swiglu_index + hidden_output * swiglu_width
    assert public_weight2_offset == logical_b2_column_major_offset


def test_current_shape_bf16_weight_bytes() -> None:
    experts_per_rank, n, k = 4, 8192, 7168
    weight_bytes = experts_per_rank * (n * k + k * (n // 2)) * BF16_BYTES
    assert weight_bytes == 704_643_072
    assert weight_bytes / 1024**2 == 672


def test_split_training_workspace_matches_legacy_total_size() -> None:
    cases = (
        (512, 3072, 5120, 6, 16),
        (4096, 8192, 7168, 16, 4),
    )
    for m, n, k, topk, experts_per_rank in cases:
        for ep in (2, 4, 8):
            capacities = {
                m * ep * min(experts_per_rank, topk),
                m * ep * topk,
            }
            for max_recv_tokens in capacities:
                legacy_workspace, _ = workspace_sizes(m, k, n, topk, ep, experts_per_rank, max_recv_tokens)
                legacy_workspace += align_bytes(max_recv_tokens * BF16_BYTES)
                per_rank_stride = align_bytes(ep * experts_per_rank, 128)
                legacy_workspace += align_bytes(ep * per_rank_stride * 4)
                legacy_workspace += align_bytes(256 * 4, 128)

                saved = training_saved_state_size(m, k, topk, ep, experts_per_rank, max_recv_tokens, n)
                transient = forward_transient_workspace_size(n, experts_per_rank, max_recv_tokens)
                assert saved + transient == legacy_workspace + 2 * align_bytes(max_recv_tokens * (n // 2) * BF16_BYTES)


def test_compact_saved_state_plus_dispatch_matches_embedded_layout() -> None:
    cases = (
        (512, 3072, 5120, 6, 2, 16, 4096),
        (4096, 8192, 7168, 16, 8, 4, 65536),
    )
    for m, n, k, topk, ep, experts_per_rank, max_recv_tokens in cases:
        dispatch = dispatch_output_size(k, max_recv_tokens)
        for save_mxfp8, save_gate_up, save_clip_masks in (
            (False, False, False),
            (False, True, False),
            (True, True, False),
            (True, True, True),
        ):
            embedded = training_saved_state_size(
                m,
                k,
                topk,
                ep,
                experts_per_rank,
                max_recv_tokens,
                n,
                save_mxfp8=save_mxfp8,
                save_gate_up=save_gate_up,
                save_clip_masks=save_clip_masks,
                embed_dispatch_output=True,
            )
            compact = training_saved_state_size(
                m,
                k,
                topk,
                ep,
                experts_per_rank,
                max_recv_tokens,
                n,
                save_mxfp8=save_mxfp8,
                save_gate_up=save_gate_up,
                save_clip_masks=save_clip_masks,
                embed_dispatch_output=False,
            )
            assert compact + dispatch == embedded


def test_training_saved_state_excludes_forward_only_buffers() -> None:
    m, n, k, topk, ep, experts_per_rank = 4096, 8192, 7168, 16, 2, 4
    max_recv_tokens = m * ep * min(experts_per_rank, topk)
    saved = training_saved_state_size(m, k, topk, ep, experts_per_rank, max_recv_tokens, n)
    transient = forward_transient_workspace_size(n, experts_per_rank, max_recv_tokens)

    ptr_a_bytes = align_bytes(max_recv_tokens * k * BF16_BYTES)
    ptr_a2_bytes = align_bytes(max_recv_tokens * (n // 2) * BF16_BYTES)
    assert saved >= ptr_a_bytes
    assert transient >= ptr_a2_bytes
    assert transient > ptr_a2_bytes


def test_saved_weight_layout_default_and_odd_capacity() -> None:
    assert training_saved_state_size(4096, 7168, 16, 8, 4, 65536, 8192) == 940_185_600 + 1024**3
    assert forward_transient_workspace_size(8192, 4, 65536) == 536_871_424
    for rows in (1, 15, 16, 17, 255, 256, 257):
        for ep in (1, 2, 8):
            saved = training_saved_state_size(33, 256, 4, ep, 3, rows, 512)
            transient = forward_transient_workspace_size(512, 3, rows)
            legacy, _ = workspace_sizes(33, 256, 512, 4, ep, 3, rows)
            assert saved + transient == legacy + align_bytes(rows * 2) + align_bytes(
                ep * align_bytes(ep * 3, 128) * 4
            ) + 1024 + 2 * align_bytes(rows * 256 * 2)


def test_saved_weight_reference_uses_source_permutations_and_empty_experts() -> None:
    import numpy as np
    from test_mega_moe import packed_weight_reference

    ids = [np.array([2, 0, 2, 0]), np.array([0, 2, 0, 2])]
    weights = [
        np.array([0x3F00, 0x3F01, 0x3F02, 0x3F03], dtype=np.uint16),
        np.array([0x3E00, 0x3E01, 0x3E02, 0x3E03], dtype=np.uint16),
    ]
    # Equal-expert sort order is intentionally different on the two sources.
    orders = [np.array([3, 1, 2, 0]), np.array([0, 2, 1, 3])]
    assert packed_weight_reference(ids, weights, orders, 0, 2).tolist() == [0x3F03, 0x3F01, 0x3E00, 0x3E02]
    assert packed_weight_reference(ids, weights, orders, 1, 2).tolist() == [0x3F02, 0x3F00, 0x3E01, 0x3E03]


def test_bf16_gmm_ping_pong_tiles_fit_a5_ub() -> None:
    one_tile_bytes = BF16_TILE_M * BF16_TILE_N * BF16_BYTES
    assert one_tile_bytes == 64 * 1024
    assert 2 * one_tile_bytes == 128 * 1024
    assert 2 * one_tile_bytes <= A5_UB_BYTES


def test_p1_dual_aiv_round_robin_carries_across_experts() -> None:
    """One-tile experts must not all collapse onto the same paired AIV."""
    expert_local_tile_counts = [1, 1, 2, 1, 3]
    next_target = 0
    targets_by_expert: list[list[int]] = []
    for tile_count in expert_local_tile_counts:
        expert_targets = []
        for _ in range(tile_count):
            expert_targets.append(next_target & 1)
            next_target += 1
        targets_by_expert.append(expert_targets)

    assert targets_by_expert == [[0], [1], [0, 1], [0], [1, 0, 1]]
    flattened = [target for expert in targets_by_expert for target in expert]
    assert abs(flattened.count(0) - flattened.count(1)) <= 1


def test_default_training_shapes_use_gmm1_output_width() -> None:
    from pathlib import Path
    from test_mega_moe import read_cases

    csv_path = Path(__file__).with_name("mega_moe_cases.csv")
    cases = read_cases(csv_path, ep=8, routing_mode="balanced")
    assert [(case["M"], case["N"], case["K"]) for case in cases] == [
        (4096, 8192, 7168),
    ]
    assert [case["N"] // 2 for case in cases] == [4096]
    assert all(case["TOPK"] == 16 for case in cases)
    assert all(case["expertPerRank"] == 4 for case in cases)
    assert all(case["maxRecvTokenNum"] == 4096 * 16 for case in cases)


def test_routing_mode_selects_receive_capacity_policy() -> None:
    from test_mega_moe import recv_capacity

    assert recv_capacity(4096, 8, "balanced") == 4096 * 16
    assert recv_capacity(4096, 8, "random") == 4096 * 8 * 4
    assert recv_capacity(4096, 8, "imbalanced") == 4096 * 8 * 4


def test_new_training_shape_peer_memory_budget_covers_layout() -> None:
    from test_mega_moe import BASE_LOCAL_MEM_BYTES, local_mem_bytes

    n, k, experts_per_rank = 8192, 7168, 4
    max_recv = 4096 * 16
    for ep in (4, 8, 16):
        _, base_peer = workspace_sizes(4096, k, n, 16, ep, experts_per_rank, max_recv)
        # The historical base helper excludes the early-weight peer buffer.
        base_peer += align_bytes(4096 * 16 * BF16_BYTES)
        base_budget = local_mem_bytes(k, m=4096, ep=ep)
        assert base_peer <= base_budget == BASE_LOCAL_MEM_BYTES


def test_nonbalanced_capacity_can_grow_beyond_fixed_memory_floor() -> None:
    from test_mega_moe import BASE_LOCAL_MEM_BYTES, local_mem_bytes

    # Balanced capacity for the default shape fits the fixed 2 GiB floor.
    assert local_mem_bytes(7168, m=4096, ep=8) == BASE_LOCAL_MEM_BYTES
    # The largest supported shape grows the peer layout beyond the floor.
    assert local_mem_bytes(7168, m=16384, ep=16) > BASE_LOCAL_MEM_BYTES


def test_backward_flag_clear_range_uses_absolute_start() -> None:
    """The flag offset is relative to the backward workspace region."""
    forward_bytes = 81_816_576
    backward_flag_offset = 10_000
    backward_bytes = 12_000
    total_bytes = forward_bytes + backward_bytes

    flag_start = forward_bytes + backward_flag_offset
    flag_bytes = total_bytes - flag_start

    assert flag_start + flag_bytes == total_bytes
    assert flag_bytes == backward_bytes - backward_flag_offset


def test_backward_dp_partials_replace_full_product_plane() -> None:
    rows, k, n, experts = 4096, 5120, 3072, 16
    optimized = backward_workspace_size(rows, k, n, experts)
    full_product_bytes = align_bytes(rows * (n // 2) * BF16_BYTES)
    partial_bytes = align_bytes(rows * (n // 512) * 4)
    assert partial_bytes == 98_304
    assert full_product_bytes == 12_582_912
    assert optimized > 0
    assert full_product_bytes - partial_bytes == 12_484_608


def test_saved_gate_up_removes_recompute_planes_from_backward_scratch() -> None:
    rows, k, n, experts = 4096, 5120, 3072, 16
    recompute = backward_workspace_size(rows, k, n, experts, "none")
    saved_bf16 = backward_workspace_size(rows, k, n, experts, "bf16")
    saved_mxfp8 = backward_workspace_size(rows, k, n, experts, "mxfp8")
    two_recompute_planes = 2 * align_bytes(rows * (n // 2) * BF16_BYTES)
    assert recompute - saved_bf16 == two_recompute_planes - align_bytes(experts * 16 * 4, 128)
    assert saved_bf16 == saved_mxfp8


def test_topk_route_aiv_partition_has_no_gap_or_overlap() -> None:
    aiv_count = 96
    for rank_rows in (1, 15, 16, 17, 257, 1024, 65_537):
        ownership_blocks = (rank_rows + 15) // 16
        blocks_per_core, extra_blocks = divmod(ownership_blocks, aiv_count)
        covered: list[int] = []
        for core in range(aiv_count):
            core_blocks = blocks_per_core + (core < extra_blocks)
            core_block_begin = core * blocks_per_core + min(core, extra_blocks)
            begin = core_block_begin * 16
            rows = min(core_blocks * 16, rank_rows - begin)
            if rows > 0:
                covered.extend(range(begin, begin + rows))
        assert covered == list(range(rank_rows))


def test_topk_route_batching_has_no_gap_or_overlap() -> None:
    for core_rows in (1, 1023, 1024, 1025, 4097):
        covered: list[int] = []
        for begin in range(0, core_rows, TOPK_ROUTE_BATCH_ROWS):
            rows = min(TOPK_ROUTE_BATCH_ROWS, core_rows - begin)
            covered.extend(range(begin, begin + rows))
        assert covered == list(range(core_rows))


def test_topk_partial_ub_rows_are_aligned_and_leave_register_guard() -> None:
    partial_capacity = 128 * 1024 // 4
    fp32_per_register = 64
    for partial_tiles in (1, 3, 6, 12, 32, 257):
        partial_stride = (partial_tiles + 7) // 8 * 8
        batch_rows = min(
            TOPK_ROUTE_BATCH_ROWS,
            (partial_capacity - (fp32_per_register - 1)) // partial_stride,
        )
        assert partial_stride * 4 % 32 == 0
        assert batch_rows > 0
        last_load_begin = (batch_rows - 1) * partial_stride
        assert last_load_begin + fp32_per_register <= partial_capacity

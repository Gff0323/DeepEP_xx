"""Pure NumPy SwiGLU numerical reference; no device-buffer capture or decoding."""

from __future__ import annotations

import numpy as np


def float32_to_bf16_bits(data: np.ndarray) -> np.ndarray:
    fp32 = np.asarray(data, dtype=np.float32)
    bits = fp32.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)
    return (rounded >> 16).astype(np.uint16)


def bf16_bits_to_float32(data: np.ndarray) -> np.ndarray:
    bits = np.asarray(data, dtype=np.uint16).astype(np.uint32) << 16
    return bits.view(np.float32)


def bf16_round(data: np.ndarray) -> np.ndarray:
    return bf16_bits_to_float32(float32_to_bf16_bits(data))


def reference_packed_intermediates(
    x: np.ndarray,
    expert_ids: np.ndarray,
    topk_weights: np.ndarray,
    w1: np.ndarray,
    w2: np.ndarray,
    grad_y: np.ndarray,
    rank: int,
    experts_per_rank: int,
    max_recv: int,
    save_mode: str = "bf16",
    swiglu_limit: float = 0.0,
) -> dict[str, np.ndarray | int]:
    """Pure NumPy reference for expert-packed BF16 arithmetic stages.

    Global tokens are rank-major. Dispatch collects source ranks in rank order,
    so np.argwhere() gives the expected packed route order for each expert.
    Every GM materialization point is rounded to BF16 like the device kernel.
    """
    f = w1.shape[1] // 2
    packed_do = []
    packed_gate = []
    packed_up = []
    packed_grad_z = []
    packed_dx = []
    packed_hp = []
    packed_dp_partials = []
    packed_routes = []
    counts = []

    for local_expert in range(experts_per_rank):
        expert = rank * experts_per_rank + local_expert
        routes = np.argwhere(expert_ids == expert)
        remaining = max_recv - sum(counts)
        routes = routes[: max(remaining, 0)]
        counts.append(int(routes.shape[0]))
        if routes.size == 0:
            continue

        tokens = routes[:, 0]
        slots = routes[:, 1]
        xr = x[tokens].astype(np.float32)
        p = topk_weights[tokens, slots].astype(np.float32)
        d_o = bf16_round(grad_y[tokens].astype(np.float32))

        z = bf16_round(xr @ w1[expert].astype(np.float32).T)
        raw_gate, raw_up = np.split(z, 2, axis=1)
        gate, up = raw_gate, raw_up
        if save_mode == "mxfp8":
            from saved_gate_up_reference import w8a8_reference

            _, gate, _, _ = w8a8_reference().mxfp8_quantize(gate, -1)
            _, up, _, _ = w8a8_reference().mxfp8_quantize(up, -1)
        # Saved planes contain raw/dequantized values before activation clamps.
        # Keep their reference separate from the clipped gradient operands.
        saved_gate, saved_up = gate, up
        gate_ind = np.ones_like(gate, dtype=np.float32)
        up_ind = np.ones_like(up, dtype=np.float32)
        if swiglu_limit > 0:
            limit = np.float32(swiglu_limit)
            gate_high = raw_gate > limit
            up_low = raw_up < -limit
            up_high = raw_up > limit
            gate = np.minimum(gate, limit)
            up = np.clip(up, -limit, limit)
            if save_mode == "mxfp8":
                gate = np.where(gate_high, limit, gate)
                up = np.where(up_low, -limit, np.where(up_high, limit, up))
            gate_ind = (~gate_high).astype(np.float32)
            up_ind = (~(up_low | up_high)).astype(np.float32)
        sigmoid = 1.0 / (1.0 + np.exp(-gate))
        # Match the forward epilogue's gate / (1 + exp(-gate)), keeping
        # SiLU and the Up product in FP32 until the BF16 H boundary.
        silu = gate / (1.0 + np.exp(-gate))
        h = bf16_round(silu * up)
        hp = bf16_round(p[:, None] * h)

        d_hp = bf16_round(d_o @ w2[expert].astype(np.float32))
        # Reconstructed H must be the same BF16 activation as forward H.
        # Keep dH's BF16 tensor boundary, but no standalone BF16 SiLU/dS.
        dp_products = bf16_round(d_hp * h)
        d_h = bf16_round(p[:, None] * d_hp)
        d_up = d_h * silu * up_ind
        d_gate = d_h * up * (sigmoid * (1.0 + gate * (1.0 - sigmoid))) * gate_ind
        grad_z = bf16_round(np.concatenate([d_gate, d_up], axis=1))

        packed_do.append(d_o)
        packed_gate.append(saved_gate)
        packed_up.append(saved_up)
        packed_grad_z.append(grad_z)
        # Saved modes replace each expert's retired dO plane with dX.
        packed_dx.append(bf16_round(grad_z @ w1[expert].astype(np.float32)))
        packed_hp.append(hp)
        # The optimized epilogue reduces each 256-wide F tile in FP32 and
        # stores one partial per packed row; P6 sums only these partials.
        packed_dp_partials.append(dp_products.reshape(dp_products.shape[0], -1, 256).sum(axis=2, dtype=np.float32))
        packed_routes.append(
            np.column_stack(
                [
                    np.full(routes.shape[0], expert, dtype=np.int64),
                    tokens.astype(np.int64),
                    slots.astype(np.int64),
                ]
            )
        )

    def concat(parts: list[np.ndarray], width: int) -> np.ndarray:
        if not parts:
            return np.empty((0, width), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)

    return {
        "valid_rows": sum(counts),
        "counts": np.asarray(counts, dtype=np.int32),
        "routes": (np.concatenate(packed_routes, axis=0) if packed_routes else np.empty((0, 3), dtype=np.int64)),
        "ptrDO": concat(packed_do, x.shape[1]),
        "ptrDX": concat(packed_dx, x.shape[1]),
        "gate": concat(packed_gate, f),
        "up": concat(packed_up, f),
        "gradZ": concat(packed_grad_z, f * 2),
        "Hp": concat(packed_hp, f),
        "dpPartials": concat(packed_dp_partials, (f + 255) // 256),
    }

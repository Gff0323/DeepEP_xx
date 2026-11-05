"""Independent NPU references using BigOp's grouped-GEMM building blocks.

The MegaMoE reference follows its declared FP32 activation and ordered combine
boundaries. Original BigOp BF16 activation/scatter-add remains the independent
ATK error baseline. Both use the same public input snapshots.
"""

import torch
import torch.nn.functional as F


def _grouped_matmul(input_tokens, weight, tokens_per_expert):
    """BigOp split-M grouped GEMM; group_list contains counts, not prefix sums."""
    import torch_npu

    group_list = tokens_per_expert.to(torch.int64).to(input_tokens.device)
    # A single split-M weight retains its expert axis, including E=1.
    weight_list = (
        [weight.contiguous()]
        if weight.shape[0] == 1
        else [weight[expert].contiguous() for expert in range(weight.shape[0])]
    )
    return torch_npu.npu_grouped_matmul(
        [input_tokens.contiguous()],
        weight_list,
        group_list=group_list,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]


def _grouped_wgrad(input_tokens, grad_output, tokens_per_expert):
    """BigOp grouped weight gradient; preserve the transposed left-hand view."""
    import torch_npu

    group_list = tokens_per_expert.to(torch.int64).to(input_tokens.device)
    return torch_npu.npu_grouped_matmul(
        [input_tokens.T],
        [grad_output],
        group_list=group_list,
        split_item=3,
        group_type=2,
        group_list_type=1,
    )[0]


def _round_to_e4m3fn(value):
    """BigOp saturated E4M3FN arithmetic with ties-to-even rounding."""
    magnitude = value.abs().clamp(max=448.0)
    tiny = torch.finfo(torch.float32).tiny
    exponent = torch.floor(torch.log2(magnitude.clamp_min(tiny))).clamp(-6, 8)
    quantum = torch.where(magnitude < 2.0**-6, 2.0**-9, torch.exp2(exponent - 3.0))
    rounded = (torch.round(magnitude / quantum) * quantum).clamp(max=448.0)
    return torch.where(value < 0, -rounded, rounded)


def _mxfp8_saved_roundtrip(value):
    """BigOp BF16 reciprocal scaling, E4M3 RINT and BF16 decoding per 32 values."""
    if value.shape[-1] % 32:
        raise ValueError("MXFP8 saved-state baseline requires last dimension % 32 == 0")
    blocks = value.reshape(-1, value.shape[-1] // 32, 32)
    maximum = blocks.float().abs().amax(dim=-1, keepdim=True)
    exponent = (torch.floor(torch.log2(maximum.clamp_min(torch.finfo(torch.float32).tiny))) - 8.0).clamp(-127, 127)
    exponent = torch.where(maximum == 0, -127.0, exponent)
    scale = torch.exp2(exponent)
    reciprocal = torch.exp2(-exponent).to(torch.bfloat16)
    normalized = (blocks * reciprocal).to(torch.bfloat16).float()
    return (_round_to_e4m3fn(normalized) * scale).to(torch.bfloat16).reshape(value.shape)


def _swiglu(gate_up, limit, raw=None):
    """BigOp BF16 small-op graph; raw forward values supply MXFP8 clip decisions."""
    gate, up = gate_up.chunk(2, dim=-1)
    # Larger limits cannot clip finite BF16 operands. This also handles the
    # native FLT_MAX disabled sentinel without overflowing a BF16 scalar cast.
    if 0 < limit < torch.finfo(torch.bfloat16).max:
        if raw is None:
            gate, up = gate.clamp_max(limit), up.clamp(-limit, limit)
        else:
            raw_gate, raw_up = raw.chunk(2, dim=-1)
            gate = torch.where(raw_gate > limit, limit, gate)
            up = torch.where(raw_up < -limit, -limit, up)
            up = torch.where(raw_up > limit, limit, up)
    return F.silu(gate) * up


def _activation_operands(gate_up, limit, raw=None):
    """Clip FP32 operands; MXFP8 derivative masks use raw forward decisions."""
    gate, up = gate_up.float().chunk(2, dim=-1)
    gate_mask = up_mask = 1.0
    if 0 < limit < torch.finfo(torch.float32).max:
        raw_gate, raw_up = (gate_up if raw is None else raw).float().chunk(2, dim=-1)
        gate_mask = raw_gate <= limit
        up_mask = (raw_up >= -limit) & (raw_up <= limit)
        gate = torch.where(raw_gate > limit, limit, gate)
        up = torch.where(raw_up < -limit, -limit, up)
        up = torch.where(raw_up > limit, limit, up)
        # Decoding can cross a boundary even when the raw value was inside.
        # Clip the reconstructed operand, retaining the original gradient mask.
        gate, up = gate.clamp_max(limit), up.clamp(-limit, limit)
    return gate, up, gate_mask, up_mask


def _swiglu_fp32(gate_up, limit, raw=None):
    gate, up, _, _ = _activation_operands(gate_up, limit, raw)
    silu = gate / (1.0 + torch.exp(-gate))
    return (silu * up).to(gate_up.dtype)


def _swiglu_backward_fp32(grad_h, gate_up, limit, raw=None):
    """Differentiate FP32 SiLU/Up; keep dH and final dGate/dUp BF16 boundaries."""
    gate, up, gate_mask, up_mask = _activation_operands(gate_up, limit, raw)
    denominator = 1.0 + torch.exp(-gate)
    silu = gate / denominator
    sigmoid = torch.ones_like(gate) / denominator
    grad_h = grad_h.float()
    dg = (grad_h * up) * (sigmoid * (1.0 + gate * (1.0 - sigmoid))) * gate_mask
    du = grad_h * silu * up_mask
    return torch.cat((dg, du), dim=-1).to(gate_up.dtype)


@torch.no_grad()
def run_baseline(inputs, device, expert_start, save_mode="bf16", swiglu_limit=0.0, *, reference="megamoe", trace=None):
    """Run the selected reference with public w1=[E,2F,H], w2=[E,H,F].

    Pack expert-major, then source-rank/token order, as BigOp's second
    permutation does. MegaMoE uses FP32 activation/derivatives; BigOp retains
    its BF16 small-op graph and autograd activation backward. Forward stays
    uncompressed in every save mode. Backward optionally rounds saved Gate/Up.
    """
    if save_mode not in ("none", "bf16", "mxfp8"):
        raise ValueError(f"Unresolved save mode: {save_mode}")
    if reference not in ("megamoe", "bigop"):
        raise ValueError(f"Unknown NPU reference: {reference}")
    activation = _swiglu_fp32 if reference == "megamoe" else _swiglu
    ids = inputs["ids"]
    indices = [(ids == expert_start + expert).nonzero(as_tuple=True) for expert in range(len(inputs["w1"]))]
    counts = torch.tensor([rows.numel() for rows, _ in indices], dtype=torch.int64, device="cpu")
    rows = torch.cat([rows for rows, _ in indices])
    slots = torch.cat([slots for _, slots in indices])
    result = {
        "ids": ids,
        "routes": [],
        "counts": counts,
        "reference": reference,
    }
    if trace is not None:
        trace.update(rows=rows.clone(), slots=slots.clone(), counts=counts.clone())
    if not rows.numel():
        result.update(
            dw1=torch.zeros(inputs["w1"].shape, dtype=torch.bfloat16, device="cpu"),
            dw2=torch.zeros(inputs["w2"].shape, dtype=torch.bfloat16, device="cpu"),
        )
        return result

    def tensor(value):
        return value.to(device=device, dtype=torch.bfloat16).contiguous()

    x, p, dy = tensor(inputs["x"][rows]), tensor(inputs["p"][rows, slots, None]), tensor(inputs["dy"][rows])
    weight1 = tensor(inputs["w1"].transpose(1, 2))
    weight2 = tensor(inputs["w2"].transpose(1, 2))
    forward_z = _grouped_matmul(x, weight1, counts)
    forward_hp = activation(forward_z, swiglu_limit) * p
    y = _grouped_matmul(forward_hp, weight2, counts)

    raw_z = _grouped_matmul(x, weight1, counts)
    z = _mxfp8_saved_roundtrip(raw_z) if save_mode == "mxfp8" else raw_z
    raw = raw_z if save_mode == "mxfp8" else None
    h = activation(z, swiglu_limit, raw)
    hp = h * p
    dhp = _grouped_matmul(dy, weight2.transpose(1, 2).contiguous(), counts)
    dw2 = _grouped_wgrad(hp, dy, counts).reshape_as(weight2).transpose(1, 2).contiguous()
    # MegaMoE keeps the dot product in FP32, using the BF16 H from forward.
    # Original BigOp retains its independent BF16 elementwise product.
    dp = (dhp.float() * h.float()).sum(dim=-1).to(dhp.dtype) if reference == "megamoe" else (dhp * h).sum(dim=-1)
    grad_swiglu = dhp * p
    if reference == "megamoe":
        dz = _swiglu_backward_fp32(grad_swiglu, z, swiglu_limit, raw)
    else:
        with torch.enable_grad():
            replay_z = z.detach().requires_grad_(True)
            replay_output = _swiglu(replay_z, swiglu_limit, raw)
            (dz,) = torch.autograd.grad(replay_output, replay_z, grad_swiglu)
    dx = _grouped_matmul(dz, weight1.transpose(1, 2).contiguous(), counts)
    dw1 = _grouped_wgrad(x, dz, counts).reshape_as(weight1).transpose(1, 2).contiguous()
    if trace is not None:
        # These are reference-side NPU intermediates, not fused-kernel dumps.
        trace.update(
            (name, value.detach().cpu().clone())
            for name, value in dict(raw_z=raw_z, forward_y=y, z=z, h=h, hp=hp, dhp=dhp, dp=dp, dz=dz).items()
        )
    result.update(dw1=dw1.cpu(), dw2=dw2.cpu())
    y, dx, dp = y.cpu(), dx.cpu(), dp.cpu()
    offset = 0
    for expert, (expert_rows, expert_slots) in enumerate(indices):
        stop = offset + expert_rows.numel()
        if stop > offset:
            result["routes"].append(
                {
                    "rows": expert_rows,
                    "slots": expert_slots,
                    "y": y[offset:stop],
                    "dx": dx[offset:stop],
                    "dp": dp[offset:stop],
                }
            )
        else:
            result["dw1"][expert].zero_()
            result["dw2"][expert].zero_()
        offset = stop
    return result


def _unpermute(tokens, rows, token_count):
    """BigOp unpermute: scatter_add into the output tensor's own dtype."""
    output = torch.zeros((token_count, tokens.shape[-1]), dtype=tokens.dtype, device=tokens.device)
    output.scatter_add_(0, rows[:, None].expand(-1, tokens.shape[-1]), tokens)
    return output


@torch.no_grad()
def unpermute_routes(buffer, ids, device, *, reference="megamoe"):
    """Combine transported routes using the selected numerical contract.

    buffer=[M,TopK,2H+1] contains exact BF16 y/dx/dp values carried as FP32.
    MegaMoE combines y and dx in TopK order in FP32, with one final BF16 cast. BigOp retains NPU BF16 scatter_add.
    """
    m, topk = ids.shape
    hidden = (buffer.shape[-1] - 1) // 2
    result = {"dtopk_weights": buffer[:, :, -1].to(torch.bfloat16).clone()}
    if reference == "megamoe":
        routes = buffer[:, :, : 2 * hidden].to(device=device, dtype=torch.bfloat16)
        y = torch.zeros((m, hidden), dtype=torch.float32, device=device)
        dx = torch.zeros((m, hidden), dtype=torch.float32, device=device)
        for slot in range(topk):
            y += routes[:, slot, :hidden].float()
            dx += routes[:, slot, hidden:].float()
        return {**result, "y": y.to(torch.bfloat16).cpu(), "dx": dx.to(torch.bfloat16).cpu()}
    if reference != "bigop":
        raise ValueError(f"Unknown NPU reference: {reference}")
    order = ids.reshape(-1).argsort(stable=True)
    rows = torch.arange(m, device="cpu").repeat_interleave(topk)[order].to(device)
    packed = buffer.reshape(m * topk, -1)[order].to(device=device, dtype=torch.bfloat16)
    route_y, route_dx = packed[:, :hidden].contiguous(), packed[:, hidden : 2 * hidden].contiguous()
    return {
        **result,
        "y": _unpermute(route_y, rows, m).cpu(),
        "dx": _unpermute(route_dx, rows, m).cpu(),
    }

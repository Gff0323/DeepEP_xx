"""Two-rank real-device lifetime/stream/queued-forward regression.

Launch with torchrun --nproc-per-node=2 after installing the training wheel.
This compares repeated calls and gradients, not an independent golden.
"""

import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from deep_ep_experimental.megamoe_train import MegaMoeBF16Train


def expect_error(action, message):
    try:
        action()
    except RuntimeError as error:
        assert message in str(error), error
    else:
        raise AssertionError(f"Expected rejection: {message}")


def main():
    rank = int(os.environ["RANK"])
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("gloo")
    group = dist.group.WORLD
    size = dist.get_world_size()
    torch.manual_seed(20260825)
    device = f"npu:{torch.npu.current_device()}"
    m, h, f, e, topk = 129, 256, 512, 2, 4
    ids = (torch.arange(m * topk, dtype=torch.int32).reshape(m, topk) % (size * e)).to(device)
    templates = [
        (torch.randn(m, h) * 0.1).to(device=device, dtype=torch.bfloat16),
        torch.full((m, topk), 0.25, dtype=torch.bfloat16, device=device),
        (torch.randn(e, 2 * f, h) * 0.02).to(device=device, dtype=torch.bfloat16),
        (torch.randn(e, h, f) * 0.02).to(device=device, dtype=torch.bfloat16),
    ]
    dy = (torch.randn(m, h) * 0.1).to(device=device, dtype=torch.bfloat16)
    capacity = 64 << 20
    modes = [("bf16", 0.0), ("none", 0.05), ("mxfp8", 0.05), ("auto", 0.0)]

    def forward(op, mode, limit):
        values = [value.clone().requires_grad_() for value in templates]
        x, weights, w1, w2 = values
        y, counts = op(x, ids, weights, w1, w2, m * size * e, save_gmm1_output=mode, swiglu_limit=limit)
        return values, y, counts

    def capture(values, y, counts):
        return [y.detach().cpu(), counts.cpu(), *(value.grad.cpu() for value in values)]

    op = MegaMoeBF16Train(group, capacity)
    expect_error(lambda: MegaMoeBF16Train(group, capacity), "already owns")
    sequential = []
    for mode, limit in modes:
        values, y, counts = forward(op, mode, limit)
        (y * dy).sum().backward()
        sequential.append(capture(values, y, counts))
    pending = [forward(op, mode, limit) for mode, limit in modes]
    expect_error(op.close, "pending autograd")
    with torch.npu.stream(torch.npu.Stream()):
        expect_error(lambda: forward(op, "bf16", 0.0), "construction stream")
    for index in reversed(range(len(pending))):
        values, y, counts = pending[index]
        (y * dy).sum().backward()
        actual = capture(values, y, counts)
        assert all(torch.equal(a, b) for a, b in zip(actual, sequential[index])), (rank, modes[index])
    # Legacy weight layouts must return gradients in those same input layouts.
    values = [value.clone() for value in templates]
    values[2] = values[2].transpose(1, 2).contiguous()
    values[3] = values[3].transpose(1, 2).contiguous()
    values = [value.requires_grad_() for value in values]
    y, counts = op(values[0], ids, values[1], values[2], values[3], m * size * e)
    (y * dy).sum().backward()
    actual = capture(values, y, counts)
    actual[4], actual[5] = actual[4].transpose(1, 2), actual[5].transpose(1, 2)
    assert all(torch.equal(a, b) for a, b in zip(actual, sequential[0])), (rank, "legacy layouts")
    for iteration in range(32):
        values, y, counts = forward(op, "bf16", 0.0)
        (y * dy).sum().backward()
    assert all(torch.equal(a, b) for a, b in zip(capture(values, y, counts), sequential[0])), (rank, "32 iterations")
    op.close()
    op.close()
    expect_error(lambda: forward(op, "bf16", 0.0), "closed")
    del pending, y, values, op
    # Recreating a closed owner must not retain stale generation/context state.
    op = MegaMoeBF16Train(group, capacity)
    values, y, counts = forward(op, "bf16", 0.0)
    (y * dy).sum().backward()
    assert all(torch.equal(a, b) for a, b in zip(capture(values, y, counts), sequential[0]))
    op.close()
    dist.destroy_process_group()
    print(f"[LIFECYCLE] rank={rank} PASS", flush=True)


if __name__ == "__main__":
    main()

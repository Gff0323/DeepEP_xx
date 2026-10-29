# MegaMoE training (experimental, Ascend950)

`deep_ep_experimental.megamoe_train.MegaMoeBF16Train` fuses routing, dispatch,
GMM1, SwiGLU, GMM2 and combine, with an autograd backward for `x`, routing
weights, `w1` and `w2`. Integer expert IDs and expert counts are not differentiable.
It is designed for **Ascend950**. Other Ascend generations are unsupported.

The BF16 source is frozen at **b8cff148**, not a moving `train` branch. See
[provenance](NOTICE.md), [migration plan (中文)](docs/MIGRATION_PLAN_ZH.md),
[validation](docs/VALIDATION.md) and [numerical semantics](docs/BF16_SWIGLU_NUMERICS.md).
The current API, code architecture, autograd bridge and resource lifetime are
described in the [framework design (中文)](docs/BF16_FRAMEWORK_DESIGN_ZH.md).
This is a separate training component alongside `experiments/megamoe` (inference).
BF16 has its own Host directory, Python module, kernels and tests. Future
quantized training variants belong in this component and can add peer
implementations when available; this BF16 import does not implement them.

## Build and installation

Use the root build workflow on Linux, CANN 9.2.0, Ascend950, Python >=3.11 and a
compatible PyTorch/torch_npu installation. Initialize the pinned submodules:

```bash
git submodule update --init third-party/shmem third-party/catlass
source /path/to/cann/set_env.sh
python -m pip install -r requirements-build.txt
python scripts/build_shmem.py
export SHMEM_ROOT="$PWD/third-party/shmem/install/shmem"
export CATLASS_ROOT="$PWD/third-party/catlass"
export LD_LIBRARY_PATH="$SHMEM_ROOT/lib:$LD_LIBRARY_PATH"
export DEEPEP_NPU_ARCH=Ascend950
export DEEPEP_BUILD_MEGAMOE_TRAIN=ON
python -m build --wheel --no-isolation
python -m pip install --force-reinstall --no-deps dist/*.whl
```

The training option defaults to OFF. It is independent of
`DEEPEP_BUILD_MEGAMOE` and `DEEPEP_BUILD_EP_GMM_FUSED`. Use a wheel, not an editable
installation. The root lock file selects CATLASS and SHMEM; training does not
embed another SHMEM or load the inference kernel SO. No CATCCOS is required.
Set runtime SDK library paths to the same SHMEM/CANN used for the wheel.
The installed `deep_ep/_build_info.json` records the actual build inputs.
Avoid empty entries (including a trailing colon) in `PYTHONPATH`: they add the
current source directory and can shadow the installed wheel's native extension.

## Usage and lifetime

For every rank in an existing EP process group, select the local device before
constructing the module. All ranks must use the same unused SHMEM bootstrap
endpoint, capacity, shape, routing interpretation, save mode and call order.

```python
import os
import torch
import torch_npu
from deep_ep_experimental.megamoe_train import MegaMoeBF16Train

torch.npu.set_device(local_rank)
os.environ["DEEPEP_SHMEM_ENDPOINT"] = "tcp://192.0.2.1:29862"  # reachable rank-0 host
op = MegaMoeBF16Train(ep_group, local_mem_size=1 << 30)
y, counts = op(x, topk_ids, topk_weights, w1, w2, max_recv_tokens,
               save_gmm1_output="bf16", swiglu_limit=0.0)
(y * grad_y).sum().backward()
op.close()  # collective, before destroying ep_group
```

Pass an existing ProcessGroup or a `deep_ep` StoreGroup as `ep_group`; the
caller owns that group. The source `ShmemInstance` is replaced by the root
Session owner; do not construct a second DeepEP buffer or inference runtime
concurrently in the same process.
The new Session selects MTE; the source test launcher selected UDMA at SHMEM
initialization. Current device qualification is single-node. Multi-node
transport, EP subgroups and larger EP sizes need separate qualification before
deployment; the API's upper rank bound is not a tested topology guarantee.

- `x`: `[M,H]`, `topk_ids`: `[M,T]` int32, `topk_weights`: `[M,T]` BF16.
  Tensor inputs are contiguous and reside on the construction NPU.
- Canonical `w1=[E,2F,H]`, `w2=[E,H,F]`; source legacy layouts
  `w1=[E,H,2F]`, `w2=[E,F,H]` remain supported. Gradients use the input layouts.
- `max_recv_tokens` bounds received routed rows on each rank. The conservative
  bound is `EP*M*min(E,T)` for distinct IDs within each token; use actual route
  demand only when guaranteed. Duplicate IDs are not part of the supported
  routing contract. Local symmetric capacity is in bytes, aligned to 2 MiB;
  Host checks that the complete communication layout fits it.
- Calls must use the construction stream and be serialized in identical order
  across ranks. Keep weights unmodified until their corresponding backward.
  Saved state and packed dispatch output are independent per forward; scratch
  is transient. Multiple queued forwards can be followed by their backwards in
  a consistent order. Cross-stream/concurrent calls and higher-order derivatives
  are unsupported. `retain_graph=True` requires keeping the operator open for
  every subsequent backward.
- `close()` synchronizes the launch stream and releases the root Session
  collectively. Callers must complete all required backwards or discard unused
  graphs before close; the operator does not track pending graphs. Garbage
  collection never performs collective SHMEM cleanup. After a device/collective
  failure restart the job.
- Public `forward` retains data required for backward. Device runners use the
  same public interface and validate final outputs, counts and gradients.

## Saved formats and numerical boundaries

| Mode | Gate/Up used by backward |
| --- | --- |
| `bf16` (default) | Saved BF16 values |
| `none` | Recomputed GMM1 |
| `mxfp8` | Saved MXFP8, plus three clip-mask bitplanes when clipping is enabled |
| `auto` | MXFP8 without clipping, BF16 with clipping |

`swiglu_limit<=0` disables clipping; positive finite limits preserve the source
semantics. MXFP8 here describes saved activations; matrix computation remains
BF16. Default GMM1 scheduling is adaptive-auto, SwiGLU decoupling is auto, and
backward dispatch splitting follows the source's local-expert cutoff (<=7).
See [dispatch policy](docs/BF16_DISPATCH_READY_DEFAULTS.md).

**Known issue retained:** the EP2/M65/H7168/F4096/E2/TopK4 BF16 intermediate
`ptrDX` check has a small number of original-threshold failures, reproduced on
the source and older versions. Do not loosen thresholds or count that case as
an all-precision pass. Final-gradient acceptance is separate from intermediate
acceptance. Actual model loss convergence is not established by an operator
test using `(y * grad_y).sum()`; framework short-training review remains required.

## Tests

```bash
python -m pytest -q experiments/megamoe_train/tests/bf16
# On an otherwise idle Ascend950 node, after installing the wheel:
python experiments/megamoe_train/tests/bf16/test_mega_moe_backward.py 0,1 \
  --seq 129 --hidden 256 --gmm-n 1024 --topk 4 --expert-per-rank 2 \
  --routing-mode balanced --save-gmm1-output bf16 \
  --shmem-ip-port tcp://127.0.0.1:29862
```

Repeat with `none`, `mxfp8`, `auto` and `--swiglu-limit 0.05` as appropriate.
The precision runner retains source thresholds and reports each rank. CPU
layout/reference tests do not qualify NPU functionality. Record the exact device
cases, dependency versions and build artifacts before qualifying a deployment.

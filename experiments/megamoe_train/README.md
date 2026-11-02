# MegaMoE training (experimental, Ascend950)

`deep_ep_experimental.megamoe_train.MegaMoeBF16Train` fuses routing, dispatch,
GMM1, SwiGLU, GMM2 and combine, with an autograd backward for `x`, routing
weights, `w1` and `w2`. Integer expert IDs and expert counts are not differentiable.
It is designed for **Ascend950**. Other Ascend generations are unsupported.

The initial BF16 implementation derives from **b8cff148**. See
[provenance](NOTICE.md) for attribution and
[numerical semantics](docs/BF16_SWIGLU_NUMERICS.md) for the computation contract.
The API, code architecture, autograd bridge and resource lifetime are described
in the [framework design (中文)](docs/BF16_FRAMEWORK_DESIGN_ZH.md).
This is a separate training component alongside `experiments/megamoe` (inference).
BF16 has its own Host directory, Python module, kernels and tests. Quantized
training implementations are not provided by this component.

Device headers use `.hpp` and are grouped by responsibility under
`csrc/kernels/bf16`: `catlass`, `epilogue`, `layout`, `schedule`, `routing` and
`common`. The root contains the forward/backward entry sources and controller
headers. Imported routing sources live under `routing/moe_init_routing_v3`;
SDK headers keep their original `.h` names. Host-only tiling helpers live in
`csrc/bf16`. Forward-only implementations use names matching their backward
counterparts; the public training interface is `MegaMoeBF16Train`.

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

`DEEPEP_BF16_FORCE_EXPERT_READY` is the single forward/backward dispatch fallback
build option. It accepts `ON` or `OFF` as an environment variable in the wheel
build, or as a CMake option (`-DDEEPEP_BF16_FORCE_EXPERT_READY=ON`). The default is
`OFF`, using M192/Ready384 forward dispatch and the saved-mode backward
row-ready policy. `ON` makes both directions wait for complete-expert dispatch:
forward disables tile-ready metadata and backward uses `splitExperts=0`.
The build supplies the same C++ value (`0` or `1`) to Host and Device.
The forced build takes precedence over the runtime
`MEGAMOE_BF16_BWD_DISPATCH_SPLIT_EXPERTS` override; changing that variable cannot
re-enable backward splitting. SwiGLU selection and saved formats are independent
of this option. For example, set `export DEEPEP_BF16_FORCE_EXPERT_READY=ON` before
the wheel build above to build the fallback.

Rebuild and reinstall the complete wheel after changing Host or device sources;
the Host extension and device library must come from the same build.
All EP ranks must use the same build and readiness option. Both readiness modes
use `MegaMoeBF16Train`, the `_C` Host extension and
`libdeepep_megamoe_train_bf16.so`.
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
caller owns that group. The training Runtime owns a root Session and selects
the MTE backend. Do not construct a second DeepEP buffer or inference runtime
concurrently in the same process.
Device qualification is single-node. ProcessGroup-based training integration,
multi-node transport, EP subgroups and larger EP sizes need separate
qualification before deployment; StoreGroup test coverage and the API's upper
rank bound are not guarantees for those integration paths or topologies.

- `x`: `[M,H]`, `topk_ids`: `[M,T]` int32, `topk_weights`: `[M,T]` BF16.
  Tensor inputs are contiguous and reside on the construction NPU.
- Canonical `w1=[E,2F,H]`, `w2=[E,H,F]`; alternative layouts
  `w1=[E,H,2F]`, `w2=[E,F,H]` are also supported. Gradients use the input layouts.
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

`swiglu_limit<=0` disables clipping; positive finite limits enable the Gate/Up
clipping defined by the numerical contract. MXFP8 describes saved activations;
matrix computation uses BF16. Default GMM1 scheduling is adaptive-auto,
SwiGLU decoupling is auto, and
backward dispatch splitting is enabled for eligible saved-mode experts when
each rank owns at most seven experts.
See [dispatch policy](docs/BF16_DISPATCH_READY_DEFAULTS.md).

The saved-state layout contains a reserved 1 KiB region that fixes subsequent
Gate/Up, scale and clipping-mask offsets. Its contents have no diagnostic or
readiness meaning. Algorithmic backward dispatch row-ready counters are
allocated and initialized separately.

**Known numerical limitation:** the EP2/M65/H7168/F4096/E2/TopK4 BF16 intermediate
`ptrDX` comparison can exceed its elementwise diagnostic tolerances.
Intermediate and final-gradient acceptance are separate; final-gradient
acceptance does not establish that every intermediate meets its tolerances.
Actual model loss convergence is not established by an operator
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
The precision runner applies its acceptance thresholds and reports each rank. CPU
layout/reference tests do not qualify NPU functionality. Record the exact device
cases, dependency versions and build artifacts before qualifying a deployment.

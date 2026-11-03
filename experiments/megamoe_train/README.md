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

There is one test entry with four core Python files in `tests/bf16/`:

- `test_mega_moe.py`: input generation, installed-wheel selection, distributed
  launch, fused forward/backward and timing.
- `cpu_reference.py`: independent chunked Torch CPU FP64 forward and autograd
  backward, with no intermediate BF16 rounding or router recomputation.
- `npu_baseline.py`: shared grouped GEMMs with two numerical references:
  original BigOp for CPU dual-baseline ratios, and the MegaMoE numerical
  contract for mixed tolerance, including mode-aware saved MXFP8 handling.
- `precision_metrics.py`: mixed tolerance, CPU dual-baseline ratios and reports.

Two optional additions support ATK and case coverage: `atk_compare.py` calls
the registered private CV comparator, and `cases.json` holds the reviewed
shape/routing/save-mode cases. No ATK dependency is needed for the original
four-file single-case run.

After installing the wheel, run from the repository root on idle Ascend950
cards. Defaults are EP2, M129, H256, F512, two local experts and TopK4:

```bash
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 --skip performance
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 --skip precision \
  --seq 129 --hidden 256 --gmm-n 1024 --topk 4 --expert-per-rank 2 \
  --warmup 5 --loops 20
```

The device list is physical; it becomes the visibility map and workers use
logical local ranks. Existing `torchrun` jobs are supported too. All nodes must
use identical shape, routing, seed and save-mode settings and a reachable
`--shmem-ip-port tcp://<rank-0-IP>:<unused-port>`. Gloo uses that port plus two,
separately from SHMEM and torchrun's rendezvous. Actual multi-node qualification
remains separate from providing a launch path.

Precision always checks `y`, `dx`, `dtopk_weights`, `dw1`, `dw2` and exact expert
counts. All numerical paths consume the same BF16-rounded input snapshots, fixed
nonduplicate routes, expert weights and upstream gradient. CPU FP64 truth is
recomputed every run. Only local expert parameters are replicated into each
reference rank; token contributions are reduced through Gloo.

The NPU references share BigOp's grouped GEMM/weight-gradient helpers. Their
roles are explicit:

- Mixed tolerance uses `megamoe_contract`: FP32 SiLU/Up and activation
  derivatives, with BF16 H, Hp, dH and final gradient boundaries. Forward y
  uses fixed TopK-order BF16 additions; dx uses TopK-order FP32 additions
  followed by one BF16 cast.
- CPU dual-baseline ratios retain `bigop_bf16_vs_cpu_fp64`: the original BF16
  small-op graph, autograd activation backward and NPU BF16 scatter-add.
  This denominator is independent of the contract reference.

Gloo transports route values; NPU operations perform combination. MXFP8 changes
backward operands only. The contract reference clips reconstructed values in
FP32 while derivative masks retain raw forward decisions. Save modes and
numerical thresholds are unchanged.

Both standards must pass for every rank and every floating-point result:

- Mixed tolerance uses the actual output dtype. For BF16, atol=rtol=2^-7,
  matched ratio >=99%, and max absolute error <=max(0.1,32*ULP(golden at the
  maximum-error location)). FP16/FP32 retain their corresponding dtype rules.
- CPU dual baseline uses additive relative epsilon 1e-7 and
  `fused_error/max(baseline_error,2^-8)`. Limits are MARE <=5, MERE <=1.5 and
  RMSE <=1.5. Non-finite values fail this standard; size mismatches and missing
  gradients cannot pass.

By default, precision runs first and performance runs only after all ranks
pass precision. `--skip performance` runs precision only. `--skip precision`
runs forward/backward finite/count smoke checks and synchronized NPU Event
timing without CPU/NPU references; its report marks
precision as `NOT_RUN`, and does not establish numerical qualification. Timing
reports forward and autograd-backward latency separately; reference computation,
reporting and loss construction are outside the timed regions.

Each run writes a new directory containing `rank_<rank>.json` and `summary.json`.
Use `--output <new-directory>` to choose it. Precision failure exits nonzero
(exit 2 for the direct launcher); setup/runtime failures also exit nonzero.
Torchrun may wrap a worker exit in its own launcher error. Restart the whole job
after a device/collective failure rather than attempting asymmetric cleanup.

The precision definitions are adapted from
`ascend_deepep_hkq@568ac7c07e2e669582f7ce684bf5dd27f503efad`; attribution and
source hashes are recorded separately in `upstream.json`. This is an expert-only
operator test; it does not validate router parameters or model convergence.
The contract reference passed four remote CPU checks covering activation
rounding, derivatives, MXFP8 clip masks and ordered accumulation. An isolated
NPU experiment on EP2/M129/H256/F512, save=bf16, clipping=0 matched the unchanged
native kernel bitwise for y and all four gradients after aligning the numerical
contract. The updated integrated entry and other modes still need device
validation; this is not full precision or performance qualification.

### ATK precision and case suites

Use the same entry with `--atk` to add the official
`cv_fused_double_benchmark` decision. This calls the installed ATK comparator;
it does not submit an `atk task` or use ATK's generic distributed flattening.
Every rank and each of `y`, `dx`, `dtopk_weights`, `dw1`, `dw2` are compared
separately. Actual is the fused output, benchmark is the original BF16 BigOp,
and golden is the unchanged CPU FP64 reference. Counts stay exact. Existing
mixed tolerance and CPU ratio checks still apply, so enabling ATK adds a gate.

ATK and `atk-private-precision` must be installed in the **same Python as the
DeepEP training wheel**. The personal image has ATK in `/opt/megamoe-atk`;
before device validation, install the intended DeepEP wheel into that environment
using `python -m pip install --no-index --no-deps /path/to/ascend_deepep.whl`
after `source /opt/atk-env.sh`. This preparation has not been executed by the
local adapter change. An ATK-only venv cannot run MegaMoE, and the old MegaMoE
venv does not automatically inherit packages from the ATK venv.

ATK limits are explicitly max/mean/RMSE = 5/1.5/1.5 with
`number_count_radio=2`. Use `--atk-ratios 10 2 2` only when that configuration
is intended. ATK's small-value rules and normalization remain its own; equal
ratio numbers do not make ATK and the existing CPU standard identical.
Missing plugins, a non-private provider, or comparator errors fail the run;
there is no silent fallback. Each rank JSON contains a separate `precision.atk`
section with package identity, thresholds and five decisions. Failed ATK
outputs retain actual/benchmark/golden tensors under `atk_rank_<rank>/`.

```bash
# Pure configuration inspection: no Torch/ATK imports, NPU use or installation.
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --atk --suite core --list-cases

# Future device validation: only after those physical cards are authorized/idle.
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --atk --suite smoke --skip performance --output /path/to/new-smoke-run
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --atk --suite core --skip performance --output /path/to/new-core-run
```

Suites contain 1 smoke case, 22 core cases, 39 extended cases (including core),
and 2 separate model cases. Core covers token tails, rectangular matrices,
TopK1/4/8, local experts 1/2/4/7/8, empty experts/receivers, three routing modes,
all saved formats and clipping. Extended adds 32/64/128/256/512 boundaries,
the local M16384 limit, a second seed and more clipping/recompute combinations.
Model includes M129/H2048/F1536 and M65/H7168/F4096; CPU FP64 can be expensive,
so these are explicit rather than silently skipped. `--case-id <id>` selects
one or more cases within a suite. Case dimensions/seed come from the manifest;
single-case flags cannot override them in suite mode.

EP always comes from the explicitly supplied device list. The same suites can
be scheduled at EP2/4/8; passing EP2 does not establish EP4/8 coverage. Suites
use one coordinator (`python`, not `torchrun`) and launch sequential fresh jobs,
with separate SHMEM/Gloo ports, per-case logs and reports. `suite_summary.json`
retains FAIL/ERROR/NOT_RUN and verifies all rank reports. Numerical failures
allow later cases to run; setup/device/communication errors stop the suite.
`--skip` semantics are unchanged. This ATK integration and suite coverage are
prepared locally; private-comparator and device regression remain unverified.

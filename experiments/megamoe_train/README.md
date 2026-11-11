# MegaMoE training (experimental, Ascend950)

`deep_ep_experimental.megamoe_train.MegaMoeBF16Train` fuses routing, dispatch,
GMM1, SwiGLU, GMM2 and combine, with an autograd backward for `x`, routing
weights, `w1` and `w2`. Integer expert IDs and expert counts are not differentiable.
It is designed for **Ascend950**. Other Ascend generations are unsupported.

The initial BF16 implementation derives from
[fft__/ascend_deepep_hkq](https://gitcode.com/fft__/ascend_deepep_hkq), branch
`ascend_megamoe_bf16`, commit **b8cff148**. See
[numerical semantics](docs/BF16_SWIGLU_NUMERICS.md) for the computation contract.
The API, code architecture, autograd bridge and resource lifetime are described
in the [framework design (中文)](docs/BF16_FRAMEWORK_DESIGN_ZH.md).

For a guided source walkthrough, open the [visual source guide (中文)](docs/visual/README.md):
architecture, forward/backward call chains, AIC/AIV swimlanes and buffer lifetimes.
Its interactive reader also opens directly from a local checkout.

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

`DEEPEP_MEGAMOE_TRAIN_TIMER` is the single device-timer build option and requires
`DEEPEP_BUILD_MEGAMOE_TRAIN=ON`. It accepts `ON` or `OFF` as an environment
variable in the wheel build, or as a CMake option
(`-DDEEPEP_MEGAMOE_TRAIN_TIMER=ON`). The default is `OFF`, where preprocessing
erases every probe: a release build carries no cycle reads, no extra pipe
barriers and no extra GM traffic. `ON` compiles the stage timer into both
kernels; see [Device timer](#device-timer).

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

The published [case manifest](tests/bf16/cases.json) contains only 12 typical
BF16 model-dimension cases. Other boundary, data-profile, saved-mode and historical
matrix cases are deferred and are not part of the repository manifest.

| Model dimensions | H | F / GMM1 N | TopK | Experts/rank | Clip | Route-weight scale | M/rank |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| Flash | 5120 | 2304 / 4608 | 6 | 4 | 10 | 1.5 | 512 / 2048 / 4096 / 8192 |
| Pro | 7168 | 3072 / 6144 | 6 | 4 | 10 | 2.5 | 512 / 2048 / 4096 / 8192 |
| GLM | 6144 | 2048 / 4096 | 8 | 8 | 0 | 2.5 | 512 / 2048 / 4096 / 8192 |

All cases use balanced routing, BF16 saving and seed 20260825. These are synthetic
operator tests with reduced expert counts, not full-model or convergence validation.
M means tokens entering the operator per rank; EP alone does not split M.
The upstream dy is fixed-seed random input shared by all comparison paths;
y.backward(dy) computes all four input gradients.
See the [test spec (中文)](tests/bf16/TEST_SPEC.md) for the rationale and known limits.

There is one entry, `tests/bf16/test_mega_moe.py`, with these helpers:

- `cpu_reference.py`: independent Torch CPU FP64 forward and autograd truth.
- `npu_baseline.py`: FP32-boundary NPU BigOp and the auxiliary numerical-contract reference.
- `atk_compare.py`: the private ATK CV double-benchmark comparator.
- `precision_metrics.py`: exact counts and mixed-tolerance reports.
- `mxfp8_diagnostics.py`: optional diagnosis for explicitly configured inputs;
  the published manifest contains no MXFP8 cases.

### Run the typical cases

After installing the wheel built from this branch, use the same Python containing
ATK and atk-private-precision. Select authorized, idle physical devices explicitly;
the launcher maps them to logical local ranks.

```bash
# List all 12 configurations without importing Torch/ATK or using a device.
python experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite priority --list-cases

# Precision: forward, all four gradients and counts.
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite priority --skip performance \
  --shmem-ip-port tcp://127.0.0.1:29862

# Performance only, without ATK or CPU FP64 reference computation.
python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite priority --skip precision \
  --shmem-ip-port tcp://127.0.0.1:29862
```

[tests/bf16/README.md](tests/bf16/README.md) is the runbook: case selection,
custom shapes, the three timing scopes and metric definitions, report fields,
and known issues. Notes: `--suite model` and `--suite all` select the same 12
published cases and the deferred smoke/core/extended suites are no longer
accepted; without --skip, precision runs before performance; suite shapes come
from the manifest while standalone shape arguments work only outside suite
mode; the default persistent mode reuses workers and one operator/runtime with
inputs recomputed per case (`--isolate-cases` gives fresh workers); CPU FP64
and complete gradients can require substantial host RAM; SHMEM uses the
supplied port and Gloo port+2.

### Precision decisions and reports

Precision uses ATK cv_fused_double_benchmark by default. The fused output is actual,
FP32-boundary BigOp (`fp32-boundaries-v2`) is benchmark, and unchanged CPU FP64 is golden. Every rank and
each of y, dx, dtopk_weights, dw1 and dw2 are compared separately. Explicit limits
are max/mean/RMSE=5/1.5/1.5 and number_count_radio=2. ATK and the internal plugin
must be installed in the same Python as the DeepEP wheel; missing or invalid
comparators fail the run rather than falling back to a substitute formula.

Mixed tolerance remains an auxiliary check against the MegaMoE numerical contract:
FP32 activation/derivatives, BF16 tensor boundaries, FP32 dp products/reduction,
and ordered FP32 y/dx combine with a final BF16 cast. BigOp uses the same numerical
boundaries and shared activation/derivative helpers; BF16 saved mode reuses its
result for both checks. Gate/Up are reused from forward when saved, and recomputed
for save=none. Passing the auxiliary contract
check does not override an ATK failure. Counts must match exactly.

Each run creates a new output directory; --output selects a new location.
The terminal prints progress, suite_summary.json records every case, and each
case retains summary.json plus rank_<rank>.json. Failed ATK outputs keep their
actual/benchmark/golden tensors. Numerical failures produce FAIL and a nonzero
exit; setup/device/communication/cleanup errors stop the suite and leave remaining
cases NOT_RUN. Restart after a device or collective error.
The NPU Event performance timings exclude reference computation and reporting;
case wall-clock timings are not operator performance measurements.

Performance follows the source harness `test/bigop_fused` (`test_speed.py`,
`timing.py`, `golden.py`) at `ascend_deepep_hkq@f575fc1`: three synchronized
Event scopes
(forward, autograd backward with the graph built outside the window, and
forward plus backward in one window), `--warmup`/`--loops` samples each, and one
synchronization around every window. No device Timer is built, enabled or read by these scopes; the opt-in
stage timer in [Device timer](#device-timer) is a separate diagnostic.
Only test-side code is derived: the step timer, kernel Timer decoding, router,
auxiliary loss and Megatron `MoELayer` plumbing are not adopted, and no kernel,
Host, API or build file changes with it.
The report names the two sides `megamoe_bf16` and `baseline`; `--baseline on`
(the default) also times the ported dropless small-operator chain — permute,
all-to-all dispatch, grouped GMM1, SwiGLU with routing weights, grouped GMM2,
all-to-all combine and weighted unpermute — on the same routing, weights and
upstream gradient, then prints per-scope speedups. The chain performs the same
number of GEMMs as the operator: it recomputes GMM1 exactly in the `none` save
mode and reuses the saved Gate/Up otherwise. `--baseline off` times the operator
alone.

Every scope also reports **MFU**: the RFK-model GMM FLOPs divided by the
windowed latency and by the 432 TFLOPS BF16 cube peak, for `megamoe_bf16` and
`baseline` alike. All-to-all, permute and activation stay inside the window, so
these are operator-level GMM utilizations rather than a model MFU. The
`roofline` block covers the same three scopes with the source's 410.4 TFLOPS
denominator (95% of peak), recording each scope's floor, measured time and
overhead; `floor / measured` is that scope's share of the derated roofline, about
5% higher than its MFU, and is derived rather than stored separately.

Workers only write JSON; rank 0 prints the summary once, one line per scope
with both sides, speedup and MFU, then the roofline line and one compact line
per remaining rank. `--baseline off` drops the baseline columns and reports the
skip reason instead.

A `--suite` run ends with the consolidated table the review asked for: one row
per case with the shape including the total expert count, all three latencies,
and the speedup, MFU and roofline floor triplets (forward/backward/complete),
plus the complete-scope overhead — all from rank 0 medians.

`--log` controls how much surrounds that table and defaults to `quiet`:
`quiet` prints only the case lines, the per-case `[PERF]` block and the table;
`rank0` adds the per-phase `[STAGE]`/`[TIME]` log, `[INPUT]` and the
`[PERF] stage` progress markers for rank 0, which is what locates a stall;
`all` prints the phase log for every rank. Use `rank0` when a run may hang.

The baseline is informational and never changes PASS/FAIL. It reports
`SKIPPED` with a reason when no HCCL subgroup can be created for `all_to_all`
(the coordination group is Gloo) or when the packed routes exceed the 16 GiB
working-set budget. The router, which the source times on both sides, is outside
this entry because the public operator takes routing ids and weights directly.

The earlier EP8 snapshot passed the four Flash and four Pro cases, and GLM M2048
and M8192. GLM M4096 failed the dtopk_weights maximum ATK ratio; GLM M512 was not
completed. Those are historical results for a different kernel, not qualification
of this branch's installed binary. Keeping only typical cases does not fix or
reclassify any earlier failure. No NPU test was run for this manifest reduction.

The precision references are adapted from
[fft__/ascend_deepep_hkq](https://gitcode.com/fft__/ascend_deepep_hkq), commit
`568ac7c07e2e669582f7ce684bf5dd27f503efad`.

## Device timer

The operator keeps an opt-in device timer for stage attribution. It is the training kernel's
own `GetSystemCycle()` instrumentation: every AIC/AIV records `[start, end]` cycle pairs into a
GM buffer whose layout is fixed at compile time. The performance numbers published by
`--suite model` stay NPU-Event based; the timer exists to explain a latency, not to report it.

Build and run:

```bash
export DEEPEP_MEGAMOE_TRAIN_TIMER=ON   # or -DDEEPEP_MEGAMOE_TRAIN_TIMER=ON
# ... rebuild and reinstall the wheel ...

python -u experiments/megamoe_train/tests/bf16/test_mega_moe.py 0,1 \
  --suite model --case-id pro_m2048 --skip precision \
  --timer --timer-ranks 0 --shmem-ip-port tcp://127.0.0.1:29862
```

The phase runs one extra forward and one extra backward after the pass/fail gates, dumps the
buffer between the two launches (both kernels reuse the same slots), and writes
`<output>/timer/rank_<r>_megamoe_bf16_{forward,backward}_timer.{bin,_parsed.csv}` plus a
per-stage summary in that case's `rank_<rank>.json`.

From Python, the surface is the buffer, not a flag:

```python
timer = op.allocate_timer()                    # zeroed int64, ~2 MiB
y, counts = op(x, ids, p, w1, w2, capacity, timer=timer)
y.backward(dy)                                 # writes the backward spans
op.timer_enabled                               # False when the SO has no timer
```

Notes and limits:

- The buffer is a fixed 251 904 int64 slots (96 logical cores x (10 fixed + 20 dynamic x 64
  expert slots)); forward and backward share the same slots, so dump between launches.
- Forward records routing, all-gather, per-expert dispatch, GMM1/GMM2 (split into input wait and
  compute), SwiGLU (wait and compute), Combine (wait and compute), the Combine tail wait and
  unpermute.
- Backward records gather, count-table restore, dO dispatch, GMM1 recompute, P1 GMM2-grad and the
  SwiGLU-grad epilogue, P2 dX/Combine, dW1/dW2, the AIC exit spins and the P5/P6 unpermute chain,
  plus per-expert waits (`AIV_P1_WAIT` accumulates, so `AIV_SWIGLU_GRAD_E = wait + compute`).
- Reads are `PipeBarrier<PIPE_ALL>` boundary pairs, so a span is wall time on that core, not
  pure compute; the parsers report the slowest lane per stage because cores split the tiles.
- Decoding is the training operator's own toolset, under one module with four commands:

  ```bash
  T=python\ -m\ deep_ep_experimental.megamoe_train.timer
  $T forward  --dir <output>/timer --rank-end 0                    # raw -> parsed CSV
  $T backward --dir <output>/timer --rank-end 0 --summary \
              --r 32768 --k 7168 --f 8192                          # parsed CSV + stage table
  $T trace    --timer-dir <output>/timer --rank-end 0 --phase backward  # Perfetto JSON
  $T check-schema                                                  # Python mirror vs C++ headers
  ```

  `trace` writes `rank_<r>_megamoe_bf16_<phase>_trace.json` plus a merged
  `all_rank_<phase>_trace.json` under `<output>/timer/timer_json/`; drop either on
  `ui.perfetto.dev`. One track per logical core (`group_<g>_<AIC|AIV>_<sub>`), waits tagged
  `cat=wait` and kept only for backward (they are the stalls that view is meant to expose).
  `backward --summary` prints the per-stage table (`AIC_max`/`AIV_max`/`all_max`, `%KERNEL`) and
  turns the five GEMM stages into TFLOPS against the 410.4 TF roofline when `--r/--k/--f` are given.
- The timer phases stay off unless `--timer` is passed, and they never affect PASS/FAIL.

# MegaMoE BF16 training kernels (Ascend950)

This directory contains the BF16 forward and backward kernels used by
`deep_ep_experimental.megamoe_train.MegaMoeBF16Train`. See the
[module README](../../../README.md) for the supported API, wheel build and tests.
The implementation uses CATLASS and SHMEM; it does not require CATCCOS.

## Computation and scheduling

```text
BF16 X -> routing/dispatch -> GMM1 -> SwiGLU + routing weight
       -> GMM2 -> combine/unpermute -> BF16 Y
```

The matrix products use FP32 accumulation and BF16 tensor outputs. SwiGLU uses
FP32 internal arithmetic with explicit BF16 boundaries; see the
[numerical contract](../../../docs/BF16_SWIGLU_NUMERICS.md).

The ordinary BF16 forward GMM1 uses M192, with L1 `192 x 256 x 256` and L0
`192 x 256 x 64` tiles. GMM2 and backward retain M128. Forward dispatch copies
one row at a time with two UB slots. A Ready384 block covers two M192 tiles;
partial blocks use their actual valid row counts.

The default `adaptive-auto` policy chooses N stripes from actual rows, N tiles
and available AICs. For at most seven local experts and sufficient rows and
metadata capacity, eligible nonempty experts use Ready384. Small experts wait
for the whole expert. Backward BF16/MXFP8 saved modes use the existing M128 /
256-row progress policy where eligible; `none` retains recomputation. See the
[dispatch policy](../../../docs/BF16_DISPATCH_READY_DEFAULTS.md) and
[backward protocol](../../../docs/BF16_BACKWARD_DISPATCH_READY.md).

The supported wheel fixes the source build defaults in the module CMake file.
Legacy source-repository shell build commands are not installed here. Kernel
Timer interfaces are excluded; reproducible performance checks use NPU Events.

## State and memory ownership

The Python training wrapper returns `(y, expert_token_nums)`. Native
`compute_train` also returns saved metadata, packed dispatch input and a forward
transient workspace. Each outstanding forward owns its own saved state. The
transient workspace can be released after the forward stream completes.
The public wrapper retains dispatch input required by backward.

Use `TrainingSavedStateLayout` and the transient layout helpers in
`include/mega_moe_training_workspace.h` for sizing. Persistent state contains
packed X, BF16 routing weights, routing metadata, expert counts and optional
Gate/Up storage. Transient state contains forward payload and readiness flags;
control words are initialized for each invocation. The symmetric window belongs
to the root Session and is borrowed by this module.

Receive capacity, actual routed rows and per-expert alignment are different
quantities. Size from the selected shape and actual supported capacity rules;
input/output tensors, parameters, saved tensors and runtime memory are additional
to the symmetric window. Buffer formulas alone do not establish device support
for arbitrary EP sizes, subgroups or multiple nodes.

Canonical weights are `W1 [E, 2F, H]`, `W2 [E, H, F]`; supported legacy weights
are `W1 [E, H, 2F]`, `W2 [E, F, H]`. The Host selects the matching CATLASS layout
without imposing a full transpose-and-copy on the legacy training path.

Only BF16 training is delivered here. W8A8, MGroup and UDMA-specific kernel
variants are not exposed by this module.

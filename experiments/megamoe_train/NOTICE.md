# MegaMoE training provenance

This component is derived from `fft__/ascend_deepep_hkq`, branch
`ascend_megamoe_bf16`, commit `b8cff14854d12aef77b6ae54b5890123759b4d28`:
<https://gitcode.com/fft__/ascend_deepep_hkq/tree/ascend_megamoe_bf16>.

The source repository distributes its implementation under Apache License 2.0
(copied as this component's `LICENSE`). Existing Huawei and other copyright
and license notices are retained in the files. CANN-derived routing helpers
retain their individual notices; see also `third-party/cann.LICENSE` at the
repository root. This import does not relicense those files under the root BSD
license. CATLASS and SHMEM remain external, separately licensed dependencies.

Migration changes: module paths and namespace, distinct kernel export names,
borrowed root Session workspace, pybind11 bindings, explicit Python/autograd
lifetime, optional build/package integration, test launchers and documentation.
The BF16 computational kernels, ready policy and numerical boundaries are
preserved. `upstream.json` maps imported files to their original paths and
records their original hashes. No CATCCOS runtime/template dependency is added.

Source checks verify the declared import revision and required files.
Original source hashes are provenance records; they do not require adapted
files to remain byte-identical to the initial implementation.

Follow-up cleanup adapts `17a685c` and `0c1d4a5`: remove unused MGroup
declarations and its inactive MMAD branch, and share alignment/size constants
with their original types. Layout and planning headers remain separate under
`csrc/kernels/bf16/layout/`.
The existing two-window TopK-weight gather implementation is retained.

Removed BF16 backward debug stage prints, GM phase markers and their 1 KiB
saved-state region. Dispatch-readiness flags and computational synchronization
are retained; Host and Device must be rebuilt together for the saved-state layout.

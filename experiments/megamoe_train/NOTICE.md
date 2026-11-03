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

Development-only tests and retired helpers are listed separately in
`source_tests_not_distributed`, retaining their original source hashes. The
single test entry now checks complete public forward/backward results.

The independent precision framework follows the expert-only numerical boundary
and acceptance definitions of the same source repository at
`568ac7c07e2e669582f7ce684bf5dd27f503efad`. Its CPU FP64 truth uses Torch CPU
forward and chunked autograd backward. NPU references share BigOp's grouped
GEMMs and MXFP8 saved-state round-trip. Original BF16 activation/autograd and
scatter-add remain the CPU dual-baseline comparator; a separate reference follows
MegaMoE's FP32 activation/derivatives and ordered combination for mixed tolerance.
Router/model construction is excluded. `precision_reference` records
this separate revision and reference-file hashes; it does not change the fixed
BF16 kernel source or preserved token hashes.

The optional ATK adapter and case manifest are migration test infrastructure.
They call the externally installed ATK/private precision packages; no ATK
implementation or private binary is redistributed in this source tree.

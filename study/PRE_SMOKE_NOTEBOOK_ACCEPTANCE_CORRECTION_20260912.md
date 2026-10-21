# Pre-smoke notebook acceptance-isolation correction (2026-09-12)

## Status and boundary

This record defines the source-only `v2r2p1` correction made before the first
successful v2r2 four-model development runtime smoke. The correction does not
alter the registered scientific method, dataset partitioning, prompts, model
revisions, model-loading rules, detector or feature-head artifacts, evaluation
logic, statistical estimands, or output namespace.

No registered validation or untouched-test image, annotation, label, or
outcome was read while diagnosing or preparing this correction. No scientific
experiment, protocol freeze, final table, or publication figure ran. The
previous v2r2 Jupyter launch stopped in its software-acceptance gate before the
server bound port 8888 and before a runtime smoke began.

```text
CORRECTION_SCOPE=NOTEBOOK_UNIT_TEST_SUBPROCESS_ENVIRONMENT_ISOLATION_ONLY
SCIENTIFIC_CONTRACT_CHANGED=NO
REGISTERED_SCIENTIFIC_RESULTS_WRITTEN=NO
VALIDATION_OR_TEST_DATA_READ=NO
MODEL_OR_PACKAGE_REBUILD_REQUIRED=NO
```

## Observed failure

The deployed v2r2 candidate environment and source passed their deployment
gate, but the interactive launch exported the real model-lock cache aliases and
revision pins before running the full unit-test suite. One test deliberately
constructs a temporary synthetic model lock. The production `HF_HUB_CACHE`
therefore conflicted with that temporary lock and the test correctly failed
closed. Because the launch block used `set -e`, Jupyter never started; the
subsequent SSH tunnel correctly reported connection refused.

The checked-in notebook repeated the same unit-test command with an unfiltered
copy of the live environment. A terminal-only workaround would therefore have
left the protocol-required **Restart Kernel and Run All Cells** path broken.

## Exact correction

The generated software-acceptance cell now copies the parent environment for
its child test process and removes exactly these lock-specific bindings from
that private copy:

- `HF_HUB_CACHE`
- `HUGGINGFACE_HUB_CACHE`
- `TRANSFORMERS_CACHE`
- `SEMUAV_QWEN_REVISION`
- `SEMUAV_SMOLVLM_REVISION`
- `SEMUAV_MOONDREAM_REVISION`
- `SEMUAV_UPPER_BOUND_REVISION`

The parent notebook environment is not mutated. The child retains and verifies
`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`SEMUAV_MODELS_LOCAL_ONLY=1`; `SEMUAV_LIVE_NOTEBOOK_SELFTEST=1` remains
child-only. The production cache and revision bindings are consequently still
present for the real four-model smoke and every later scientific phase.

Two behavioral tests prove both the exact seven-variable isolation and
fail-closed offline guards. The notebook was regenerated from its checked-in
builder and the full local suite passed 422/422 primary tests and 168/168
registered subtests.

## Lineage and reuse

`v2r2p1` is an immutable sibling source release that supersedes the defective
v2r2 source entrypoint. It retains the scientific run name
`semuav_visdrone_v2r2`, the candidate interpreter
`release_20260909_visdrone_inference_candidate_v2r2`, the existing v2r2
kernelspec, and the dedicated `modules-v2r2` trusted-code cache. The candidate
receipt remains truthfully bound to the base v2r2 source from which its package
inventory was built. The v2r2p1 compatibility verifier requires all runtime,
configuration, prompt, acceptance-canary, annotation, and requirement bytes to
remain identical to that authenticated base release.

The next authorized action remains the four-model development runtime smoke on
the registered DET-train canary. Scientific validation remains blocked until
that smoke passes 4/4 and the inference environment is sealed.

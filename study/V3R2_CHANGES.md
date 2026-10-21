# v3r2 changes relative to v3r1

v3r2 was written before any test access. The rationale is in
`PRE_TEST_CALIBRATION_GATE_AMENDMENT_20261013_V3R2.md`.

## Scientific source

| File | Change |
|---|---|
| `semuav/experiments.py` | `registered_wire_format` resolves `l0_cbor`/`l1_cbor`/`l2_cbor` to CBOR in Experiment 3 and in dynamic trials (receipted v3r1c). |
| `semuav/experiments.py` | `_drop_verified_episode_keys` drops the candidate copies of `visual_group_id`/`mission_profile`, after verifying them, before Experiment 3 (receipted v3r1 patch 1). |
| `semuav/experiments.py` | `build_causal_candidate_profiles(..., gold_actions, calibration_trusted)` adds class-balanced success and a 0 high-confidence-error placeholder when calibration is untrusted (pre-specified revision). |
| `semuav/policy_reducer.py` | Chunks are reordered to the INSERT column order (receipted v3r1 patch 2). |
| `semuav/control.py` | `FullPolicy(calibration_trusted=...)`: an untrusted calibration cannot trigger evidence escalation. |
| `semuav/visdrone_runner.py` | The validation runner derives `calibration_trusted` from the registered gate for the selected model and passes gold actions to the profiles. The policy specification records `calibration_trusted`, the success estimand and `controller_revision`. `best_fixed_validation` is chosen among rate-screened fixed representations (receipted v3r1c). `_controller_policies` passes `calibration_trusted` to every `FullPolicy` variant. |
| `semuav/visdrone_runner.py` | `calibration_gate_row` keeps the measured FAIL and passes the freeze only under the amendment. H4 is `NOT_ESTIMABLE` when calibration is untrusted. The embedded-H5 calibration flag now uses `calibration_claim_enabled`. |
| `semuav/registry.py` | `frame_stage` may install an exact v3r1 stage before computing, during v3r2 validation only. |
| `semuav/v3r1_import.py` | New module: deployment-pinned, hash-verified, exact-key reuse of v3r1 validation stages and receiver/feature caches. It never reuses `04_*` and refuses test-accessed sources or configuration drift. |
| `semuav/visdrone_recovery.py` | The v2r9 raw-inference recovery also accepts the v3r2 run name. |

Raw inference and measurement producers are byte-identical to v2r13; the
raw-compatibility test enforces this.

## Notebook, configuration and deployment

| File | Change |
|---|---|
| `build_visdrone_paper_notebook.py` / notebook | v3r2 identity (release, receipt, run, kernel). Validation enables the pinned v3r1 import and prints reuse audits. The freeze routes a calibration FAIL only under the amendment and binds the amendment record into the frozen protocol. |
| `configs/visdrone_paper.yaml` | `study_id`, default cache root and `pre_test_calibration_gate_amendment_record`. Nothing else changes. |
| `v3_deployment.py` | v3r2 namespaces. At deployment it inventories the quiescent v3r1 run, pins the inventory in the receipt and verifies it, and enables import for validation only. |
| `v3r2_launch_env.py`, `v3r2_preflight.py` | `v3r2_launch_env.py` exports the data, output, detector and feature-head locations recorded by v3r1. `v3r2_preflight.py` checks before Jupyter starts that every non-`paths` configuration field equals v3r1. |
| `paper_figures/` + notebook cell | Class-balanced paper figures and tables (Experiments 1–7), drawn from the run's own results at the end of validation. Read-only; a failure never blocks scientific cells. |
| `tests/test_v3r2_corrections.py`, `tests/test_policy_reducer.py`, `tests/test_v3_deployment.py` | Regression tests for every correction, the revision, the amendment and exact reuse. The reducer column-order test and the CBOR decode test both fail on v3r1. |

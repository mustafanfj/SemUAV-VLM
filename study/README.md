# SemUAV-VLM VisDrone paper study — v3r2

v3r2 is the pre-test successor of v3r1 (`release_20261013_visdrone_paper_v3r2`,
run `semuav_visdrone_v3r2`). It makes three changes:

- The four receipted v3r1 in-kernel corrections are now ordinary source.
- It registers the pre-test calibration-gate amendment and the pre-specified
  calibration-aware controller with class-balanced candidate success
  (`PRE_TEST_CALIBRATION_GATE_AMENDMENT_20261013_V3R2.md`).
- Validation reuses exact, hash-verified v3r1 computations for unchanged
  producers (`semuav/v3r1_import.py`).

The untouched test split has never been opened. `V3R2_CHANGES.md` lists
every change and `VALIDATION_REPORT.md` the checks performed. Start with
`VISDRONE_PAPER_RUN_GUIDE.md`.

The v3r1 description below is retained for history.

Mission-conditioned progressive representation selection for constrained UAV
visual uplinks. This is the source successor to the supplied 2026-09-21 v2r13
release, correcting the odd-height video failure and interleaving experiment
results with publication figures.

**Start with `VISDRONE_PAPER_RUN_GUIDE.md`.** The current notebook is
`SemUAV_VLM_VisDrone_Paper_Study.ipynb`. It requires this adjacent source tree
and the existing NYUAD dataset, frozen checkpoints and candidate environment.
It is not a standalone cloud notebook. The generic full-study notebook and
`legacy/v2r13` files are historical references, not the v3r1 launch target.

## Notebook flow

Each experiment completes its tables, exports its PNG/PDF/SVG figures,
and displays them before the next experiment. Long combined runner calls use
synchronous experiment-boundary callbacks; a callback is not a background
job. Validation results are visibly labeled development evidence. The final
read-only report repeats the same experiment-by-experiment presentation from
hash-verified artifacts after a COMPLETE test ledger.

| Experiment | Main output |
|---|---|
| 1 | Perception, validity and measured latency |
| 2 | Paired representation/receiver Pareto frontier |
| 3 | Static-link deadline-success surface |
| 4 | Controller utility, resources and representation mix |
| 5 | Mission risk and uncertainty endpoints |
| 6 | Receiver/protocol faults and recovery |
| 7 | Workstation readiness and explicit absent target evidence |

Experiment 7 is not a Jetson benchmark. H5 remains
`NOT_APPLICABLE_NO_TARGET_EVIDENCE`. H2/H3 retain their registered estimability
limits. Source-generated pictures and software fixtures are not paper results.

## Runtime and recovery

- Release: `release_20260928_visdrone_paper_v3r1`.
- Run: `semuav_visdrone_v3r1`.
- Assigned hardware: physical GPU 1, exposed as logical `cuda:0`.
- Existing environment: `release_20260912_visdrone_inference_candidate_v2r3`.
- Retained VLMs: Qwen3-VL-2B, SmolVLM2-2.2B, and the 8B upper-bound reference.
- Authenticated raw recovery: 132 v2r9 shards / 4,152 rows, verified on the lab
  machine before use. Their bytes are not contained in this package.

V3R1 creates a separate namespace. Prior releases, trained detector/feature
head, datasets, model snapshots, receipts and recovery source remain read-only.
Derived products and native receiver measurements are recomputed for v3r1.
Completed native measurement rows can resume only within an exactly matched
checkpoint context. There is no package installation or retraining step.

Use the sequence **prepare → validation → freeze → authorized test once →
report**. Only the ten documented parameter values may be edited. Restart the
kernel and run all after changing phase. A fresh v3r1 three-model smoke and
scientific-use seal are required before validation. A test OPEN ledger
consumes its attempt even if execution fails; test is never a diagnostic run.

## Review and verification

`V3_AUDIT_AND_CHANGES.md` explains fixes, compatibility and measurement limits.
`VALIDATION_REPORT.md` states the checks actually performed and the hardware
checks still needed. The manifest identifies every source payload. Historical
contract tests are explicitly routed to their archived v2r13 reference; current
science, codec, recovery, plotting and v3 deployment tests target active code.

Run local software acceptance from this directory with:

```bash
python -m unittest discover -s tests -q
```

This command does not run the scientific campaign. Rebuilding a source archive
with `build_v3_release.py` creates a new identity and requires matching launcher
checksums; do not edit a deployed sealed source tree.

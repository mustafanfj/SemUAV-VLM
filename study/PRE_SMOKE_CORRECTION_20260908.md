# SemUAV-VLM VisDrone paper study — v2r1 pre-smoke correction

Date: 2026-09-08  
Corrected release ID: `release_20260908_visdrone_paper_v2r1`  
Corrected study ID: `semuav_vlm_visdrone_paper_v2r1`  
Corrected run name: `semuav_visdrone_v2r1`  
Correction timing: before v2 prepare/smoke execution and before any registered
validation or untouched-test access  
Scientific-result status: none produced

## Purpose and lineage

The deployed v2 release passed its software and deployment gates, but an
independent pre-smoke audit found two operational/provenance defects before the
four-model smoke was run. The v2 deployment, its receipts, and its candidate
environment remain historical evidence and must not be edited or repurposed.
This correction creates a new v2r1 release lineage so that new execution cannot
be confused with the already deployed v2 identity.

No registered validation experiment, untouched-test run, model selection,
calibration, hypothesis test, paper table, or publication figure had run when
this correction was adopted. The correction does not reinterpret an empirical
result.

## Defect 1 — overly broad pre-seal reuse verification

The v2 prepare path would have verified the complete reused development bundle
before writing the scientific-use inference-environment seal. Some of those
files contain validation-derived annotations and outcomes. Although the v2
runtime smoke itself uses only one registered DET-train canary, that broader
pre-seal verification was inconsistent with the declared rule that no
validation-derived outcome bytes are read before sealing.

The v2r1 prepare path instead verifies a fixed, restricted pre-seal inventory:

- the exact model lock and local model snapshots;
- detector training plan, receipt, environment inventory, and checkpoint;
- the registered train sample and its portable artifact manifest; and
- the train-only L4 feature-head lock, weights, and encoder receipt.

The code explicitly rejects development-bundle, calibration-sample,
validation-sample, and publisher-validation paths at this gate. It records the
restricted inventory in `REUSED_V1_FROZEN_ARTIFACTS.json`, binds that immutable
receipt into the inference-environment seal, and defers complete development
bundle plus registered calibration/validation verification until the start of
the validation phase, after the seal verifies. The pre-seal receipt is never
expanded or rewritten later.

## Defect 2 — Jupyter kernelspec discovery namespace

The v2 deployment registered its scientific kernelspec under the canonical
Jupyter data directory:

```text
/data/Lane Detection Datasets/musta/semuav_vlm/.cache/jupyter/data
```

The v2 run guide instead selected a release-suffixed `data-v2` directory, which
could hide the deployed kernelspec. The v2r1 guide uses the canonical
`JUPYTER_DATA_DIR` above. Jupyter configuration and runtime state remain safely
isolated in `config-v2r1` and `runtime-v2r1`; IPython and Matplotlib state use
their own v2r1 namespaces.

## New execution identity

The correction uses distinct writable identities throughout:

| Item | v2r1 identity |
|---|---|
| Release | `release_20260908_visdrone_paper_v2r1` |
| Study | `semuav_vlm_visdrone_paper_v2r1` |
| Run/output namespace | `semuav_visdrone_v2r1` |
| Study cache | `.cache/visdrone_paper_v2r1` |
| Candidate environment | `release_20260908_visdrone_inference_candidate_v2r1` |
| Kernelspec | `semuav-vlm-20260908-v2r1` |
| Kernel display name | `Python (SemUAV-VLM 2026-09-08 v2r1)` |

The notebook requires the run name to equal `semuav_visdrone_v2r1`; prefixes or
suffixes are not accepted. The v2 candidate environment and v2 output/cache
namespaces are not valid substitutes.

## Methodological invariants

This is a provenance and execution-order correction, not a new semantic method.
The following registered v2 method contracts remain unchanged:

- semantic architecture:
  `detector_plus_bounded_vlm_verifier_plus_deterministic_host_composer_v2`;
- model-authored decision contract:
  `semuav.visdrone-proposal-decision.v1`;
- wire packet contract: `semuav.v1`;
- model-selection estimand:
  `group_equal_critical_action_plus_object_exact_v2`;
- the same four locked model snapshots and revisions;
- the same detector checkpoint, L4 feature head, prompts, 32-proposal policy,
  384-token cap, registered train canary, data partitions, sampling rules,
  endpoint definitions, hypothesis gates, and one-time test rule.

Nothing is retrained, redownloaded, relabeled, resampled, or tuned by this
correction. The v1 frozen artifacts remain read-only inputs, and the old
`release_20260830` Python environment remains forensic-only.

## Authorization and next gate

Local software tests and deployment verification authorize development smoke
only. They do not authorize registered validation or the one-time test. The
next executable gate is one production-shaped, hash-bound DET-train canary run
with exactly one PASS row for each of the four locked models. Only exact 4/4
success, an unchanged post-smoke environment fingerprint/package inventory,
and a valid `INFERENCE_ENVIRONMENT_SEAL.json` authorize the validation phase.

If any v2r1 identity, artifact hash, process-ownership check, GPU-exclusivity
gate, model row, environment receipt, or seal check fails, execution stops and
the evidence is preserved. Validation and untouched test remain closed.

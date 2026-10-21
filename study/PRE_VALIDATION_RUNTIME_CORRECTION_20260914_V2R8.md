# SemUAV-VLM VisDrone pre-validation runtime correction — v2r8

**Correction date:** 2026-09-14  
**Predecessor release:** `release_20260914_visdrone_paper_v2r7`  
**Predecessor run:** `semuav_visdrone_v2r7`  
**Corrected source release:** `release_20260914_visdrone_paper_v2r8`  
**Corrected run namespace:** `semuav_visdrone_v2r8`

## Purpose and timing

This is a narrow, prospective source/runtime correction recorded after the
v2r7 validation attempt stopped and before Experiment 1 perception evaluation,
model selection, protocol freeze, or any untouched-test access. It fixes one
missing Python import and does not change a scientific hypothesis, model
population, dataset partition, sampling rule, trace design, receiver design,
metric, threshold, selection rule, statistical procedure, or claim boundary.

The inherited
`PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md` and
`PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md` remain controlling and
byte-identical. They are historical records and are not renamed or rewritten by
this correction.

## Source-authenticated defect facts

The authenticated v2r7 `semuav/visdrone_runner.py` source:

1. defines `_warmup_request(...)` with a return annotation of
   `InferenceRequest` and constructs `InferenceRequest(...)` in its body;
2. does not import `InferenceRequest` from `semuav.perception`; and
3. calls `_warmup_request(...)` in both the development-validation perception
   runner and the one-time confirmatory-test runner.

`InferenceRequest` is defined in `semuav/perception.py`. Static inspection found
no circular-import issue. The defect is therefore a missing module-level import,
not a model, GPU, dataset, detector, cache, temporal-sampling, or scientific-
design defect.

The v2r8 correction adds exactly this import to the existing perception import
group:

```python
from .perception import (
    DetectorOnlyBackend,
    HFMultimodalBackend,
    InferenceRequest,
    ModelSpec,
    ...
)
```

It also advances `RUNNER_VERSION` from
`semuav.visdrone-paper-runner.v5` to
`semuav.visdrone-paper-runner.v6` and adds direct regression coverage for both
supported `_warmup_request(...)` proposal sources:

- proposals embedded in `detector_proposals_json`; and
- proposals loaded from the matching detector-cache row.

These facts are independently verifiable from the authenticated v2r7 and v2r8
source trees. They are distinct from the operator-observed execution boundary
below.

## Operator-observed v2r7 execution boundary

The following boundary is transcribed from the terminal/notebook output supplied
by the operator. It was not encoded in the pre-run v2r7 deployment receipt and
must not be represented as though that deployment receipt authenticated the
runtime facts.

Before the exception, the v2r7 run had:

- completed its fresh exact three-model runtime smoke with all three registered
  models passing;
- frozen the dedicated trusted-code-cache receipt and scientific-use inference
  environment seal;
- entered the validation input gate and verified the reused development bundle;
- freshly recomputed the exact eligible validation temporal capacity as seven
  MOT sequences under the registered 64-frame, stride-one rule;
- created or verified the deterministic v2r7 validation temporal sample and its
  audit artifacts;
- entered the Experiments 1–2 orchestration; and
- completed its detector prerequisite, including detector inference, proposal
  and resource measurements, the `validation_detector` stage cache and status
  record, and `validation_detector_residency.json`.

Execution then stopped in
`run_validation_perception_and_frontier(...)` while constructing the unmeasured
VLM warmup request:

```text
NameError: name 'InferenceRequest' is not defined
```

The exception occurred in `_warmup_request(...)` before the first call to
`experiment1_perception(...)`. Consequently, the v2r7 attempt did **not**:

- execute validation VLM inference;
- compare a VLM prediction with a registered truth endpoint;
- complete Experiment 1 or any of Experiments 2–7;
- calibrate or select a compact VLM;
- build a native-receiver or feature cache;
- freeze a scientific protocol;
- open the untouched-test permit or read untouched-test media, labels, or
  outcomes; or
- generate a confirmatory result, paper report, paper table, paper figure, or
  completion receipt.

The partial v2r7 temporal-selection and detector-stage artifacts remain
predecessor evidence. They are not relabeled, copied, or resumed as v2r8
scientific outputs.

## v2r8 correction and reuse boundary

v2r8 uses a new immutable release, deployment receipt, run/output namespace,
trusted-code cache namespace, Jupyter kernel identity, runtime fingerprint,
runtime-smoke input/report, and scientific-use seal. It must execute a fresh
exact 3/3 development runtime smoke because the authenticated source and runner
version changed. No v2r7 smoke row or partial validation output is reused as a
v2r8 result.

The immutable datasets, v1 frozen development bundle, detector training lineage
and checkpoint, feature-head lineage, pinned model snapshots, and unchanged
v2r3 candidate virtual environment may be reused only after their existing hash
and identity gates pass. v2r8 must freshly recompute the exact eligible
validation capacity of seven before accepting or writing its own deterministic
temporal sample. A recomputed capacity other than exactly seven remains a hard
stop.

The validation/test design remains:

| Contract element | v2r8 value |
|---|---:|
| Active VLM rows | 3 |
| Validation temporal sequences | 7 |
| Validation trace realizations | 12 |
| Validation fully paired sequence–trace cells | 84 |
| Observations per sequence | 64 |
| Source-frame stride | 1 |
| Replay rate | 5 Hz |
| Untouched-test temporal sequences | 15 |
| Untouched-test trace realizations | 30 |
| Untouched-test fully paired sequence–trace cells | 450 |

H2 and H3 remain structurally `NOT_ESTIMABLE` under the unchanged registered
minimums. H1 and H4 definitions and all test-time gates are unchanged.

## Machine-readable boundary markers

```text
V2R8_SOURCE_CORRECTION=MISSING_INFERENCEREQUEST_IMPORT_ONLY
V2R8_RUNNER_VERSION=semuav.visdrone-paper-runner.v6
V2R7_FAILURE_STAGE=VALIDATION_EXPERIMENTS_1_2_UNMEASURED_VLM_WARMUP_REQUEST
V2R7_VALIDATION_VLM_INFERENCE_EXECUTED=NO
V2R7_EXPERIMENT1_PERCEPTION_EVALUATED=NO
V2R7_MODEL_SELECTION_COMPLETED=NO
V2R7_NATIVE_RECEIVER_CACHE_BUILT=NO
V2R7_FEATURE_CACHE_BUILT=NO
V2R7_EXPERIMENTS_2_TO_7_STARTED=NO
V2R7_PROTOCOL_FROZEN=NO
V2R7_UNTOUCHED_TEST_ACCESSED=NO
V2R7_REPORT_OR_FIGURES_GENERATED=NO
V2R8_SCIENTIFIC_CONTRACT_CHANGED=NO
V2R8_PARTIAL_V2R7_SCIENTIFIC_OUTPUTS_REUSED=NO
V2R8_FRESH_EXACT_CAPACITY_RECOMPUTATION_REQUIRED=YES
V2R8_FRESH_THREE_MODEL_RUNTIME_SMOKE_REQUIRED=YES
NEXT_GATE=FRESH_V2R8_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE
```

# SemUAV-VLM VisDrone paper study v2r13 — low-evidence correction and recovery report

**Record date:** 2026-09-21 UTC  
**Release:** `release_20260921_visdrone_paper_v2r13`  
**Run:** `semuav_visdrone_v2r13`  
**Status:** `PROSPECTIVE_PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION`  
**Immediate predecessor:** `release_20260921_visdrone_paper_v2r12` / `semuav_visdrone_v2r12`  
**Raw-inference source:** `release_20260920_visdrone_paper_v2r9` / `semuav_visdrone_v2r9`  
**Scientific contract changed relative to v2r12:** **false**

## Immediate v2r12 boundary

v2r12 completed its fresh exact 3/3 runtime smoke and scientific-use seal.
The supplied validation traceback reaches the native Experiment 2 visual
receiver, which calls `DetectorOnlyBackend.infer` and then the strict semantic
packet validator. The validator correctly rejects:

```text
ValueError: Insufficient evidence requires event=insufficient_evidence and recommended_action=REVIEW
```

The source-level cause is a contradictory packet when a priority proposal
passes the registered 0.05 priority threshold but the frame has no proposal
meeting the unchanged 0.35 sufficient-evidence threshold. The detector backend
sets `sufficient_evidence=false` and `recommended_action=REVIEW` but can retain
a positive occupancy event. v2r13 changes the event to
`insufficient_evidence` whenever the existing evidence predicate is false.
It preserves thresholds, actions, evidence content, strict validation, model
population, sampling, endpoints, and measurement design.

The correction is in `DetectorOnlyBackend.infer`, so it applies to native
visual receiver calls and direct detector calls through the same policy.
Successfully validated predecessor raw Experiment 1 rows remain eligible for
exact verification and reuse. Derived summaries, calibration, selection,
semantic packets, native/feature caches, and later results are recomputed.
The v2r12 traceback does not establish completed Experiment 2 results, later
experiments, protocol freeze, untouched-test access, or final figures.

## Earlier v2r11 boundary

v2r11 completed source deployment and its unit-test gate. During prepare, its
candidate-ready verifier authenticated the deployment receipt but compared the
receipt's v2r10 acceptance-stop boundary with the older v2r9 native-receiver
schema. `StudyGateError` stopped the run before runtime smoke. Prepare-only
model-lock and empty run-directory scaffolding may exist, but no raw-shard
recovery import, scientific experiment, protocol freeze, untouched-test
access, result, or figure was produced.

v2r12 corrected the v2r11, v2r10, and v2r9 terminal-boundary routing. v2r13
preserves those distinct authenticated provenance layers and adds the v2r12
visual-receiver stop. Earlier receipts remain unmodified.

## Earlier v2r10 boundary

v2r10 completed source deployment, its full unit-test gate, a fresh exact 3/3
runtime smoke, trusted-code freeze, and scientific environment seal. During a
clean validation Run All, its software-acceptance subprocess inherited the
parent-only v2r9 Experiment-1 recovery authorization. An ordinary dry-run test
then entered the recovery path, which correctly rejected the nonofficial run
context. The acceptance gate stopped before any v2r10 validation experiment,
protocol freeze, untouched-test access, result, or figure.

v2r11 removed exactly the three recovery-control variables from the child
unit-test environment; v2r13 inherits that isolation unchanged. The
authenticated parent retains them, the recovery importer's official-run
restriction remains fail-closed, and the scientific design is unchanged.

## Earlier v2r9 boundary

The v2r9 development-validation run passed the fresh exact 3/3 runtime smoke,
froze its empty trusted-code inventory, sealed the inference environment,
verified the complete development input bundle, and recomputed the registered
seven-sequence temporal-validation capacity. It then completed all raw
Experiment 1 detector/VLM inference needed by the general and temporal routes:
132 content-addressed shards containing 4,152 inference rows.

Execution stopped on the first native Experiment 2 receiver plan:

```text
StudyGateError: Decoded semantic packet has unregistered action for
visdrone:det_train:0000030_00754_d_0000036:profile:general_mobility
```

That plan was the registered `l0_cbor` heartbeat. L0 intentionally transmits
identity, mission, time, telemetry, and provenance without a recommended
action. The v2r9 receiver incorrectly sent every semantic representation
through its action-bearing path, converted the absent action to an empty
string, and rejected it. The already registered L0 receiver outcome is
`NO_ACTION`. This is a downstream interface defect, not a failed VLM output or
a scientific result.

v2r10 advanced the native fusion implementation from
`semuav.visdrone-native-fusion.v2` to
`semuav.visdrone-native-fusion.v3`. A canonical L0 heartbeat is now strictly
validated against the frozen control context and returns `NO_ACTION`; L1 and L2
remain action-bearing and fail closed on missing or unregistered actions. L3
receiver inference is also bound to the same authoritative, pre-shared mission,
ROI, timestamp, and telemetry context exercised by the native-cache path. L3,
visual payloads, the detector, model set, prompts, thresholds, sampling,
estimands, statistics, and untouched-test policy are otherwise unchanged.
v2r13 preserves that L0 and authoritative-control-context behavior.

## Exact recovery boundary

To avoid repeating hours of outcome-blind model inference, v2r13 may import
only the exact raw v2r9 Experiment 1 CSV-gzip shards frozen by the deployment
recovery receipt:

| Scope | Shards | Rows |
|---|---:|---:|
| General calibration + validation, detector plus three VLM routes | 48 | 1,464 |
| Temporal validation, detector plus selected compact route | 84 | 2,688 |
| **Total** | **132** | **4,152** |

For every requested shard, the v2r13 gate reconstructs its v2r9 cache key from
the authenticated predecessor config, code, and model-lock identities; verifies
the exact content-addressed path, byte count, SHA-256, row count, row order,
observation IDs, model, revision, prompt, schema fields, and latency domain; and
then copies the same compressed bytes under the v2r13 cache key. The stage
ledger records the recovery-receipt and source-artifact hashes.

The following v2r9 products are **not** imported:

- Experiment 1 summaries or metrics;
- calibration results or model-selection objects;
- semantic packet caches or policy manifests;
- native-receiver or feature caches;
- Experiment 2 frontier results;
- any Experiment 3–7 result;
- a frozen protocol, untouched-test artifact, table, figure, or report.

Those products are recomputed under v2r13. If any recovery identity, hash,
coverage, or content check fails, the affected inference does not receive a
recovery hit and the run fails closed.

## Predecessor evidence

The following v2r9 identities are pinned by the v2r13 release/deployment
records:

| Artifact | SHA-256 |
|---|---|
| Source archive | `73b94ea64ac21fc61c7a3e0cec1038397c6dfc327c58d962ac96dd3052e48cf7` |
| Source manifest | `c63bddc8676d05d5ee9a3f196daa407681177fb980385c4750fb60536baf7efd` |
| Checksum inventory | `2df86df69fa36ae975f4f380b851dcd6fb28a59ff7868ff45f73f6dc806bd7dc` |
| Deployment script | `3f6b863284313418bf21097bfa0e548f19b22fef11db1175560e09f9b3b48f28` |
| Deployment receipt | `b5962f818a3ffa616e24bbf3ce188c150eeed0c0401ca101e9c483e2b13aef27` |
| Runtime-smoke input | `a3c9d33c87d2b620acd41be55d5b5494cdb75bcbe6a0abb82a5f625cce4674e1` |
| Runtime-smoke report | `7642b89cd36a87faafc62c44e88baec5bb45d3ea8d7c3438e2ad0293aab585b5` |
| Active model lock | `340dd59ca55a9be491ac806b50f9778a12d2b0b4a447485627aaf422dda05353` |
| Model-lock derivation | `947c5e1ad828eed9beaf6538f227fd871b406f3fab11bf6ce7202dabf23749b0` |
| Trusted-code receipt | `ad45cd62384697243de0391bc929b05356ba4252d80fd93854f5d1a90d4a55ed` |
| Scientific-use seal | `01f18450cac8bc5d1edf4dd45cf7302e0b2903d53609e0850cddf723bbf69ec3` |
| Run code hash | `fbf80e5ec3067ff8ca9c150c71a0ec2a3bcf5cffe9ec3663bb79c973789bdf76` |

The 132-shard inventory itself is created on the lab only after the v2r9 run is
quiescent. It is stored as
`private/visdrone_v2r10_deployment/V2R9_EXP1_RECOVERY_SOURCE.json`, bound by
the v2r10 deployment receipt, and reverified by the v2r13 deployer.

## Scientific status

This correction occurs during development validation, before protocol freeze
and before untouched-test access. The v2r9 error produced no Experiment 2
frontier, no Experiments 3–7 results, no confirmatory result, and no paper
figure. The untouched test remains closed.

v2r13 requires:

1. a fresh full unit-test gate;
2. a fresh exact 3/3 development runtime smoke;
3. a fresh trusted-code receipt and environment seal;
4. a fresh validation input/capacity gate;
5. hash-verified import of only the authorized raw shards;
6. recomputation of all derived validation products and Experiments 2–7;
7. protocol freeze and backup before the one-time untouched test; and
8. the report phase to produce final tables, figures, and manifests.

## Design consequences unchanged

Validation remains seven MOT sequences by twelve trace realizations, yielding
84 fully paired temporal-validation cells. The untouched-test design remains
15 sequences by 30 traces, yielding 450 planned temporal cells. H2 remains
structurally `NOT_ESTIMABLE` because the registered test capacity is below both
its 30-group and 500-cell minima. H3 remains structurally `NOT_ESTIMABLE`
because 15 test groups cannot meet its separate positive/negative group
minima. H1 remains confirmatory by design. H4 is confirmatory only if both
frozen denominator gates pass. H5 remains
`NOT_APPLICABLE_NO_TARGET_EVIDENCE`.

```text
V2R9_RUNTIME_SMOKE=PASS_EXACT_3_OF_3
V2R9_RAW_EXPERIMENT1_SHARDS=132
V2R9_RAW_EXPERIMENT1_ROWS=4152
V2R9_FAILURE_STAGE=FIRST_NATIVE_L0_HEARTBEAT_RECEIVER_PLAN
V2R9_NATIVE_RECEIVER_CACHE_COMPLETE=NO
V2R9_PROTOCOL_FROZEN=NO
V2R9_UNTOUCHED_TEST_ACCESSED=NO
V2R10_NATIVE_RECEIVER_CORRECTION=STRICT_L0_HEARTBEAT_VALIDATION_THEN_NO_ACTION
V2R12_RUNTIME_SMOKE=PASS_EXACT_3_OF_3
V2R12_FAILURE_STAGE=NATIVE_VISUAL_RECEIVER_LOW_EVIDENCE_PACKET
V2R13_CORRECTION=LOW_EVIDENCE_EVENT_CONSISTENCY
V2R13_RECOVERY_SCOPE=RAW_EXPERIMENT1_SHARDS_ONLY
SCIENTIFIC_VALIDATION_AUTHORIZED=NO
UNTOUCHED_TEST_AUTHORIZED=NO
NEXT_GATE=FRESH_V2R13_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE
```

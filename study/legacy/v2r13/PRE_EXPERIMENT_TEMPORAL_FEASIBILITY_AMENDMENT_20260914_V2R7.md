# SemUAV-VLM VisDrone pre-experiment temporal-feasibility amendment — v2r7

**Record date:** 2026-09-14 UTC  
**New source release:** `release_20260914_visdrone_paper_v2r7`  
**New run namespace:** `semuav_visdrone_v2r7`  
**Predecessor:** immutable v2r6 post-seal validation-input attempt  
**Development-validation bytes read before this amendment:** **yes**  
**Validation outcomes evaluated before this amendment:** **no**  
**Untouched-test data read before this amendment:** **no**  
**Experiments 1–7 started before this amendment:** **no**  
**Protocol frozen before this amendment:** **no**  
**Scientific contract changed relative to v2r6:** **yes**

## Decision

v2r7 changes only the registered temporal **validation** capacity from fifteen
to seven MOT sequences. Each selected sequence still contains exactly 64
consecutive publisher frames at source stride 1 and is replayed at 5 Hz. The
validation trace design remains twelve realizations, so the exact required
fully paired temporal-validation design changes from 180 to 84 cells:

```text
validation_sequences = 7
validation_trace_realizations = 12
validation_fully_paired_scene_trace_cells = 7 * 12 = 84
```

The untouched-test temporal design remains exactly fifteen sequences, 64
observations per sequence, source stride 1, 5 Hz, and thirty held-out trace
realizations. The static validation requirement remains fifteen independent
visual groups. No model, prompt, checkpoint, proposal rule, calibration rule,
threshold, policy, trace, estimator, endpoint, claim gate, or one-time-test
protection changes.

## Triggering evidence and exact stop boundary

The v2r6 prepare phase completed a fresh exact 3/3 runtime smoke and sealed the
scientific environment. Its controlling hashes were:

| Artifact | SHA-256 |
|---|---|
| v2r6 deployment receipt | `662c170acd2fe93b73dbac3192c647fab99e65394ecfb5eb2754fb87074bb732` |
| v2r6 three-model smoke input | `34cb2582afb0017f7dde1bacbb5604be74ac3f3c66625c7aaa892e3cd6c44c94` |
| v2r6 three-model smoke report | `aa23f74990bcb8c67988a312d3d6ae40055aa02b40ad24273bd926a7f36eedf8` |
| v2r6 reused-v1 receipt | `ec58674bf90fadca14d3225716ec7bc8c82f8a35749b7f3c54098c520e03f97f` |
| v2r6 inference environment | `b96929c7f546ab153ff6518b5e70726e9e7eeff2948c21d510059d3ebc75876c` |
| v2r6 three-model lock | `6afa89eb96ed4293575c86608410e2372430b742812216cbb3ded2c2fdfac0ce` |
| v2r6 model-lock derivation | `a013af4f5ff303fa983a15353d4747efe7992e026e8600311878ef3eed193c19` |
| v2r6 trusted-code receipt | `ffea7fbaeb2d1cb9d229ead884e9898c55ce4df5387a03ed6f4c2c5581b3e249` |
| v2r6 scientific-use seal | `5645254faabc005254ac7729d7d7cdaf1e153af8c5f49a8b291eb419f5999582` |

Those listed artifact identities are hash-authenticated. The seven-sequence
capacity and the subsequent no-sample/no-cache/no-experiment stop-flow facts
below are operator-observed terminal output; they were not persisted in a
v2r6 receipt or artifact. This v2r7 amendment records that observation without
upgrading its evidence level, and requires a fresh exact recomputation before
any temporal sample write.

The v2r6 validation-input gate verified the physical-GPU exclusivity gate, the
exact 3/3 smoke, the reused-v1 receipt, the inference environment, the
scientific seal, and the complete frozen development bundle. It then loaded
development-validation manifest bytes and attempted the registered,
outcome-blind temporal selection. Its terminal output reported exactly seven
eligible `mot_val` sequences for the locked requirement of 64 contiguous
frames at stride 1 and stopped before selecting or writing a temporal sample:

```text
Registered temporal sampling has insufficient eligible MOT sequences:
7/15 in validation; each needs 64 frames at source stride 1
```

That failure occurred before any registered temporal sample artifact was
written, before native-receiver or feature-cache construction, before model
selection or any Experiment 1–7 execution, before protocol freeze, and before
untouched-test authorization or access. Development-validation bytes were
nevertheless read; v2r6 therefore must not be described as pre-validation or
as a run in which validation data remained closed.

## Why the amendment is outcome-blind

Eligibility is determined only from split/component/sequence identity,
source-frame indices, exact mission-profile completeness, and the existence of
a complete arithmetic 64-frame window. It does not use a detector prediction,
VLM output, gold action, endpoint, calibration score, policy result, latency,
energy, or network outcome. The observed `7/15` is consequently a capacity
terminal-observed feasibility fact about the development manifest, not a favorable or
unfavorable scientific result and not a basis for tuning a model or estimator.

The amendment adopts the predecessor-terminal-observed eligible capacity of
seven under the unchanged 64-frame, stride-1, 5-Hz episode definition, pending
a fresh v2r7 exact-capacity recomputation before any sample is written. It does
not search among alternate episode lengths, strides, or selection rules after
inspecting outcomes.

## New identity and non-reuse rule

Because the registered sample count and validation paired-cell gate change,
v2r7 uses a new release, deployment receipt, run/output namespace, Jupyter
kernel, trusted-code cache, smoke report, and inference-environment seal. The
v2r6 source and run tree remain immutable evidence. No v2r6 validation artifact
or scientific row is relabeled or copied into v2r7.

The v2r6 prospective three-model amendment remains controlling: the active
VLM order is `qwen3_vl_2b`, `smolvlm2_2b`, and
`qwen3_vl_8b_upper_bound`; the first two are compact selection candidates and
the 8B model is a non-selectable upper-bound reference. Moondream remains
excluded for the previously recorded one-call exact-schema feasibility reason.

## Statistical and claim consequences

The seven-sequence temporal validation set is development evidence used for
selection, diagnostics, and protocol construction. Its lower number of
independent sequence clusters reduces validation precision and power and must
be reported. It does not enlarge any confirmatory claim.

The untouched-test temporal design remains fifteen sequences by thirty traces,
or 450 group-by-trace cells. H2 therefore remains structurally
`NOT_ESTIMABLE` because fifteen groups and 450 cells remain below the frozen
30-group and 500-cell minima. H3 remains structurally `NOT_ESTIMABLE` because
the fifteen-sequence test cohort cannot satisfy the separate 140 positive-group
and 73 negative-group minima. Their measured endpoints remain descriptive.
H1, conditional H4, H5 status, the restricted three-model population, and all
prohibited claims remain unchanged.

## Machine-readable decision summary

```text
V2R7_TEMPORAL_FEASIBILITY_AMENDMENT=POST_SEAL_PRE_EXPERIMENT_OUTCOME_BLIND
SCIENTIFIC_CONTRACT_CHANGED=true
V2R6_DEVELOPMENT_VALIDATION_BYTES_READ=YES
V2R6_VALIDATION_OUTCOMES_EVALUATED=NO
V2R6_TEMPORAL_ARTIFACT_WRITTEN=NO
V2R6_CACHE_BUILD_STARTED=NO
V2R6_EXPERIMENTS_1_TO_7_STARTED=NO
V2R6_PROTOCOL_FROZEN=NO
V2R6_UNTOUCHED_TEST_ACCESSED=NO
V2R7_VALIDATION_TEMPORAL_SEQUENCES=7
V2R7_VALIDATION_OBSERVATIONS_PER_SEQUENCE=64
V2R7_VALIDATION_TRACE_REALIZATIONS=12
V2R7_VALIDATION_FULLY_PAIRED_SCENE_TRACE_CELLS=84
V2R7_VALIDATION_STATIC_MIN_INDEPENDENT_GROUPS=15
V2R7_TEST_TEMPORAL_SEQUENCES=15
V2R7_TEST_TRACE_REALIZATIONS=30
V2R7_REQUIRED_RUNTIME_SMOKE=FRESH_EXACT_3_OF_3_PASS
SCIENTIFIC_VALIDATION_AUTHORIZED=NO
UNTOUCHED_TEST_AUTHORIZED=NO
NEXT_GATE=FRESH_V2R7_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE
```

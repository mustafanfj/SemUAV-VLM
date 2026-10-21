# SemUAV-VLM VisDrone pre-validation runtime correction — v2r9

**Correction date:** 2026-09-20  
**Predecessor release:** `release_20260914_visdrone_paper_v2r8`  
**Predecessor run:** `semuav_visdrone_v2r8`  
**Corrected source release:** `release_20260920_visdrone_paper_v2r9`  
**Corrected run namespace:** `semuav_visdrone_v2r9`

## Purpose and timing

This is a narrow, prospective source/runtime correction recorded after the
v2r8 validation attempt stopped and before Experiment 1 perception completed,
model selection, receiver-cache construction, protocol freeze, or untouched-test
access. It corrects the representation of measured-runtime metadata at the
stage-cache-key boundary. It does not change a scientific hypothesis, model
population, dataset partition, sampling rule, temporal design, trace design,
receiver design, metric, threshold, selection rule, statistical procedure, or
claim boundary.

The inherited records
`PRE_VALIDATION_RUNTIME_CORRECTION_20260914_V2R8.md`,
`PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md`, and
`PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md` remain historical,
controlling where applicable, and byte-identical. They are not renamed or
rewritten by this correction.

## Source-authenticated defect facts

In v2r8, `run_validation_perception_and_frontier(...)` copied the registered
detector measurement columns from the first pandas detector-cache row into
`backend.measurement_metadata`. Pandas materialized the integral fields
`measurement_warmup_iterations` and `measurement_repetitions` as NumPy
`int64` scalars. The same mapping was then placed in the scientific
`inference_cache_context` supplied to
`semuav.perception.run_inference_cache_sharded(...)`.

Before a shard compute callback can run, `semuav.registry.frame_stage(...)`
derives its deterministic cache key through
`hash_object(...) -> canonical_json(...) -> _plain(...)`. The v2r8 canonical
serializer deliberately supported dataclasses, mappings, sequences, paths, and
native JSON scalars, but did not reinterpret NumPy scalar types. Python's strict
JSON encoder therefore rejected the `numpy.int64` value:

```text
TypeError: Object of type int64 is not JSON serializable
```

This was a source/runtime boundary defect. It was not a model, GPU, detector,
dataset, temporal-sampling, network-trace, or scientific-design failure.

## Narrow correction

v2r9 advances `RUNNER_VERSION` from
`semuav.visdrone-paper-runner.v6` to
`semuav.visdrone-paper-runner.v7` and adds
`_normalize_measurement_metadata(...)` in `semuav.visdrone_runner`.

Immediately before both validation and confirmatory Experiment 1 cache-key
construction, the helper recursively converts only supported NumPy scalar
types to their meaning-equivalent native JSON types:

- `numpy.bool_` to `bool`;
- `numpy.integer` to `int`;
- finite NumPy floating scalars no wider than `float64` to `float`; and
- `numpy.str_` to `str`.

Mappings and lists/tuples are normalized recursively. Keys must be nonempty
strings. Non-finite floats, extended-precision floats, complex values, arrays,
sets, pandas missing sentinels, and other unsupported objects fail closed with
`StudyGateError`. A strict `json.dumps(..., allow_nan=False)` verification is
performed. The exact normalized mapping is used for both the stage cache key
and the persisted `measurement_metadata_json`; the previous permissive
`default=str` provenance encoding is removed.

The global registry serializer is intentionally unchanged. This keeps the
correction local to the measured-runtime metadata boundary and avoids changing
unrelated source, configuration, protocol, or artifact hashes by implicit type
coercion.

Regression coverage verifies:

1. exact NumPy-to-native scalar values and native Python types;
2. strict JSON round-trip and cache-hash equivalence with the corresponding
   native mapping;
3. fail-closed behavior for non-finite and unsupported values;
4. normalization before cache-key construction in both validation and
   confirmatory paths; and
5. the exact v2r8 detector-cache scenario, using pandas `numpy.int64`
   measurement fields, through
   `run_validation_perception_and_frontier(...)` and a real
   `frame_stage(...)` cache-key computation.

## Operator-observed v2r8 execution boundary

The following execution boundary is transcribed from the operator-supplied
notebook output. It is separate from the source-authenticated defect and was
not encoded as a post-run fact in the pre-run v2r8 deployment receipt.

Before the exception, v2r8 had:

- passed its fresh exact three-model runtime smoke;
- frozen the trusted-code cache and scientific-use inference seal;
- verified the post-seal development bundle;
- freshly recomputed exactly seven eligible validation MOT sequences;
- created or verified the deterministic 7-sequence, 64-observation temporal
  validation sample and 7-by-12 paired design;
- entered validation Experiments 1–2; and
- completed the detector prerequisite, including its measured detector cache
  and residency evidence.

The operator-reported v2r8 prepare evidence is bound by these exact digests:

| Evidence | SHA-256 |
|---|---|
| Deployment receipt | `bc76e680f6dcbcefb87d46c897da344d4cf03c453941e68748b034566f305531` |
| Reused-v1 receipt | `1aa3a7e147c62b01019ad332cb3634220e010f8bc9ea9362ca062dc3898128bb` |
| Inference fingerprint | `42012b2a5d01a5ca1aed5c3ef8005168918fab3f6b759f0d1e59668443d48ab3` |
| Three-model lock | `6dc3187478b753e49767661072e52f521c3ed435d4cb155df4cc60157a0bca3e` |
| Model-lock derivation | `3f1aeb2d070339b12adfe39616769ce6e5ff9aee1e37dd4ffa9f43c60c93f860` |
| Smoke input | `2494b5ffa02419a65a31982661e64c959d6f9133e874ac0215bc5b1e460ff5e3` |
| Exact 3/3 smoke report | `663b1be98236119f2fefa1bfd95d281615c66910d73ad7b29f4ea459262b2d41` |
| Trusted-code receipt | `fe903f0edfc812be9e924771757bbb5e84f03d94ee50fc1ac387de85288dc006` |
| Scientific-use seal | `5093c34c4b975f1b65c3c5a3f92bbef77f53e672c2e850e9ce2ec3a6c726d4f0` |

The first detector-only Experiment 1 shard then reached cache-key
canonicalization. `frame_stage(...)` failed before writing a `RUNNING` stage
record and before calling its compute callback.
Consequently v2r8 did **not**:

- execute validation VLM inference;
- evaluate a registered scientific endpoint;
- complete Experiment 1 or any of Experiments 2–7;
- calibrate or select a compact VLM;
- build the native-receiver or feature cache;
- freeze a protocol;
- open the untouched-test permit or read untouched-test media, labels, or
  outcomes; or
- generate a confirmatory result, paper report, paper table, paper figure, or
  completion receipt.

The v2r8 partial temporal-selection and detector-stage artifacts remain
predecessor evidence. They are not relabeled, copied, or resumed as v2r9
scientific outputs.

## v2r9 fresh-run boundary

v2r9 uses a new immutable release, deployment receipt, run/output namespace,
trusted-code cache namespace, Jupyter kernel identity, runtime fingerprint,
runtime-smoke input/report, and scientific-use seal. Because its authenticated
source and runner identity changed, it must run a fresh exact 3/3 development
runtime smoke. It must also freshly recompute the exact eligible validation
capacity of seven before accepting or writing its own deterministic temporal
sample. A capacity other than seven remains a hard stop.

The immutable datasets, v1 development bundle, detector training lineage and
checkpoint, feature-head lineage, pinned three-model snapshots, and unchanged
v2r3 candidate environment may be reused only after their existing hash and
identity gates pass. No v2r8 smoke row or partial validation artifact is a v2r9
scientific result.

The registered design remains unchanged:

| Contract element | v2r9 value |
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
V2R9_SOURCE_CORRECTION=STRICT_NUMPY_MEASUREMENT_METADATA_NORMALIZATION_ONLY
V2R9_RUNNER_VERSION=semuav.visdrone-paper-runner.v7
V2R8_FAILURE_EXCEPTION=TypeError
V2R8_FAILURE_MESSAGE=Object_of_type_int64_is_not_JSON_serializable
V2R8_FAILURE_STAGE=FIRST_EXPERIMENT1_SHARD_CACHE_KEY_BEFORE_COMPUTE
V2R8_VALIDATION_VLM_INFERENCE_EXECUTED=NO
V2R8_EXPERIMENT1_PERCEPTION_COMPLETED=NO
V2R8_MODEL_SELECTION_COMPLETED=NO
V2R8_NATIVE_RECEIVER_CACHE_BUILT=NO
V2R8_FEATURE_CACHE_BUILT=NO
V2R8_EXPERIMENTS_2_TO_7_STARTED=NO
V2R8_PROTOCOL_FROZEN=NO
V2R8_UNTOUCHED_TEST_ACCESSED=NO
V2R8_REPORT_OR_FIGURES_GENERATED=NO
V2R9_SCIENTIFIC_CONTRACT_CHANGED=NO
V2R9_PARTIAL_V2R8_SCIENTIFIC_OUTPUTS_REUSED=NO
V2R9_FRESH_EXACT_CAPACITY_RECOMPUTATION_REQUIRED=YES
V2R9_FRESH_THREE_MODEL_RUNTIME_SMOKE_REQUIRED=YES
NEXT_GATE=FRESH_V2R9_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE
```

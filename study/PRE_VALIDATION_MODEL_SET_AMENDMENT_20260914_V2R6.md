# SemUAV-VLM VisDrone pre-validation model-set amendment — v2r6

**Record date:** 2026-09-14 UTC  
**New prospective release:** `release_20260914_visdrone_paper_v2r6`  
**New run namespace:** `semuav_visdrone_v2r6`  
**Predecessor evidence:** immutable failed v2r5 development smoke  
**Validation data read before this amendment:** **no**  
**Untouched test data read before this amendment:** **no**  
**Experiments 1–7 run before this amendment:** **no**  
**Scientific contract changed relative to v2r5:** **yes**

## Decision and timing

This amendment prospectively reduces the registered VLM set from four to three
before any calibration, validation, model selection, protocol freeze, or test
access. The v2r6 VLM set is exactly:

1. `qwen3_vl_2b` — compact selection candidate;
2. `smolvlm2_2b` — compact selection candidate; and
3. `qwen3_vl_8b_upper_bound` — non-selectable upper-bound reference.

`moondream2` is excluded from the v2r6 scientific campaign. It is not silently
skipped and no v2r5 PASS row is carried forward. All three retained models must
execute again from scratch under the v2r6 identity and pass one new exact 3/3
development-smoke gate before validation can open.

## Evidence that triggered the amendment

The authenticated v2r5 one-image development smoke completed all four isolated
workers. Its rows were:

| v2r5 model | Result | Pre-validation interpretation |
|---|---|---|
| `qwen3_vl_2b` | `PASS` | One strict schema-valid, provenance-valid response. |
| `smolvlm2_2b` | `PASS` | One strict schema-valid, provenance-valid response. |
| `moondream2` | `FAIL_SCHEMA` | The worker returned normally, but the model response was not one exact JSON object. |
| `qwen3_vl_8b_upper_bound` | `PASS` | One strict schema-valid, provenance-valid response. |

The Moondream worker had return code zero, did not time out, and did not fail a
CUDA or context-budget gate. Its failure was the registered one-call exact-
schema feasibility gate: the native generation did not produce one object that
the unchanged strict parser could accept. The parser correctly performed no
substring extraction, repair, retry, fallback completion, or detector-output
substitution.

This is not an observed accuracy, mission-performance, calibration, latency,
energy, memory, or comparative-quality result. The smoke used the registered
DET-train canary solely to establish that a locked route can execute once and
satisfy the exact interface. No validation labels, validation outcomes, test
images, or test labels informed the exclusion.

## Preservation of v2r5

The v2r5 report has `all_passed=false`; consequently it created neither a
trusted-code scientific receipt nor a valid inference-environment seal. The
notebook stopped before registered validation data could be opened. It did not
freeze a protocol, authorize the one-time test, select a model, run Experiments
1–7, or create scientific results.

The complete v2r5 release, deployment receipt, run tree, smoke input, four-row
report, worker inputs/results/stdout/stderr, cache state, and failed gate are
immutable development evidence. They must not be edited, deleted, reset,
resumed, relabeled as a 3/3 result, or used as rows in v2r6. The three v2r5 PASS
rows demonstrate only that those individual routes passed that historical
v2r5 smoke; they do not authorize science and do not satisfy the new v2r6 gate.

## Prospective rationale

The original design treated runtime/interface feasibility as a conjunctive
precondition for admitting a model to scientific evaluation. Repeated
pre-validation releases corrected distinct Moondream integration issues while
keeping exact one-call, no-repair semantics. v2r5 resolved the auxiliary
tokenizer identity, context budget, and process isolation, but the native route
still failed the exact-schema feasibility criterion.

Rather than weaken the parser after observing its output, add a retry, create a
model-specific repair path, or treat a failed row as missing, v2r6 narrows the
registered model population. This is a model-set amendment, not an inference
about Moondream's accuracy and not evidence that Moondream is generally unable
to perform the task. The exclusion applies only to this frozen one-call
detector-conditioned interface and study campaign.

## Revised Experiment 1 design

The general calibration/validation portion of Experiment 1 now evaluates
exactly four end-to-end pipelines per registered sample:

| Pipeline | Role in v2r6 |
|---|---|
| `detector_only` | Registered non-VLM comparator and calibration route. |
| `qwen3_vl_2b` | Compact VLM selection candidate. |
| `smolvlm2_2b` | Compact VLM selection candidate. |
| `qwen3_vl_8b_upper_bound` | Descriptive, non-selectable upper-bound reference. |

Thus, the exact counts are three VLM pipelines plus one detector-only pipeline,
for four general Experiment 1 pipelines. Compact-model selection is restricted
to the two compact candidates. The 8B route cannot win selection, and
`moondream2` is neither evaluated nor represented by a missing row. After
selection, the registered temporal/controller Experiment 1 path continues to
measure exactly two routes: `detector_only` and the single selected compact
model.

All retained pipelines continue to use the same authenticated detector,
ordered top-32 proposal policy, 768-by-768 letterboxed image, contract-v2
proposal context, exact JSON decision parser, deterministic host composer,
generation cap, calibration estimand, and measurement boundary. No threshold,
weight, tie-break, sampling unit, hypothesis test, network condition, or test-
access rule changes in this amendment.

## New blocking smoke and seal gate

The v2r6 pre-validation gate is exact and conjunctive:

- the report contains exactly three rows in the locked order above;
- there are no duplicate, omitted, extra, inherited, or manually supplied
  rows;
- each row comes from a new verified isolated worker under the v2r6 source,
  environment, input, prompt, contract, and model revision;
- every row has `status="PASS"`, strict-schema validity, and complete provenance;
- `all_passed=true` means all three retained rows passed;
- the dedicated trusted-code inventory is exactly empty and verifies that no
  retained route uses trusted remote code; and
- the v2r6 inference-environment seal binds the exact 3/3 report and all
  required artifacts before validation access.

A 2/3 result, skipped upper bound, inherited v2r5 row, warning promoted by a
gate, timeout, malformed output, missing artifact, or hash drift keeps
validation and test closed. The new 3/3 requirement is not a waiver of the old
4/4 gate inside v2r5; it belongs only to the prospective v2r6 contract.

## Claim narrowing

Every report, table, figure, abstract, and conclusion from v2r6 must identify
the retained model population. In particular:

- comparative VLM findings cover two compact candidates and one 8B
  upper-bound reference, not the original four-model set;
- selection findings cover only Qwen3-VL-2B versus SmolVLM2-2.2B;
- no result may claim or imply Moondream accuracy, calibration, ranking,
  latency, memory, energy, robustness, mission utility, or inferiority;
- no result may generalize a retained-model effect to all compact VLMs or all
  VLM architectures;
- the upper-bound model remains descriptive and non-deployable/non-selectable;
  and
- all semantic metrics remain properties of the composed detector + proposal
  verifier + deterministic host-composer pipeline, not free-form VLM packet
  authorship.

H1–H4 keep their registered estimators and gates, but their model-related scope
is conditional on the retained feasible set. H2 and H3 retain their registered
structural `NOT_ESTIMABLE` handling. H5 remains
`NOT_APPLICABLE_NO_TARGET_EVIDENCE` without real target-hardware evidence.

## Unchanged protections

Publisher train/calibration/validation/test roles, source-truth construction,
natural grouping, detector and feature-head lineage, frozen network campaigns,
receiver semantics, resource accounting, one-time test ledger, backup gate,
and prohibited-claim ledger remain unchanged. Test-dev remains untouched until
an exact v2r6 3/3 smoke, verified environment seal, completed validation,
protocol freeze, backup verification, supervisor approval, and explicit
one-time authorization all succeed.

```text
V2R5_RUNTIME_SMOKE=FAILED_3_OF_4_PASS
V2R5_ALL_PASSED=false
V2R5_FAILED_EVIDENCE=IMMUTABLE
V2R5_VALIDATION_DATA_READ=NO
V2R5_UNTOUCHED_TEST_DATA_READ=NO
V2R5_EXPERIMENTS_1_TO_7_EXECUTED=NO
V2R6_MODEL_SET_AMENDMENT=PROSPECTIVE_PRE_VALIDATION
V2R6_REGISTERED_VLM_COUNT=3
V2R6_COMPACT_SELECTION_CANDIDATE_COUNT=2
V2R6_EXPERIMENT1_GENERAL_PIPELINE_COUNT=4
V2R6_EXPERIMENT1_TEMPORAL_PIPELINE_COUNT=2
V2R6_REQUIRED_RUNTIME_SMOKE=EXACT_3_OF_3_PASS
MOONDREAM_V2R6_STATUS=EXCLUDED_ONE_CALL_EXACT_SCHEMA_INFEASIBLE
MOONDREAM_ACCURACY_CONCLUSION=NOT_EVALUATED
SCIENTIFIC_CONTRACT_CHANGED=true
SCIENTIFIC_VALIDATION_AUTHORIZED=NO
UNTOUCHED_TEST_AUTHORIZED=NO
NEXT_GATE=FRESH_V2R6_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE
```

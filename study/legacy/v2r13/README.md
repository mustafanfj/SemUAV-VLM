# SemUAV-VLM VisDrone paper study — low-evidence packet correction v2r13

This source-only package is the v2r13 narrow detector low-evidence packet
correction for the executable SemUAV-VLM VisDrone2019 paper study.
Its sole current study entry point is
`SemUAV_VLM_VisDrone_Paper_Study.ipynb`.

The package contains protocol, implementation, acceptance tests, experiment
orchestration, analysis, and paper-export code. It contains **no v2r13 empirical
results**. A source build or unit-test pass is software acceptance, not a
runtime-smoke pass and not scientific acceptance.

Read these records before execution:

- [`PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md`](PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md)
  records the v2r12 native visual receiver stop and prospective event correction;
- [`PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md`](PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md)
  records the v2r11 pre-smoke receipt-verification stop and exact provenance
  routing correction;
- [`PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md`](PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md)
  records the v2r10 pre-experiment software-acceptance stop and the exact
  child-only recovery-environment correction;
- [`PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md`](PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md)
  records the v2r9 L0 native-receiver defect, its prospective correction, and
  the exact outcome-blind Experiment 1 shard-recovery boundary;
- [`PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md`](PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md)
  remains the controlling outcome-blind seven-sequence validation amendment
  and exact 7-by-12 paired-cell consequence;
- [`PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md`](PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md)
  remains the controlling historical record for the three-model population;
- [`PRE_VALIDATION_RUNTIME_CORRECTION_20260913_V2R5.md`](PRE_VALIDATION_RUNTIME_CORRECTION_20260913_V2R5.md)
  records the prospective v2r5 runtime correction;
- [`VISDRONE_PAPER_RUN_GUIDE.md`](VISDRONE_PAPER_RUN_GUIDE.md) defines the
  current operation; and
- [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md) records the v2r12 terminal
  boundary, exact v2r13 recovery scope, and remaining scientific gates.

Older notebooks and configurations remain only as acceptance fixtures. They
are not study entry points and must not be run.

## Current boundary: v2r12 stopped in the native Experiment 2 visual receiver

v2r12 passed its fresh three-model runtime smoke and scientific-use seal, then
entered validation Experiments 1–2. Native visual receiver execution reached
`DetectorOnlyBackend.infer`. The source reproduces the failure when a priority
detection meets the registered 0.05 priority threshold but no detection meets
the unchanged 0.35 sufficient-evidence threshold. The backend sets `sufficient_evidence=false` and
`recommended_action=REVIEW`, but retained a positive occupancy event. The
strict semantic validator rejected that contradictory packet.

v2r13 assigns `event=insufficient_evidence` whenever the existing evidence
predicate is false. It retains the registered thresholds, REVIEW action,
model population, sampling, measurements, and semantic validation rules.
The valid v2r9 raw Experiment 1 shards remain eligible for exact verified
reuse; all derived products and downstream experiments are recomputed.
No v2r12 Experiment 2 completion, later experiment, protocol freeze,
untouched-test result, or final figure is established by the supplied traceback.

## Earlier boundary: v2r11 stopped before its fresh runtime smoke

The v2r11 source deployment and full unit-test gate completed. During prepare,
the notebook authenticated the deployment receipt but incorrectly checked its
v2r10 acceptance-stop boundary against the older v2r9 native-receiver schema.
`StudyGateError` stopped execution before runtime smoke. Only prepare scaffolding
may exist; no recovery import, validation experiment, protocol freeze,
untouched-test access, result, or figure was produced.

v2r12 routed the then-immediate v2r11, inherited v2r10, and inherited v2r9 terminal
boundaries to their exact authenticated schemas and rejects hybrid objects.
That correction changed no scientific contract and left the v2r11 release
and receipt unchanged. v2r13 preserves these separate historical layers.

## Earlier boundary: v2r10 stopped in software acceptance before experiments

The v2r10 source deployment, fresh exact 3/3 model smoke, trusted-code receipt,
and scientific environment seal completed. During a clean validation Run All,
the software-acceptance subprocess inherited the parent-only v2r9 recovery
authorization. A generic dry-run fixture then correctly failed the recovery
importer's official-run restriction. The stop occurred before Experiment 1 or
Experiment 2 executed under v2r10 and before untouched-test access.

The inherited child-only correction removes exactly the three recovery-control
variables from the private unit-test subprocess while retaining them in the
authenticated scientific parent. v2r13 inherits that correction unchanged.
The already frozen recovery inventory remains byte-identical at the v2r10
private receipt path: 132 raw shards and 4,152 rows.

## Earlier boundary: v2r9 stopped at the first native Experiment 2 payload

v2r9 completed its fresh exact 3/3 model smoke and scientific-use seal. During
development validation it completed the registered detector and all three VLM
routes for the general Experiment 1 population, deterministic compact-model
selection, and the detector plus selected compact-model temporal routes.

Execution then stopped while building the native receiver cache for Experiment
2. The first registered representation is the actionless `l0_cbor` heartbeat,
but the v2 receiver incorrectly required every semantic representation to
contain `recommended_action`:

```text
StudyGateError: Decoded semantic packet has unregistered action for
visdrone:det_train:0000030_00754_d_0000036:profile:general_mobility
```

This is a deterministic receiver-interface defect, not an invalid model
response or a scientific outcome. No native cache, representation frontier,
later experiment, protocol freeze, untouched-test access, report, paper figure,
or completion receipt was produced. Preserve the complete v2r9 release,
deployment receipt, smoke/seal evidence, run tree, and traceback unchanged.

v2r10 advanced the native fusion contract from
`semuav.visdrone-native-fusion.v2` to `.v3`: an exact L0 heartbeat yields the
registered `NO_ACTION`, while L1/L2 remain strict action-bearing packets. v2r13
inherits that correction unchanged. To avoid repeating hours of identical inference,
v2r13 may import only the 132
deployment-frozen raw Experiment 1 detector/VLM shards (4,152 rows). Every
shard is key-reconstructed, hash-verified, content-validated, and copied under
the new v2r13 cache key. Summaries, calibration, selection, semantic packets,
native/feature caches, Experiments 2–7, protocol freeze, test, and reporting are
all recomputed under v2r13.

The scientific design is unchanged: validation remains 7 sequences × 12
traces = 84 fully paired cells, static validation remains at least 15 groups,
and untouched test remains 15 sequences × 30 traces = 450 cells. v2r13 still
requires its own unit-test gate, exact 3/3 smoke, and scientific-use seal before
the recovery-assisted validation run.

## Earlier boundary: v2r5 failed before validation

The authenticated v2r5 development smoke executed four isolated model workers:

| v2r5 model | Row | Meaning |
|---|---|---|
| `qwen3_vl_2b` | `PASS` | Strict output and provenance passed. |
| `smolvlm2_2b` | `PASS` | Strict output and provenance passed. |
| `moondream2` | `FAIL_SCHEMA` | The worker exited normally, but its one-call response was not one exact JSON object. |
| `qwen3_vl_8b_upper_bound` | `PASS` | Strict output and provenance passed. |

The report correctly recorded `all_passed=false`. The `moondream2` result was
an interface-feasibility failure under the frozen one-call, exact-object,
no-repair policy. It was **not** an accuracy, calibration, mission-performance,
latency, energy, memory, or ranking result. The harmless Jupyter `IProgress`
warning was unrelated to this gate.

Because the four-model v2r5 gate failed, v2r5 did not create a trusted-code
scientific receipt or inference-environment seal. No registered validation or
untouched-test data was read, the protocol was not frozen, the one-time test
was not authorized, no model was selected, and none of Experiments 1–7 ran.

The complete v2r5 release, deployment receipt, output/cache trees, smoke input,
four-row report, and worker artifacts are immutable failed-development
evidence. Do not resume, rerun, reset, delete, edit, relabel, or copy any v2r5
PASS row into v2r6 or v2r13.

## Prospective v2r6 amendment

Before observing calibration, validation, or test outcomes, v2r6 narrows the
registered VLM population to the exact feasible set below:

| Registered v2r6 model | Role |
|---|---|
| `qwen3_vl_2b` | Compact selection candidate. |
| `smolvlm2_2b` | Compact selection candidate. |
| `qwen3_vl_8b_upper_bound` | Descriptive, non-selectable upper-bound reference. |

`moondream2` is excluded from the v2r6 campaign because it did not satisfy the
pre-validation one-call exact-schema feasibility gate. This exclusion does not
establish that Moondream is inaccurate or generally incapable; it establishes
only that the locked route is outside this study's feasible model population.
The parser, no-retry rule, and no-repair rule are not weakened after seeing the
response.

This is a prospective scientific-contract change. v2r6 has a new release,
deployment, run/output, kernel, worker, smoke-report, trusted-code-cache, and
environment-seal identity. All three retained models must rerun from scratch on
the same registered DET-train canary. Only a new exact 3/3 PASS may authorize
validation.

## Bound v2r13 identity

| Item | Required value |
|---|---|
| Release ID | `release_20260921_visdrone_paper_v2r13` |
| Package directory | `SemUAV-VLM-VisDrone-Paper-Study` |
| Notebook | `SemUAV_VLM_VisDrone_Paper_Study.ipynb` |
| Configuration | `configs/visdrone_paper.yaml` |
| Run/output namespace | `semuav_visdrone_v2r13` |
| Study root | `/data/Lane Detection Datasets/musta/semuav_vlm` |
| Reused Python candidate | `$STUDY_ROOT/venvs/release_20260912_visdrone_inference_candidate_v2r3` |
| Scientific kernelspec | `semuav-vlm-20260921-v2r13` |
| Kernel display name | `Python (SemUAV-VLM 2026-09-21 v2r13)` |
| Physical/logical GPU | Physical GPU 1, exposed alone as logical `cuda:0` |
| Transformers trusted-code cache | `$STUDY_ROOT/.cache/huggingface/modules-v2r13` |
| Proposal-decision contract | `semuav.visdrone-proposal-decision.v2` |
| Generation policy | `semuav.visdrone-generation-policy.v5` |
| Paper runner | `semuav.visdrone-paper-runner.v7` |
| Native fusion policy | `semuav.visdrone-native-fusion.v3` |
| Generated-continuation cap | 256 tokens for each retained VLM |
| Required smoke | Exact 3/3 PASS, no inherited row |
| Authorized development reuse | 132 verified v2r9 raw Experiment 1 shards / 4,152 rows only |

The authenticated immutable v2r3 Python candidate is reused because
`requirements.txt` is unchanged. v2r13 must not install, upgrade, remove, or
mutate any package in that candidate. The already frozen v1 development bundle,
model-lock source evidence, detector checkpoint/training receipt, and L4
feature head are reused only after their exact hashes verify.

The v2r5 Starmie tokenizer lock and cache remain historical v2r5 evidence. They
are not active v2r13 inputs, are not copied into the v2r13 seal, and must not be
used to imply that Moondream participates in v2r13.

## Common detector-conditioned contract

The authenticated detector and each retained VLM see the same 768-by-768,
aspect-preserving letterboxed RGB image. Up to 32 detector proposals are
deterministically ordered and assigned stable zero-based IDs. Each proposal is
mapped to the actual processed image plane and quantized to 12-bit coordinates:

```text
{"v":2,"q":4095,"p":[[id,label,x1,y1,x2,y2],...]}
```

Each retained VLM must author exactly one compact JSON decision:

```text
{"e":true,"a":[...],"l":[...]}
```

The strict parser rejects malformed JSON, prose, Markdown fences, duplicate or
extra keys, invalid/repeated proposal IDs, non-increasing arrays, an `l` value
not in `a`, and nonempty arrays when `e=false`. There is no substring
extraction, response repair, retry, self-correction, fallback completion, or
detector-output substitution. After validation, deterministic host code
expands the short fields and composes the full semantic packet.

Generation policy v5 remains greedy, one-call, and fail-closed with
`max_new_tokens=256`. Qwen and Smol retain only their registered non-semantic
assistant scaffolds. The active v2r13 policy contains no Moondream route or
Moondream tokenizer dependency.

## Exact Experiment 1 population

General calibration/validation Experiment 1 evaluates exactly four pipelines
per registered sample:

1. `detector_only`;
2. `qwen3_vl_2b`;
3. `smolvlm2_2b`; and
4. `qwen3_vl_8b_upper_bound`.

That is three VLM pipelines plus the detector-only comparator. Compact-model
selection is limited to the two compact candidates. The upper-bound route is
descriptive and cannot be selected. Moondream is absent rather than represented
by a missing or failed scientific row.

After selection, temporal/controller Experiment 1 measures exactly two routes:
`detector_only` and the single selected compact model. All downstream semantic,
network, controller, and resource measurements therefore identify their source
route and remain conditional on the retained v2r13 model population.

## Per-model subprocess smoke

The development smoke launches each retained model in a new `exec`-created
Python process. Each child receives a hash-bound request, runs offline, writes
separate stdout/stderr and an atomic result, and exits. The parent authenticates
the interpreter, source, environment, model revision, prompt, input, contract,
generation policy, result, and artifact hashes.

The report is accepted only when:

- it contains exactly the three registered v2r13 model IDs in locked order;
- every row comes from a distinct verified child and has `status="PASS"`;
- strict schema, model identity, prompt/input, and generation provenance pass;
- `all_passed=true` means all three retained rows passed;
- the dedicated `modules-v2r13` trusted-code inventory is exactly empty and
  verifies that no retained route uses trusted remote code; and
- `INFERENCE_ENVIRONMENT_SEAL.json` binds the exact 3/3 evidence and has status
  `SEALED_FOR_SCIENTIFIC_VALIDATION_AND_TEST`.

A 2/3 result, skipped upper bound, inherited v2r5 row, malformed worker
artifact, timeout, CUDA error, missing receipt, or hash drift keeps validation
closed. The v2r13 3/3 requirement does not reinterpret or waive v2r5's failed
four-model gate.

## Scientific scope and claim narrowing

The primary endpoint remains the correct ground-station action delivered before
the registered deadline. `RESPOND` means immediate ground-station notification;
it does not mean emergency dispatch, accident detection, physical proximity, or
real-world intervention.

All semantic outcomes for a VLM route are properties of the composed detector +
proposal verifier + deterministic host-composer pipeline. The VLM is not
credited with free-form box localization, relation authorship, or complete
packet authorship. Detector recall remains a ceiling and must be reported.

v2r13 claims must be explicitly limited as follows:

- comparative VLM results cover two compact candidates and one 8B upper-bound
  reference;
- model selection compares only Qwen3-VL-2B and SmolVLM2-2.2B;
- no accuracy, calibration, ranking, latency, memory, energy, robustness, or
  mission-performance conclusion is permitted for Moondream;
- retained-model findings cannot be generalized to all compact VLMs, all VLM
  families, or the excluded model; and
- the 8B route remains an upper-bound reference, not a deployable candidate.

The registered hypothesis interpretations are unchanged except that every
model-comparative statement is conditional on the retained v2r13 population:

| Hypothesis | Registered interpretation under v2r13 |
|---|---|
| H1 | Estimable by design on the frozen static-link grid using group-equal averaging and a visual-group bootstrap. Null or negative results remain valid. |
| H2 | Structurally `NOT_ESTIMABLE`: the unchanged 15-sequence untouched-test cohort yields 450 group-by-trace cells, below the registered 30-group and 500-cell minima. Descriptive paired controller results remain exported. |
| H3 | Structurally `NOT_ESTIMABLE`: the unchanged 15-sequence untouched-test cohort cannot meet the separate positive- and negative-group minima. Descriptive miss/false-alert endpoints remain exported. |
| H4 | Estimable only if the frozen matched-coverage and high-confidence count gates pass; otherwise reported as auditable `NOT_ESTIMABLE`. |
| H5 | `NOT_APPLICABLE_NO_TARGET_EVIDENCE`: RTX 4090 measurements are not Jetson evidence. |

All six VisDrone components retain their registered roles: DET/MOT train for
study training/calibration, DET/MOT val for development validation, and DET/MOT
test-dev untouched until the one-time OPEN ledger. Publisher data plus frozen
mission rules define truth; neither the detector nor a VLM supplies truth.

Communication experiments remain controlled network emulations, not field
wireless measurements. GPU-board energy covers sender and ground native-
receiver compute under
`combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system`;
radio, CPU, and total-system energy remain excluded.

## Phase gates

Only the notebook's parameter assignments are changed during normal use. Save,
restart the kernel, and run all cells once for one phase.

| Phase | Required predecessor and access | Main product |
|---|---|---|
| `prepare` | Registered DET-train canary and verified train-safe provenance; validation/test closed | New three-model subprocess smoke, trusted-code inventory, environment seal |
| `validation` | Exact v2r13 3/3 PASS report, verified seal, and authenticated recovery receipt | Verify/copy 132 raw Experiment 1 shards, recompute all derived Experiment 1 products, run Experiments 2–7, produce validation evidence and protocol freeze |
| `test` | Completed validation, frozen protocol, verified backup, supervisor approval, explicit one-time authorization | One paired test-dev campaign and immutable ledger |
| `report` | Hash-verified finalized results after a `COMPLETE` ledger | Tables, figures, manifests, claim ledger, completion receipt |
| `embedded` | Final completion receipt | Verification/display of frozen H5 status only |

Validation must not begin merely because software tests pass, because three
rows passed inside the failed v2r5 four-model smoke, or because v2r6 through
v2r9 passed 3/3. Only a fresh exact v2r13 3/3 smoke plus the v2r13 scientific-use seal opens
validation. This integrity gate is designed to catch accidental or operational
source drift; it is not a defense against hostile code inserted ahead of its
own first statement. Test must never be used as a smoke check. Once the test ledger
opens, a disconnect or failure consumes the one-time attempt.

## Environment boundary

- Host and user: [redacted lab workstation].
- Jupyter binds only to `127.0.0.1` and is reached through the SSH tunnel.
- All caches, temporary files, outputs, checkpoints, Jupyter state, and model
  artifacts remain below `/data`.
- Hugging Face and Transformers are offline during model execution.
- Scientific phases prohibit package installation and automatic dependency
  changes.
- Do not run `apt`, reboot the shared workstation, expose Jupyter on `0.0.0.0`,
  or place credentials/tokens in the notebook.

## Source versus scientific acceptance

Run the final source suite from the authenticated v2r13 release with the reused
candidate Python and the clean environment specified by the run guide. The
final deployer and handoff state the exact primary-test and registered-subtest
totals. Those totals establish source-software acceptance only.

No placeholder, warning, previous PASS row, or manual inspection is permission
to weaken a gate. Follow `VISDRONE_PAPER_RUN_GUIDE.md` and stop on any mismatch,
traceback, `ERROR`, `FAIL_SCHEMA`, `FAILED`, absent artifact, or incomplete
three-model report.

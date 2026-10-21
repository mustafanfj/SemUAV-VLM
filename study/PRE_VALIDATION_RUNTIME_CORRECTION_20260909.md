# SemUAV-VLM VisDrone paper study — v2r2 pre-validation runtime correction

Date: 2026-09-09  
Corrected release ID: `release_20260909_visdrone_paper_v2r2`  
Corrected study ID: `semuav_vlm_visdrone_paper_v2r2`  
Corrected run name: `semuav_visdrone_v2r2`  
Correction timing: after the v2r1 development-only runtime smoke failed, but
before any registered validation or untouched-test access  
Scientific-result status: none produced  
Scientific contract changed: **true**

## Purpose and immutable lineage

This record adopts a bounded pre-validation correction after the first v2r1
four-model runtime smoke completed with two passing models and two failing
models. The failed smoke was a development compatibility gate on one registered
DET-train canary. It was not an experiment, model comparison, validation
measurement, hypothesis result, paper table, or publication figure.

The deployed v2r1 release, its candidate environment, deployment receipts,
runtime-smoke input receipt, raw outputs, and failed smoke report are immutable
historical evidence. They must not be edited, repaired in place, relabeled as a
v2r2 result, or used to seed PASS rows in the new release. The v2r2 correction
therefore uses a distinct release, study, run/output, cache, candidate
environment, Jupyter kernel, and private-receipt lineage.

The authoritative failed report is:

```text
/data/Lane Detection Datasets/musta/semuav_vlm/outputs/visdrone_paper/semuav_visdrone_v2r1/frozen_inputs/MODEL_RUNTIME_SMOKE.json
SHA-256: 42c6071688c63a9e369b6c6f9b623227478b9e774f0e60a741d8b402825f4a4b
```

The exact result was **2/4 PASS**:

| Locked model | v2r1 status | Interpretation |
|---|---|---|
| `smolvlm2_2b` | `PASS` | Historical development-smoke evidence only; it must run again in v2r2. |
| `qwen3_vl_8b_upper_bound` | `PASS` | Historical development-smoke evidence only; it must run again in v2r2. |
| `qwen3_vl_2b` | `FAIL_PROVENANCE` | The emitted decision was Markdown-fenced and invalid under the exact-output contract; the v2r1 classifier mislabeled that schema failure as provenance failure. |
| `moondream2` | `ERROR` | The pinned Transformers dynamic-module staging path omitted a transitive custom-code module required by the locked Moondream snapshot. |

No row from that report is reusable. A v2r2 smoke report is valid only if it is
created fresh under `semuav_visdrone_v2r2` and contains exactly one new PASS row
for every locked model.

## Scientific-access status when this correction was adopted

Before this correction:

- no registered validation image, annotation, label, or outcome had been read;
- no untouched test-dev image, annotation, label, or outcome had been read;
- no validation calibration, model selection, protocol freeze, or one-time test
  authorization had occurred;
- none of Experiments 1–7 had executed as a scientific experiment;
- no hypothesis test, scientific result, paper table, or publication figure had
  been produced; and
- no scientific-use inference-environment seal had been issued.

Accordingly, the changes below are a prospective method and runtime correction,
not a post-result adjustment. `scientific_contract_changed=true` because the
frozen prompt and the Qwen serialization policy change the executable
scientific inference method, even though the three-key decision schema, host
composer, wire packet, data partitions, endpoints, and hypothesis gates remain
unchanged.

## Finding 1 — Qwen3-VL-2B fenced and copied the prompt exemplar

The locked Qwen3-VL-2B-Instruct revision is:

```text
89644892e4d85e24eaac8bacfd4f463576704203
```

Its v2r1 row terminated normally by EOS after 30 generated tokens and did not
reach the 384-token cap. Its retained raw text had SHA-256
`a2f8150a7ee77abe11f55073e9230e8c07ee0aeccb93be02d9bea963116169eb`
and was exactly a fenced form of this decision:

```json
{"sufficient_evidence":true,"accepted_proposal_ids":[0,3],"limited_visibility_proposal_ids":[]}
```

The content inside the fence reproduced the complete concrete `[0,3]` answer
shown in `prompts/visdrone_proposal_decision_v1.txt`, whose SHA-256 is
`f6aa2991aaf253502c75b1b30a9412dfc8b65089347a4b2ac58f79b107b31010`.
This means a fence-stripping parser change alone could have made the smoke pass
while concealing exemplar copying. Such cleanup is scientifically
unacceptable and is not adopted.

The v2r1 row otherwise carried the expected registered identity, including:

- record model ID `qwen3_vl_2b`;
- record revision `89644892e4d85e24eaac8bacfd4f463576704203`;
- registered canary observation
  `visdrone:det_train:0000008_00889_d_0000039:profile:general_mobility`;
- prompt hash `1d29d69ca2e64f93da1a6cb02fc03c54e4faa30b027fd0ac795b5f66aa0eb8df`;
- decision-context hash
  `42bcf60600f335871276eb87606d1eeff2b858163279dca2a937a25096396491`;
- composer-policy hash
  `f83b8befd8b5c2bba9d5688e4eb063b6eeb49b8cfb7738213eb36172ea49eb3c`;
  and
- 32 detector proposals under decision contract
  `semuav.visdrone-proposal-decision.v1`.

The output was therefore an exact-output/schema failure, not an identity or
lineage mismatch. In v2r1, `decision_core_valid` and the parsed-packet hash were
incorrectly folded into the provenance predicate before the schema predicate.
That ordering produced `FAIL_PROVENANCE` and replaced the more informative
strict-parser error.

## Qwen correction adopted in v2r2

The v2r2 release freezes a new
`prompts/visdrone_proposal_decision_v2.txt`. It retains the same three-key
semantic decision contract and ID/visibility rules but removes the complete
JSON answer exemplar and every concrete proposal-ID answer. The release
manifest binds the exact final prompt bytes.

Both Qwen models, not only the failing compact model, use one common frozen
serialization policy:

- the host supplies exactly one assistant prefill character: `{`;
- the pinned Transformers 4.57.1 processor receives that prefill as
  `content=[{"type":"text","text":"{"}]`, and the frozen policy binds this
  container as `assistant_content_shape=typed_text_block_list`;
- the native chat template is invoked with `add_generation_prompt=False` and
  `continue_final_message=True`;
- only newly sampled continuation tokens are decoded;
- the exact parser receives the host prefill concatenated with that unmodified
  continuation;
- generation remains greedy with `max_new_tokens=384`;
- token-cap and EOS decisions inspect the generated continuation, not the
  host-supplied prefix; and
- output-token count and confidence cover generated continuation tokens only.

The audit record distinguishes the host-supplied prefill from model-generated
bytes and binds the generation-policy identity, input identity, continuation,
and assembled candidate text. A second opening brace, Markdown fence, prose,
trailing value, duplicate/missing/extra key, invalid proposal ID, unfinished
object, token-cap termination, or non-EOS completion still fails closed.

There is no fence stripping, JSON substring extraction, schema repair, retry,
self-correction, fallback completion, or substitution of detector output after
a model failure. The SmolVLM2 path retains its native generation-prompt mode;
the Moondream path retains its native query interface.

The smoke classifier is also corrected without weakening acceptance:

- wrong model, revision, observation, prompt, request, context, generation
  policy, or composer identity is `FAIL_PROVENANCE`;
- correct identity with malformed or schema-invalid output is `FAIL_SCHEMA`,
  retaining the original parser/schema error; and
- `PASS` still requires strict decision validity, exact composer-policy
  identity, a valid composed packet, EOS termination, no token-cap hit, and all
  bound hashes.

The same Qwen policy and prompt must be consumed by the development smoke and
the later scientific validation/test backend. A hidden model-specific notebook
override is prohibited.

## Finding 2 — Moondream local-directory custom-code closure was incomplete

The locked Moondream repository and revision are:

```text
vikhyatk/moondream2
6b714b26eea5cbd9f31e4edb2541c170afa935ba
```

The exact locked snapshot is complete. In particular, its root `rope.py` is
present with SHA-256
`0032ace75443a81888b957b28b7fd00ac808effdfdace14855d31b5c14ea5770`.
The v2r1 loader nevertheless passed the local snapshot directory to
Transformers 4.57.1 with trusted remote code. The resulting dynamic-module
cache package staged direct custom-code imports but did not materialize the
complete transitive closure required by `moondream.py`; its root `rope.py` was
absent. Import then failed with `FileNotFoundError` from the derived
Transformers module cache.

This was not a missing model-lock file, corrupt model weight, network failure,
or scientific model result. It was a compatibility failure in the chosen
local-directory custom-code loading route.

## Moondream correction adopted in v2r2

After exact `MODEL_LOCK.json` verification, v2r2 loads Moondream by its official
repository ID `vikhyatk/moondream2` plus the exact locked revision above, while
enforcing `local_files_only=True`, `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, `SEMUAV_MODELS_LOCAL_ONLY=1`, and the exact locked
Hugging Face cache root. This selects the repository-aware Transformers path so
the pinned custom-code dependency closure is staged from already-present local
bytes. It does not permit a network fallback.

One shared registered-model load-reference helper is used by both the runtime
smoke factory and the later validation/test factory. Qwen and SmolVLM2 continue
to load from their exact locked snapshot directories; only Moondream uses the
official repository reference with the exact revision and offline cache.

The derived Transformers custom-code cache is isolated for v2r2 at:

```text
/data/Lane Detection Datasets/musta/semuav_vlm/.cache/huggingface/modules-v2r2
```

The v2r2 launcher must set that exact `HF_MODULES_CACHE` before importing
Transformers. The first v2r2 smoke must not rely on the stale v2r1 dynamic-module
cache. Manual copying of `rope.py`, patching the locked model snapshot, deleting
historical caches, loosening the model lock, or enabling online downloads is
prohibited.

## New v2r2 execution identity

| Item | v2r2 identity |
|---|---|
| Release | `release_20260909_visdrone_paper_v2r2` |
| Study | `semuav_vlm_visdrone_paper_v2r2` |
| Run/output namespace | `semuav_visdrone_v2r2` |
| Study cache | `.cache/visdrone_paper_v2r2` |
| Candidate environment | `release_20260909_visdrone_inference_candidate_v2r2` |
| Candidate build records | `visdrone_inference_candidate_build_v2r2` |
| Kernelspec | `semuav-vlm-20260909-v2r2` |
| Kernel display name | `Python (SemUAV-VLM 2026-09-09 v2r2)` |
| Jupyter configuration/runtime namespaces | `config-v2r2` and `runtime-v2r2` |
| Dynamic custom-code cache | `modules-v2r2` |

The canonical Jupyter data directory remains unsuffixed so the installed kernel
is discoverable. The v2r1 release, candidate environment, kernel, output root,
and caches remain historical and must not be repurposed.

## Methodological invariants retained

The following registered scientific design elements do not change:

- four model repositories and exact revisions in the verified model lock;
- detector checkpoint, detector proposal ordering, threshold, and top-32 cap;
- common image preprocessing and model-native processors;
- decision schema `semuav.visdrone-proposal-decision.v1`;
- deterministic host composer and transmitted packet schema `semuav.v1`;
- semantic architecture interpretation as a detector-conditioned VLM proposal
  verifier plus deterministic packet composer;
- immutable v1 detector, L4 feature-head, and registered data artifacts;
- publisher truth, data partitions, sampling rules, registered estimands,
  deadlines, network traces, endpoints, hypothesis gates, and statistical
  procedures; and
- the one-time untouched-test authorization rule.

No detector or feature head is retrained, no model is replaced or redownloaded,
no dataset item is relabeled or resampled, and no validation/test outcome is
used to choose this correction.

## Mandatory v2r2 acceptance sequence

Local tests and deployment checks authorize development smoke only. Before any
validation access, v2r2 must satisfy every one of these gates:

1. Seal and deploy the exact v2r2 source, configuration, prompt, tests,
   candidate builder, and operator guide under the distinct identities above.
2. Build a fresh sibling v2r2 inference candidate and verify its exact ready
   receipt, package inventory, runtime identity, offline guards, and immutable
   v2r1 predecessor lineage.
3. Launch the v2r2 notebook from the v2r2 candidate kernel with the release-
   scoped cache paths established before model/CUDA imports.
4. Reverify the restricted train-safe v1 artifact inventory and the exact
   production-shaped registered DET-train canary input.
5. Run a **fresh all-four-model smoke**. SmolVLM2 and Qwen3-VL-8B must rerun;
   their v2r1 PASS rows cannot be copied or accepted.
6. Require exactly one v2r2 `PASS` row for each of `qwen3_vl_2b`,
   `smolvlm2_2b`, `moondream2`, and `qwen3_vl_8b_upper_bound`, with exact model,
   revision, prompt, generation-policy, request, context, composer, raw-output,
   and composed-packet identities.
7. Reverify the inference runtime and installed-package inventory after all
   four models have executed. Any import-time mutation or unexpected cache/
   environment identity blocks promotion.
8. Only after exact 4/4 success, atomically write and independently verify
   `INFERENCE_ENVIRONMENT_SEAL.json` with status
   `SEALED_FOR_SCIENTIFIC_VALIDATION_AND_TEST`.

Until all eight gates pass, the status remains:

```text
READY_FOR_DEVELOPMENT_SMOKE_ONLY
SCIENTIFIC_VALIDATION_AUTHORIZED=NO
UNTOUCHED_TEST_AUTHORIZED=NO
```

Validation may begin only from the exact sealed v2r2 environment and release.
The untouched test remains closed until validation, calibration, selection,
protocol freeze, evidence backup, and explicit one-time authorization are
complete. Any later change to the prompt, generation policy, model load route,
cache policy, dependency inventory, or source bytes requires another distinct
release identity and another fresh four-model smoke.

## Relationship to earlier records

`METHODOLOGY_AMENDMENT_20260908.md` and
`PRE_SMOKE_CORRECTION_20260908.md` remain immutable historical records. This
document supersedes only their v2r1 runtime-specific statements that the prompt
contains the concrete decision exemplar, Qwen uses no assistant prefill,
Moondream loads through a local-directory trusted-code route, and v2r1 is the
next executable identity. It does not silently rewrite those records or alter
their account of what v2r1 actually specified and executed.

The v2r1 2/4 smoke must be reported only as pre-validation engineering evidence
that motivated this correction. It must never be presented as model accuracy,
comparative performance, an ablation, or a scientific result.

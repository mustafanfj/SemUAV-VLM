# Pre-validation runtime correction: v2r3 to v2r4

Date: 2026-09-12  
Status: prospective development-only correction; scientific validation and test remain blocked

## Trigger

The authenticated v2r3 deployment passed **429/429** primary tests and
**181/181** registered subtests. Its first four-model development smoke ran only
on the registered VisDrone training canary. Qwen3-VL-2B completed, but
SmolVLM2-2.2B produced an asynchronous CUDA indexing failure after v2r3 routed
it through Qwen's typed assistant-continuation template. The CUDA context was
then poisoned and the cleanup call surfaced the device assertion again. The
smoke remained incomplete: it produced no complete authoritative four-row
report and no scientific-use environment seal. No validation or test image,
label, experiment, statistic, or scientific result was accessed or produced.
The failed v2r3 state is historical development evidence and is not reused as a
passing smoke.

## Isolating evidence

A fresh-process, Smol-only diagnostic used the same locked model revision, the
same registered training canary, the same frozen prompt and proposal context,
and strict parsing. It changed only the host transport prefix while preserving
SmolVLM2's native user-only multimodal generation rendering.

| Route | Observed result |
|---|---|
| Native render plus `{` | Model completed normally but returned non-JSON key syntax; strict result `FAIL_SCHEMA` |
| Native render plus `{"` | Model completed normally and returned one exact JSON object; strict result `PASS` |

For the passing route, `token_id_max=49279` was below
`embedding_rows=49280`, 1,377 image tokens and 17 image blocks were present,
generation ended by EOS, and no CUDA indexing failure occurred. This diagnostic
is compatibility evidence only and is not a scientific observation.

## Prospective v2r4 correction

The generation policy advances from
`semuav.visdrone-generation-policy.v3` to
`semuav.visdrone-generation-policy.v4`, with SHA-256
`7444721316b04aa9ae1511916e2511c945d501ec1389bb3c5682958e8a5fb624`.

- Qwen3-VL-2B and Qwen3-VL-8B retain
  `qwen_open_brace_assistant_continuation`: a typed assistant `{` is continued
  with `continue_final_message=True`.
- SmolVLM2-2.2B uses `smol_native_double_quote_assistant_prefix`: its native
  user-only template is rendered with
  `add_generation_prompt=True` and `tokenize=False`; the exact two-character
  host scaffold `{"` is appended to the rendered prompt before the same
  processor tokenizes it with the same letterboxed image.
- Moondream remains on its unchanged native query route.
- The model still authors every decision value. The strict exact-JSON parser,
  frozen prompt, proposal contract, deterministic host composer, model
  revisions, canary, token cap, no-retry rule, and no-repair rule are unchanged.
- Smoke provenance records and verifies the exact strategy, prefix length,
  prefix SHA-256, and generated-continuation SHA-256 independently for each
  model.
- Runtime-smoke cleanup now preserves the primary inference error and writes one
  explicit failing row per locked model if an unrecoverable CUDA cleanup error
  prevents later models from running.

This prospectively changes SmolVLM2's pre-generation rendering and prefix. It
is therefore a generation-method and scientific-contract change, not a
behaviorally neutral runtime adjustment:
`scientific_contract_changed=true`.

The v2r4 source-only release reuses the authenticated immutable v2r3 Python
candidate because `requirements.txt` is byte-identical. It uses distinct v2r4
source, run/output namespace, trusted-code cache, deployment receipt, and
Jupyter kernel identities. No package installation, candidate mutation, or
candidate rebuild is required.

The v2r4 source-only package has passed its current local acceptance suite:
**431/431** primary tests and **180/180** registered subtests. This is local
source evidence only; it does not constitute a v2r4 deployment or runtime-smoke
pass.

## Authorization boundary

v2r4 is not authorized for validation merely because this source correction
exists. The next required gate is one fresh production-shaped development smoke
with exactly one `PASS` row for each of the four locked models. Only after the
4/4 report, its bound input/provenance, the runtime inventory, and the new
trusted-code cache are sealed may the validation phase execute Experiments 1–7.
The untouched test partition remains closed until the later frozen-protocol,
one-time authorization step.

## Machine-readable release facts

```text
V2R3_RUNTIME_SMOKE=FAILED_INCOMPLETE
V2R3_DEPLOYED_SOFTWARE_ACCEPTANCE=PASS_429_OF_429
V2R3_DEPLOYED_REGISTERED_SUBTESTS=PASS_181_OF_181
V2R3_COMPLETE_REPORT_WRITTEN=NO
V2R3_SCIENTIFIC_USE_ENVIRONMENT_SEAL_WRITTEN=NO
VALIDATION_OR_TEST_DATA_READ=NO
EXPERIMENTS_1_TO_7_EXECUTED=NO
V2R4_CORRECTION=PROSPECTIVE_PRE_VALIDATION
V2R4_SOURCE_ONLY_RELEASE=YES
V2R4_RELEASE_ID=release_20260912_visdrone_paper_v2r4
V2R4_RUN_OUTPUT_NAMESPACE=semuav_visdrone_v2r4
V2R4_CANDIDATE_ENVIRONMENT=release_20260912_visdrone_inference_candidate_v2r3
V2R4_KERNELSPEC=semuav-vlm-20260912-v2r4
V2R4_TRUSTED_CODE_CACHE=modules-v2r4
V2R4_LOCAL_SOFTWARE_ACCEPTANCE=PASS_431_OF_431
V2R4_LOCAL_REGISTERED_SUBTESTS=PASS_180_OF_180
V2R4_REMOTE_DEPLOYMENT=PENDING
V2R4_RUNTIME_SMOKE=PENDING
GENERATION_POLICY_VERSION=semuav.visdrone-generation-policy.v4
GENERATION_POLICY_SHA256=7444721316b04aa9ae1511916e2511c945d501ec1389bb3c5682958e8a5fb624
SCIENTIFIC_CONTRACT_CHANGED=true
NEXT_GATE=FRESH_V2R4_FOUR_MODEL_DEVELOPMENT_RUNTIME_SMOKE
SCIENTIFIC_VALIDATION_AUTHORIZED=NO
```

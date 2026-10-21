# Pre-validation runtime correction: v2r2p1 to v2r3

Date: 2026-09-12  
Status: prospective correction adopted before any registered validation or test

## Boundary

The v2r2p1 production-shaped development smoke used one registered DET-train
canary and the four locked VLM revisions. It was a runtime-acceptance gate, not
a scientific experiment. No registered validation or test image, annotation,
label, or outcome was read. None of Experiments 1–7 ran; no calibration, model
selection, protocol freeze, hypothesis result, paper table, or figure exists.

The v2r2p1 smoke report is immutable failed-development evidence. Its rows may
not be deleted, edited, reclassified, retried in place, or copied into a later
smoke report.

## Confirmed v2r2p1 result

| Model | Status | Confirmed observation |
|---|---|---|
| `qwen3_vl_2b` | `PASS` | Exact three-key JSON and bound provenance passed. |
| `smolvlm2_2b` | `FAIL_SCHEMA` | The model returned one object-like continuation with unquoted keys; the strict JSON parser correctly rejected it. |
| `moondream2` | `ERROR` | Its trusted model code requested the secondary tokenizer identity `moondream/starmie-v1`, which was absent as an independent Hub-cache entry even though the required tokenizer bytes were already present in the locked Moondream snapshot. |
| `qwen3_vl_8b_upper_bound` | `PASS` | Exact three-key JSON and bound provenance passed. |

The exact SmolVLM2 text was:

```text
 {sufficient_evidence: true, accepted_proposal_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]}
```

Its SHA-256 was
`1d221d5c932013c0f1a116dfada17ecd1f136fb2d6714d93bc6ada60bc4ac01f`.
It terminated normally after 133 generated tokens and did not reach the token
cap. This is a schema failure, not truncation and not a model-loading failure.
The parser did not repair the unquoted keys.

The locked Moondream revision is
`6b714b26eea5cbd9f31e4edb2541c170afa935ba`. Its locked snapshot already
contains `tokenizer.json` with SHA-256
`337da36be7a71a6e88aa9148967a7bc8736f4b47c7de8e19ba92b89e80734cfc`.
Static inspection confirmed the fixed secondary reference
`moondream/starmie-v1`; an offline cache lookup for that separate identity
returned no entry. The error therefore reflects local artifact routing, not a
scientific model outcome.

## Prospective v2r3 correction

The v2r3 policy is `semuav.visdrone-generation-policy.v3`.

For `qwen3_vl_2b`, `smolvlm2_2b`, and `qwen3_vl_8b_upper_bound`, the policy uses
the existing strategy identifier
`qwen_open_brace_assistant_continuation`. The identifier is retained for
compatibility, but its behavior is generic: the host supplies exactly one typed
assistant-continuation character, `{`, before generation; the native processor
continues that message; and only newly generated continuation tokens are
decoded. The host brace contains no proposal ID or decision value. It is
prepended once and recorded independently from the model continuation.

This prospective change gives SmolVLM2 the same bounded JSON-opening scaffold
that already produced exact output for both Qwen models. It does not quote keys
after generation, extract a substring, strip Markdown, repair JSON, retry,
self-correct, or substitute detector output. The unchanged exact-object parser
still fails closed on any malformed result.

Moondream2 retains its `native_query` strategy. Its exact locked snapshot,
revision, processor, and model weights are unchanged. After verifying the
snapshot inventory and recorded tokenizer hash, the v2r3 backend receives the
locked `tokenizer.json` path and hash. During trusted local-only model
construction, it temporarily replaces only the staged `moondream.py` module's
`Tokenizer` binding so the fixed
`Tokenizer.from_pretrained("moondream/starmie-v1")` call resolves through the
real `Tokenizer.from_file` loader against that verified file. The original
module binding is restored in `finally`. The recorded provenance route is
`locked_snapshot_tokenizer_json_temporary_module_injection`. Network fallback
remains disabled; no snapshot file is edited, no tokenizer is downloaded, no
independent Starmie cache tree is accepted, and no unlocked artifact is
introduced.

## New indivisible execution identity

| Item | v2r3 identity |
|---|---|
| Release | `release_20260912_visdrone_paper_v2r3` |
| Run | `semuav_visdrone_v2r3` |
| Candidate environment | `release_20260912_visdrone_inference_candidate_v2r3` |
| Kernelspec | `semuav-vlm-20260912-v2r3` |
| Trusted-code cache | `modules-v2r3` |

The v2r3 change is prospective and occurs before validation, so
`scientific_contract_changed=true`. The frozen source partitions, detector
proposals, model repositories and revisions, decision schema, packet composer,
representation definitions, network traces, estimands, endpoints, hypothesis
gates, and one-time-test rule remain unchanged.

## Required next gate

v2r3 must run one fresh production-shaped development smoke with exactly one
new row for each of the four locked models. Historical PASS rows are not
transferable. Scientific validation remains blocked until all of the following
are true:

1. all four v2r3 rows have `status=PASS`;
2. request, model/revision, prompt, policy, continuation, raw response,
   decision-context, composer, packet, EOS, and token-cap identities verify;
3. the dedicated `modules-v2r3` cache is inventoried and frozen; and
4. `INFERENCE_ENVIRONMENT_SEAL.json` verifies with status
   `SEALED_FOR_SCIENTIFIC_VALIDATION_AND_TEST`.

Only then may the separate v2r3 validation run execute Experiments 1–7.

```text
V2R2P1_RUNTIME_SMOKE=FAILED_2_OF_4
V2R2P1_REPORT_IMMUTABLE=YES
VALIDATION_OR_TEST_DATA_READ=NO
EXPERIMENTS_1_TO_7_EXECUTED=NO
V2R3_CORRECTION=PROSPECTIVE_PRE_VALIDATION
NEXT_GATE=FRESH_V2R3_FOUR_MODEL_DEVELOPMENT_RUNTIME_SMOKE
SCIENTIFIC_VALIDATION_AUTHORIZED=NO
```

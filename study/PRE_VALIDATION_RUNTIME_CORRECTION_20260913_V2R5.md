# Prospective pre-validation runtime correction — v2r5

**Adopted:** 2026-09-13 UTC  
**Stage:** development runtime acceptance, before validation, test, protocol
freeze, model selection, or Experiments 1–7  
**Supersedes:** `release_20260912_visdrone_paper_v2r4`  
**New release:** `release_20260913_visdrone_paper_v2r5`  
**New run:** `semuav_visdrone_v2r5`  
**Scientific contract changed relative to v2r4:** **true**

## Decision

Release v2r4 is a completed, failed development-smoke attempt and is retained
unchanged as evidence. It is not resumed, repaired in place, or used as a source
of PASS rows. Release v2r5 prospectively changes the bounded VLM transport and
runtime-smoke execution method, then reruns all four registered models from
scratch on the same registered DET-train canary. Validation and test remain
locked until the new report contains exactly four independently executed PASS
rows and the v2r5 inference-environment seal verifies.

## Observed v2r4 boundary

The authenticated v2r4 source deployment passed 431/431 primary software tests
and 180/180 registered subtests. Its one-image development runtime smoke wrote a
complete four-row report with `all_passed=false`:

| Model | v2r4 status | Development observation |
|---|---|---|
| `qwen3_vl_2b` | `PASS` | Strict proposal-decision schema and provenance passed. |
| `smolvlm2_2b` | `PASS` | Strict proposal-decision schema and provenance passed. |
| `moondream2` | `ERROR` | CUDA device-side indexing assertion during native query. |
| `qwen3_vl_8b_upper_bound` | `ERROR` / not run | The shared process was no longer recoverable after the Moondream CUDA failure. |

The frozen v2r4 evidence includes:

- `MODEL_RUNTIME_SMOKE_INPUT.json` SHA-256
  `eaaca41b4557c0f25d99780d5627746c55c128ab3f0fbab46571bdd9adba33f9`;
- `MODEL_RUNTIME_SMOKE.json` SHA-256
  `7f7b5ca2b57dfa202cf983486429d7c539c45bf7d81d306d56af605c7a0ed3d5`;
- `INFERENCE_ENVIRONMENT_FINGERPRINT.json` SHA-256
  `99a38604a1e429753bf2dc4d024526251541b59417e6e36d226d2ecf63a428f7`;
- `INFERENCE_PIP_FREEZE.txt` SHA-256
  `5b6aacbcd2b891c1bf968511d17b3385aff2233e99d0bae440f81bc7f977d5d2`;
- `REUSED_V1_FROZEN_ARTIFACTS.json` SHA-256
  `1756f05df59bda3843e6514247f5aedcdcaa51046e278c6997d74745577b5ff6`.

`TRUSTED_CODE_CACHE.json` and `INFERENCE_ENVIRONMENT_SEAL.json` were not
created. No validation or test data was read, and no scientific experiment ran.

## Exact failure diagnosis

The failed Moondream revision has a fixed text context of 2,048 positions and
places the encoded image prefix at position 730. The v2r4 canary supplied 32
full-precision proposal rows and reserved 384 generated tokens. A read-only CPU
probe using the exact v2r4 query construction measured:

| Serialization | Query tokens | Prefill positions | Prefill overflow | Prefill + 384 | Reserved overflow |
|---|---:|---:|---:|---:|---:|
| Full precision | 1,930 | 2,664 | 616 | 3,048 | 1,000 |
| Six decimals | 1,367 | 2,101 | 53 | 2,485 | 437 |
| Five decimals | 1,311 | 2,045 | 0 | 2,429 | 381 |
| Four decimals | 1,297 | 2,031 | 0 | 2,415 | 367 |

The tokenizer vocabulary size was 50,295 and the largest emitted token ID was
50,286, so a token-ID/vocabulary overflow is ruled out. The device assertion was
therefore caused by position indexing beyond the fixed context, not by an
out-of-range lexical token. Decimal rounding alone is insufficient because it
does not preserve the registered generation reserve.

The diagnosis also found that v2r4 intercepted the remote-code request for
`moondream/starmie-v1` but substituted the different `tokenizer.json` stored in
the locked `vikhyatk/moondream2` snapshot. That route was deterministic, but it
did not reproduce the secondary tokenizer dependency requested by the locked
Moondream source. The substituted file had 2,114,924 bytes, vocabulary size
50,295, and SHA-256
`337da36be7a71a6e88aa9148967a7bc8736f4b47c7de8e19ba92b89e80734cfc`;
that digest is historical v2r4 failure evidence and must not satisfy the v2r5
Starmie lock. v2r5 corrects the dependency explicitly.

## Prospective v2r5 method

### Common compact decision transport

All four VLMs receive the same new compact context, rather than applying a
Moondream-only truncation. Proposal boxes are mapped deterministically onto the
actual 768-by-768 letterboxed image shown to the models and quantized to the
closed 12-bit coordinate interval 0–4095. Context rows remain ordered and have
the shape `[proposal_id,label,x1,y1,x2,y2]`. The compact context keys are
`v` (contract version), `q` (coordinate maximum), and `p` (proposal rows).

The model-authored wire decision uses exactly three short keys:
`e` (sufficient evidence), `a` (accepted proposal IDs), and `l` (limited-
visibility proposal IDs). The strict parser rejects duplicate keys, prose,
fences, extra members, invalid IDs, non-increasing arrays, and any `l` value not
contained in `a`. Only after strict validation does deterministic host code
expand the short keys to the existing canonical internal names for composition
and scoring. No semantic value is supplied or repaired by the host.

The common generated-continuation cap is reduced prospectively from 384 to 256
tokens. The longest legal compact decision with both arrays containing all 32
IDs is far below that cap. Greedy decoding, zero retries, exact-object parsing,
and fail-closed behavior remain unchanged. These changes are registered as
`semuav.visdrone-proposal-decision.v2` and
`semuav.visdrone-generation-policy.v5`.

### Exact auxiliary tokenizer dependency

The tokenizer consumed by the locked Moondream source is pinned separately as
repository `moondream/starmie-v1`, file `tokenizer.json`, immutable artifact
revision `360559108ac4586b96328de81f817859a72fd5ca`. The source archive does not
redistribute the tokenizer. A bounded acquisition step stores it only in the
dedicated `starmie-v2r5` cache and freezes its resolved commit, byte count, and
SHA-256 in `MOONDREAM_TOKENIZER_LOCK.json`, whose required schema is
`semuav.moondream-starmie-tokenizer-lock.v1` and lock type is
`huggingface_single_file_auxiliary_dependency`. Runtime is local-only and
accepts only the exact locked bytes. The lock and its parent chain must be
non-symlinked. The tokenizer snapshot entry may use Hugging Face's canonical
same-repository symlink only when its resolved regular-file target remains
inside `models--moondream--starmie-v1/blobs/`. The unchanged Moondream
remote-code constructor is still narrowly intercepted, but now receives the
tokenizer it requested rather than the model-repository tokenizer.

### Pre-CUDA context guard

Moondream encodes the image first, tokenizes the exact assembled query with the
locked Starmie tokenizer, includes the native prefix and suffix positions, and
verifies that prefill plus the complete 256-token generation reserve fits the
model's fixed context. Failure raises a normal study gate before text-model CUDA
indexing. The observed image position, query-token count, context limit,
reserved generation count, and remaining margin are recorded in inference
provenance.

### Fresh process per smoke model

Each real smoke model runs in a new `exec`-created Python process. The parent
process verifies the common input and lock, writes one hash-bound worker
request, captures stdout/stderr and an atomic result, validates child
attestation, and then advances to the next model regardless of an earlier
child's exit status. Kernel teardown destroys each child's CUDA context, so a
device failure cannot suppress a later registered model. The final schema-v3
report still contains exactly one row for each locked model and fails closed if
any worker result, identity, log, hash, or provenance field is missing or
different.

## Identity and immutability rules

v2r5 uses a new source release, run/output tree, deployment receipt, Jupyter
kernel, worker-artifact directory, trusted-code cache `modules-v2r5`, and
auxiliary tokenizer cache `starmie-v2r5`. The authenticated v2r3 Python
candidate is reused byte-for-byte because `requirements.txt` is unchanged.
No v2r4 output, cache, notebook execution, receipt, or smoke row is copied into
v2r5. The v2r4 tree must not be reset, deleted, or rerun.

## Next gate

`NEXT_GATE=FRESH_V2R5_FOUR_MODEL_DEVELOPMENT_RUNTIME_SMOKE`

The gate passes only when the v2r5 report has exactly the registered four model
IDs, every row has `status=PASS`, `all_passed=true`, every worker attestation and
token-budget field verifies, the dedicated trusted-code cache is inventoried,
and the inference-environment seal has status
`SEALED_FOR_SCIENTIFIC_VALIDATION_AND_TEST`. Until then, validation and test
remain prohibited.

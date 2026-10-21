# SemUAV-VLM VisDrone pre-validation acceptance/recovery isolation — v2r11

**Recorded:** 2026-09-21  
**Predecessor:** `release_20260921_visdrone_paper_v2r10` / `semuav_visdrone_v2r10`  
**Corrected release:** `release_20260921_visdrone_paper_v2r11`  
**Corrected run:** `semuav_visdrone_v2r11`

## Observed boundary

The v2r10 source deployment, unit-test gate, three-model development smoke,
environment seal, and deployment-frozen v2r9 Experiment-1 recovery inventory
were valid.  On a clean validation Run All, the notebook's software-acceptance
subprocess failed before any validation experiment executed.  The single
failure was
`test_detector_revision_and_vlm_cache_bind_checkpoint_residency`.

The test correctly constructed a non-scientific fixture run context.  However,
the validation parent had already enabled authenticated v2r9 Experiment-1
recovery, and the acceptance subprocess inherited that validation-only switch.
The recovery importer then correctly rejected the fixture because recovery is
restricted to the official full scientific run.

## Prospective correction

v2r11 removes exactly these three validation-only variables from the private
software-acceptance subprocess environment:

- `SEMUAV_ENABLE_V2R9_EXP1_RECOVERY`;
- `SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT`; and
- `SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256`.

The scientific parent environment remains unchanged.  Therefore the official
v2r11 validation cells still receive the deployment-authenticated recovery
receipt and may import only the same 132 frozen raw v2r9 Experiment-1 shards
containing 4,152 rows.  Generic unit-test fixtures receive no recovery
capability.  The recovery importer's full-run restriction is not weakened.

Regression coverage verifies the exact three-variable child isolation, the
unchanged eight model-lock-specific child removals, exact offline guards, and
the byte-for-byte preservation of the parent environment.

## Scientific scope

This is a pre-experiment process-isolation correction.  It changes no dataset,
sample, split, model, checkpoint, prompt, detector output, inference row,
payload, receiver policy, threshold, estimator, hypothesis, or untouched-test
rule.  No v2r10 validation experiment executed after the acceptance failure,
and untouched test access remained closed.  All derived Experiment-1 products,
model selection, semantic packets, receiver/feature caches, and Experiments
2–7 remain scheduled for fresh computation under v2r11 after the authenticated
raw-shard import.

No operator may bypass the acceptance gate, edit the sealed notebook outside
the ten approved parameter right-hand sides, or manually set recovery
variables.  Any missing, changed, or incompatible recovery artifact remains a
hard stop.

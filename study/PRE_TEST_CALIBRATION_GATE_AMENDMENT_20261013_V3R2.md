# Pre-test amendment: calibration gate, controller revision and v3r1 corrections (v3r2)

**Date:** 2026-10-13
**Predecessor:** `release_20260928_visdrone_paper_v3r1` / `semuav_visdrone_v3r1`
**Successor:** `release_20261013_visdrone_paper_v3r2` / `semuav_visdrone_v3r2`
**Test status at the time of writing:** the untouched test split (DET/MOT
test-dev) has never been opened. No protocol has been frozen. No test ledger
exists for any v3 run.

This amendment is written and sealed in source before any test access. It
requires supervisor sign-off before the v3r2 protocol freeze.

## 1. What the v3r1 validation run showed

The registered calibration gate requires, for the detector and the selected
model, a validation ECE whose visual-group bootstrap upper bound is at most
0.05 (0.10 for priority rows), with temperature scaling fitted on the
calibration split. On v3r1 validation it failed for all four models.
Every temperature fit reached its upper bound (T = e^3 = 20.09), and ECE was
0.34–0.46. The cause is structural: the models report a mean confidence of
0.76–0.99 while their action accuracy on the calibration split is 9–41%.
Temperature scaling of a binary correctness probability can only move it
toward 0.5, so it cannot reach these accuracies. No implementation defect was
found in the gate. Its FAIL result stands and is reported.

Under v3r1, a failed gate blocks the protocol freeze, and therefore the test.
The v3r1 controller also consumed the uncalibrated signal in two ways. Model
uncertainty triggered evidence escalation. A missing high-confidence-error
estimate defaulted to the maximal placeholder 1.0, which both forced
escalation and zeroed the success term. On validation this made the
controller behave exactly like always-VLM.

## 2. Amendment

1. **Calibration gate scope.** A measured calibration FAIL keeps its FAIL
   record in every gate table and frozen artifact. It no longer blocks the
   protocol freeze when the frozen controller cannot consume model
   confidence. "Cannot consume" means the frozen policy specification records
   `calibration_trusted = false` under controller revision
   `v3r2-calibration-aware-escalation-and-balanced-profiles`, and
   `calibration_claim_enabled` is false. In that case:
   - H4 (confidence-gated evidence escalation) is `NOT_ESTIMABLE`, with the
     reason recorded in the hypothesis table.
   - No calibration or confidence claim is eligible for the manuscript.
   - If calibration passes, nothing changes from v3r1.
   - Calibration also remains part of the registered VLM eligibility rule.
     With every model failing calibration, the selected perception pipeline is
     `detector_only` and no VLM claim is enabled, exactly as registered.
2. **Controller revision (pre-specified).** When `calibration_trusted` is
   false:
   - Uncertainty and the high-confidence-error estimate cannot trigger
     evidence escalation (`FullPolicy.select`).
   - The confidence-derived high-confidence-error estimate is 0 for every
     candidate (v3r1 used the maximal placeholder 1 when support was
     missing), so the uncalibrated signal can neither force escalation nor
     discount the success term.
   - The registered VLM-context trigger is unchanged. It schedules perception
     work from detector risk/uncertainty and mission priority; it does not
     choose the transmitted representation.

   In every case, candidate success (`predicted_task_success`) is the
   class-balanced receiver recall on development samples: the mean over gold
   actions of the per-class recall. This replaces raw accuracy, because gold
   actions are 75% RESPOND and raw accuracy rewards guessing the majority
   class. The same estimand selects each fixed comparator.
3. **Primary reported endpoints.** These are class-balanced and are computed
   from the frozen test outputs by the released analysis scripts:
   - receiver recall balanced over gold actions, with a visual-group cluster
     bootstrap;
   - the cohort-balanced correct-and-timely rate (the mean of the
     critical-episode and non-critical-episode rates), with a two-way
     sequence × trace cluster bootstrap.

   Raw accuracy and raw rates are reported as secondary. The registered H1–H3
   tests keep their registered endpoints and are reported as registered.

## 3. Receipted v3r1 corrections moved into source

The first three corrections were applied in the v3r1 kernel with receipts.
The fourth, the controller revision above, was prepared but not run. All four
are now ordinary source:

| # | Correction | Effect on numbers |
|---|---|---|
| 1 | Experiment 3 drops the candidate copies of `visual_group_id`/`mission_profile` after verifying they equal the manifest values (merge-suffix `KeyError`). | None |
| 2 | The policy reducer orders `usecols` chunks to match its INSERT column order (`ValueError int('fixed_l0_heartbeat')`). | None |
| 3 | `l0_cbor`/`l1_cbor`/`l2_cbor` resolve to the CBOR wire format. In v3r1 every L0–L2 message failed with `decode_error`; the affected Experiment 3 shards and all Experiment 4 trials were quarantined and recomputed. `best_fixed_validation` is now chosen only among rate-screen-passing fixed representations (the v3r1 rule chose an L4 variant that sent 0 bytes). | Corrects L0–L2 outcomes and the best-fixed comparator |
| 4 | Calibration-aware controller with class-balanced profiles (Section 2). | Changes Experiment 4 controller behaviour |

## 4. Reuse of unchanged v3r1 validation computations

v3r2 reuses v3r1 validation computations only where the producer code is
unchanged and the inputs are byte-identical:

- perception inference
- detector measurement
- native receiver and feature caches
- the Experiment 2 frontier
- Experiment 3 static transport (after correction 3)

A stage is reused only on an exact cache-key match. If, for example, the
Experiment 3 inputs do not reproduce byte-identically, those shards are
recomputed, which is slower but equally correct. Experiment 3 shards that
still show the v3r1 L0–L2 decode defect are never reused.

Reuse is checked in four steps. At deployment, a read-only inventory of the
quiescent v3r1 run is written, and its digest is pinned in the v3r2
deployment receipt. At run time, a stage is reused only when the v3r2 key
material reproduces the v3r1 cache key exactly. Every reused file is re-hashed
against the inventory. Native receiver and feature caches are then
re-validated against the v3r2 manifest, semantic cache and design. Experiment
4 partitions are never reused.

The following are recomputed under v3r2:

- calibration and model selection
- the candidate profiles and policy specification
- Experiments 4–7, the claim screen and every gate

## 5. What does not change

The following are unchanged:

- models and prompts
- sample registration and the test design (15 sequences × 30 traces)
- the link grid and seeds
- thresholds, margins and the H1–H3 definitions
- the trace design and the one-time test ledger

Raw inference and measurement producers stay byte-identical to v2r13; the
existing raw-compatibility test enforces this.

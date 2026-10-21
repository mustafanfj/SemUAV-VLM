# SemUAV-VLM VisDrone pre-validation native-receiver correction and recovery — v2r10

**Recorded:** 2026-09-21  
**Predecessor:** `release_20260920_visdrone_paper_v2r9` / `semuav_visdrone_v2r9`  
**Corrected release:** `release_20260921_visdrone_paper_v2r10`  
**Corrected run:** `semuav_visdrone_v2r10`

## Observed boundary

The v2r9 development-validation run completed detector inference, all three
registered VLM routes, deterministic validation selection, and the selected
compact temporal route. It then stopped on the first native Experiment 2
payload, before a native cache, representation frontier, protocol freeze, or
untouched-test access completed. The exception was:

`Decoded semantic packet has unregistered action for visdrone:det_train:0000030_00754_d_0000036:profile:general_mobility`

## Root cause

The first native payload is the registered `l0_cbor` heartbeat. The frozen L0
wire projection intentionally contains identity, mission, time, telemetry, and
provenance but no `recommended_action`. The production native receiver treated
every semantic variant as action-bearing, converted the absent field to an
empty string, and rejected it. The registered Experiment 2 contract already
defines the L0 receiver result as `NO_ACTION`. This was a downstream interface
defect, not an invalid VLM response or an outcome-dependent design decision.

## Prospective correction

The native fusion policy advances from
`semuav.visdrone-native-fusion.v2` to
`semuav.visdrone-native-fusion.v3` and registers variant-specific behavior:

- a strictly validated, actionless `l0_cbor` heartbeat yields `NO_ACTION`;
- `l1_cbor` and `l2_cbor` still require an exact registered transmitted action;
- L3 still applies the registered severity fusion of transmitted semantics and
  the detector result on transmitted ROI pixels;
- malformed or unknown variants remain fail-closed.

An end-to-end regression of the corrected native path also exposed a second
pre-validation interface defect: visual and ROI receiver requests had retained
the mission profile and registered ROI, but supplied a synthetic zero capture
time and omitted the authoritative latitude, longitude, altitude, and yaw that
the semantic contract requires. The receiver-control schema therefore advances
to `semuav.visdrone-receiver-control.v2`. Each hashed, gold-free control record
now binds the manifest's outcome-independent capture time and authoritative
flight telemetry in addition to mission profile and ROI. Image and video
receiver requests copy those bound values exactly. L3 preserves the bound
capture time and flight telemetry and replaces only the ROI field in a fresh
request mapping with the registered crop-coordinate receiver ROI
`[0.0, 0.0, 1.0, 1.0]`; the frozen control record is never mutated. No zero or
other placeholder telemetry is invented. Context construction fails closed on
missing, malformed, or disagreeing mission-profile/ROI telemetry.

No payload, prompt, model, snapshot, detector, threshold, calibration method,
sampling rule, endpoint, hypothesis, or untouched-test rule changes.

## Outcome-blind recovery boundary

Only raw v2r9 Experiment 1 detector/VLM shard artifacts may be carried forward.
The recovery gate reconstructs the predecessor cache key for every requested
shard, verifies the quiescent predecessor run identity and a deployment-frozen
artifact inventory, rehashes the exact gzip bytes, and verifies row order,
observation IDs, model/revision, prompt, and schema fields before registering a
byte-identical copy under the v2r10 cache key.

The v2r9 derived summaries, calibration, selected model, policy manifests,
semantic packets, native/feature caches, Experiment 2 results, and every later
experiment are not imported. They are recomputed under v2r10. Imported rows are
explicitly marked in the stage ledger as verified v2r9 development-inference
reuse and remain bound to the recovery-receipt digest.

The expected recovery set is 132 raw shards / 4,152 rows:

- general calibration+validation: 366 rows for detector-only and each of the
  three registered VLMs (48 shards total);
- temporal validation: 1,344 rows for detector-only and the deterministically
  selected compact model (84 shards total).

The recovery verifier does not rely solely on `status.sqlite`, because general
and temporal stages in v2r9 shared stage names and later rows replaced earlier
ledger entries while content-addressed artifacts remained present.

Operators must not copy cache files or set recovery environment variables by
hand. The v2r10 deployer freezes the source inventory and the notebook enables
that authenticated receipt only during the validation phase. A missing,
changed, incompatible, or incomplete shard stops recovery fail-closed.

## Scientific status

This amendment occurs during development validation. The untouched test remains
closed. v2r10 requires a fresh unit-test gate, fresh three-model smoke, and a
fresh environment seal before validation. If any predecessor identity, shard,
hash, coverage, or compatibility check fails, the run stops. Do not bypass the
gate or substitute an unregistered cache.

Successful recovery does not itself produce a paper result. v2r10 must still
recompute the derived Experiment 1 products, execute Experiments 2–7, freeze
and back up the validated protocol, complete the separately authorized
untouched test once, and run the report phase to export final tables and
figures.

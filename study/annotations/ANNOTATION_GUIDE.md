# Annotation overlay contract

Create one row per `sample_id × task × annotator_id`. Each confirmatory item must
have at least two independent annotators. `blinded` must be `true`: annotators
must not see model predictions, payloads, policies, network state, or method
names. For every disagreement, repeat the same adjudicated label and adjudicator
ID on the item's rows after adjudication.

Required task names:

- `gold_action`: `NO_ACTION`, `REVIEW`, or `RESPOND`, derived from visible facts
  and the already assigned mission profile.
- `critical`: Boolean. Mark true only under the frozen visually supported
  critical-event definition.
- `cohort`: `natural_test`, `safety`, `ood`, or `development`.
- `sufficient_evidence`: Boolean.
- `event`: one supported ontology value.
- `objects`: canonical JSON list of objects shaped as
  `{"label": "<frozen object label>"}`; optional normalized geometry may be
  added, but bare strings and detector-specific class IDs are invalid.
- `relations`: canonical JSON list shaped as
  `{"subject": "<present frozen object label>", "predicate": "<frozen relation>", "object": "<present frozen object label>"}`.
  Bare strings and detector-specific track IDs are invalid annotation forms.
- `evidence_roi`: normalized canonical JSON box `[x1,y1,x2,y2]` linked to the
  supported claim, or `[]` only when `sufficient_evidence=false`.

Recommended tasks include visible severity, evidence ROI/geometry, and each
mission-specific field used in a receiver decision. Put boxes/polygons in
`evidence_geometry_json` using a declared coordinate convention. Report median
evidence localization IoU, with a target of at least 0.70.

Do not annotate intent, identity, assistance need, legal fault, accident cause,
or fire cause from imagery. Use `sufficient_evidence=false` and a `REVIEW` gold
action when the frozen mission rules require human/evidence escalation.

Agreement gates are at least 0.80 overall chance-corrected agreement and 0.70
for each rare confirmatory class. Revise the ontology and reannotate before test
unblinding if a gate fails.

## Embedded log schema

`embedded_log_template.jsonl` is an illustrative schema only. Replace every
example row with event-level measurements from the registered target run. Each
`trial_id × policy` must contain exactly these stages, once each and in this
order: `capture`, `detector_done`, `representation_ready`, `enqueue`,
`tx_done`, `receive`, `decode_done`, `decision`. Stage indices must be 0--7;
timestamps must be strictly increasing inside the chain; sender/receiver roles
must match the template; and all timestamps used for end-to-end latency must be
in one named, synchronized `clock_domain_id`. The evaluator derives latency as
`decision.monotonic_ns - capture.monotonic_ns`. A supplied `latency_ms` is
retained only as an audit field and cannot affect a gate.

Record compute and radio energy as nonnegative **stage deltas**, not repeated
cumulative readings. `trial_total_energy_j` must equal the sum of all
`stage_compute_energy_j + stage_radio_energy_j` values within the chain.
Similarly, uplink `wire_bytes_delta` is charged only on `tx_done`, and its sum
must equal `trial_total_wire_bytes`. A transmitted trial must have positive
measured radio energy. Any missing stage, clock reversal, changing trial-level
field, invalid role, negative measurement, or inconsistent total is a hard
integrity failure. A complete log also has one real `hardware_id`, the frozen
`platform_class`, controlled trace/link identifiers, memory and temperature on
every stage, and immutable sender/receiver revisions. Example values are never
experimental observations.

For H5, also create a registered target-hardware H2--H4 retest summary from the
raw paired target trials. Begin the event artifact with
`embedded_raw_trial_template.csv` and the reconciliation document with
`embedded_retest_summary_template.csv`. The raw artifact must include the exact
frozen policy family on identical sample/mission/trace/seed tuples and bind each
row to the target-log SHA-256, protocol hash, hardware identity, code revisions,
and analysis-plan version. It must meet the frozen minima: 500 independent
natural-test visual groups, 200 safety-critical positive groups, 500 safety
noncritical/hard-negative groups, 150 OOD groups, 30 held-out traces, and six
held-out trace families (or the values frozen in that run's protocol).

The summary contains one H2 row per frozen comparator plus exactly one H3 and
one H4 row and references exactly one immutable raw artifact and SHA-256. It is
not result evidence. The evaluator opens and hashes the artifact, reconstructs
correct-timely action, critical misses, false alerts, high-confidence semantic
errors, selective risks/coverages, latency, bytes, and energy; reruns clustered
bootstrap intervals and Holm correction; and recomputes paired-design, sample,
budget, and frozen calibration gates. Every reported value and flag must agree
with that recomputation. Missing recomputation inputs block support, and a
crafted summary cannot certify H2--H5.

H4 uses the validation-frozen 10% matched-coverage rule, not a post-hoc
confidence threshold: within each visual group and each of `proposed_full` and
`ablate_evidence_escalation`, select the top calibrated semantic rows using the
label-blind rule `top_calibrated_semantic_score_within_visual_group`. The raw
artifact must record `selected_mode` and a finite calibrated score. Selection
counts must match exactly between policies in every visual group; insufficient
semantic rows make H4 and H5 ineligible. Record the frozen target coverage,
selection-rule name, and recomputed coverage-gate result in the summary.
The H4 target-device row must reproduce the frozen group-matched coverage
estimand: the top calibrated semantic scores at the protocol's predeclared 10%
coverage, with identical selected counts for method and reference inside every
visual group. Two positive but unequal coverages are not sufficient evidence.

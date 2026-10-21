# Low-evidence detector packet correction — v2r13

## Observed stop

The operator's v2r12 validation execution reached Experiment 2 native receiver
cache construction. A decoded visual frame was processed by the frozen detector
and `DetectorOnlyBackend.infer`. Its packet failed the existing semantic validator:

> Insufficient evidence requires event=insufficient_evidence and recommended_action=REVIEW

The traceback ends before `run_validation_perception_and_frontier` returns.
It does not identify the affected sample or supply an accuracy outcome. It is
not evidence that Experiment 2, later experiments, or the untouched test finished.

## Cause and correction

The detector comparator admits registered priority candidates at confidence 0.05.
Its separate sufficient-evidence condition requires at least one retained object
at confidence 0.35. A priority candidate below 0.35 can therefore exist in a
frame that lacks sufficient evidence. The former event selection gave an ROI or
outside-ROI event precedence over that frame state. The action was already REVIEW,
but the event contradicted `sufficient_evidence=False`.

v2r13 sets `event="insufficient_evidence"` whenever the existing evidence condition
is false, after mission-profile validation and before the unchanged action rule.
This also corrects the equivalent low-confidence flame/smoke branch in the legacy
wildfire comparator. No detection is dropped or strengthened. Confidence cutoffs,
ROI geometry, object labels, risk, uncertainty, action rules, semantic validators,
prompts, model set, measurement repetitions, sample design, and hypothesis gates
remain unchanged. Unknown mission profiles still fail validation.

## Preserved inference and recomputation

Every previously valid detector packet already satisfied the invariant. The
correction changes only packets that the predecessor validator rejected; valid
cached detector packets and all VLM inference outputs are unchanged. A local
255-case comparison against the v2r12 detector implementation found 229 previously
valid packet JSON strings byte-identical and 26 previously rejected low-evidence
packets corrected to the required insufficient-evidence/REVIEW state. These are
software regression checks, not empirical research results.

The existing authenticated v2r9 raw-inference recovery remains limited to 132
artifacts and 4,152 rows, under recovery receipt SHA-256
`6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608`.
Existing checksum, request, model, row-coverage, and cache-context checks remain
active. No historical release, raw shard, or receipt is rewritten. This does not
authorize recovery of derived summaries, selections, calibrated semantic packets,
receiver caches, or later experiment results.

v2r13 recomputes derived Experiment 1 outputs and Experiment 2 onward after a fresh
three-model development smoke and scientific environment seal. Native receiver
cache construction is committed only after its complete validation. Measurements
held in memory at the v2r12 failure must therefore be repeated. No claim is made
that all prior wall-clock time is recoverable.

## Regression coverage

The correction is checked through the actual detector policy and semantic
validator, including the native visual receiver, ROI crops, causal clips,
confidence thresholds, mixed priority/nonpriority evidence, and profile rejection.
No validator is disabled, and a failed schema is never relabeled as a PASS.

The release preserves the authenticated candidate environment and historical
artifacts. It does not install packages, train models, download models, access the
untouched test, or freeze the final scientific protocol during deployment.

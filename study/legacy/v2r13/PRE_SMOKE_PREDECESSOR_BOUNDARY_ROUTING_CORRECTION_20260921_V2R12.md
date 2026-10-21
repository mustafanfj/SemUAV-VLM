# Pre-smoke predecessor-boundary routing correction (v2r12)

## Scope

This source-only correction changes no scientific hypothesis, dataset split,
model set, prompt, generation policy, measurement method, metric, statistical
procedure, or untouched-test rule. It performs no scientific experiment and
does not authorize untouched-test access.

## Observed v2r11 stop

The authenticated v2r11 source deployment completed successfully. During the
prepare phase, `_verify_candidate_ready_receipt` rejected the authentic v2r11
deployment receipt (SHA-256
`befdce3067b571933dd94d35758c2c29385a06a8bb02ea93ae681117dcc17f3e`)
with `StudyGateError` before runtime smoke, validation software acceptance, or
Experiments 1–7. The receipt correctly carried the immediate predecessor's v2r10
acceptance-stop boundary, while the notebook verifier incorrectly evaluated
that field against the older v2r9 native-receiver terminal-boundary schema.

Prepare-only scaffolding may already exist: the derived three-model lock and
derivation receipt, plus an empty v2r11 `RunContext`/output-directory shell.
No science facts were produced: no runtime smoke, recovery import, scientific
experiment, protocol freeze, untouched-test access, or report/figure generation
occurred.

## Correction

v2r12 verifies each terminal boundary only at its authenticated provenance layer:

- the v2r12 deployment receipt carries the exact v2r11 pre-smoke stop;
- the authenticated v2r11 receipt remains byte-for-byte unchanged and retains
  its v2r10 acceptance-stop boundary;
- the authenticated v2r10 lineage retains the v2r9 native-receiver boundary;
- hybrid boundaries, missing fields, and unbound extra fields are rejected.

## Recovery continuity

The existing hash-bound v2r9 Experiment-1 recovery receipt remains unchanged:
132 raw shard artifacts containing 4,152 inference rows. Only those raw rows
may be imported into the exact v2r12 full run. Model selection, semantic
packets, receiver caches, Experiment-1 derived results, and Experiments 2–7
must be recomputed. A fresh v2r12 three-model development runtime smoke and a
new scientific-use environment seal remain mandatory.

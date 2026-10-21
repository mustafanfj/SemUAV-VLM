# SemUAV-VLM v3r1 repair notes

## What failed

The supplied workstation log shows that v3 deployment reached its full CPU
acceptance suite and then failed while importing test helpers. Two tests in
`test_visdrone_v3_experiment_flow.py` could not import
`tests.test_visdrone_runner_contract`. The
`test_visdrone_v3_receiver_checkpoints.py` module could not import
`tests.test_visdrone_receivers`.

These are test import defects in our delivered release. The earlier
`Software acceptance tests: PASS` line in that log did not mean that the full
deployment passed. The final full-suite result was a failure, so the v3 launcher
and notebook must not be used as if deployment had completed.

The same failure pattern was reproduced locally with an installed-style
package named `tests` shadowing the release's test directory. The workstation
log confirms the unresolved imports but does not identify which `tests` package
Python resolved there. The exact installed-package collision on that workstation
remains an inference; its installed package was not inspected.

## Corrected release boundary

V3r1 changes the three helper imports to the sibling module names used by test
discovery, removing their dependency on the ambiguous top-level `tests` package
name. A regression check covers the shadow-package condition. The correction
is packaged as a separately sealed source release.

Its deployment must pass the complete CPU acceptance
suite before writing a ready receipt and reporting
`V3R1_SOURCE_DEPLOYMENT_COMPLETE`. The acceptance requirement is not bypassed.

The operator uses the new `SemUAV_V3R1_Release.zip` and its included deploy and
launch scripts. The new source sibling is
`release_20260928_visdrone_paper_v3r1`; the receipt directory is
`private/visdrone_v3r1_deployment`; the kernel is
**Python (SemUAV-VLM 2026-09-28 v3r1)**; and the run name is
`semuav_visdrone_v3r1`.

Leave the partial v3 deployment, failure log and predecessor releases intact.
This repair does not call for deleting an environment, changing installed
packages, training models or downloading model/dataset assets. It continues to
use the existing v2r3 candidate and frozen assets under the release's receipt
and hash checks.

## Running and interpreting the result

Follow `START_HERE.md` in order. If checksum verification or deployment fails,
stop and retain the exact error. Do not loosen a gate or modify a receipt.
After deployment passes, the fresh development model smoke is still required
before validation. A successful deployment is software evidence; it is not a
GPU/model smoke result or a completed scientific campaign.

The existing scientific controls remain in force: edit only the ten parameter
assignment values, complete validation before freezing, verify the backup and
authorization before the one-time test, and generate the read-only report only
after a COMPLETE test ledger. A test attempt recorded as OPEN remains consumed
even if execution fails.

See `VALIDATION_REPORT.md` for the checks actually completed on the corrected
package. This document makes no claim that remote GPU experiments have run.

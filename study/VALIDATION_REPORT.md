# SemUAV-VLM v3r2 software verification

Release: `release_20261013_visdrone_paper_v3r2` (pre-test; the untouched test
split has never been opened). The v3r1 report follows below for history.

## Completed for v3r2 (CPU, offline)

| Check | Observed result |
|---|---|
| Full CPU suite | **594 tests passed**; with `cbor2` absent, 593 passed and 1 was skipped (the CBOR decode test runs on the lab candidate, which has `cbor2`) |
| CBOR decode regression | An `l0_cbor` heartbeat through Experiment 3 gives `decode_error` with v3r1 source and `applied` with v3r2 source (checked with a minimal local CBOR codec) |
| Reducer column-order regression | The new test fails on v3r1 source and passes on v3r2 |
| Controller revision | Untrusted calibration: no evidence escalation for high uncertainty, and no forced escalation from a high-confidence-error placeholder. Balanced success equals the mean of per-class recall |
| Calibration amendment | The gate passes only for a measured FAIL under the v3r2 untrusted-calibration controller. Every other combination still fails. H4 is `NOT_ESTIMABLE` when calibration is untrusted |
| v3r1 reuse | Exact-key reuse, including when a shard index shifts. A changed input leads to recomputation. `04_*` stages are never reused. Tampered bytes, a test-accessed source and configuration drift are all refused. Stale pre-fix Experiment 3 rows lead to recomputation |
| Deployment integration | 11 tests, including pinning and verifying the v3r1 inventory, excluding controller trials, and refusing a test-accessed or changed source |
| Configuration preflight | A v3r1-style configuration loaded from a different release root with the derived launch environment compares equal |
| Raw producers | Perception, model, measurement, hardware, prompt, canary and requirements files are byte-identical to the authenticated v2r13 inventory |
| Notebook | 61 cells (59 + the validation paper-figures cell and its heading); every code cell parses; the v3r2 kernel, release, receipt and run namespaces agree |
| Paper-figure scripts | Run under `python -I -B` on the v3r1 validation data and produce all figures and tables |
| Independent review | A separate review of the v3r1→v3r2 diff found no blocking issue. Its findings (Experiment 2 stage-name scope, stale Experiment 3 guard, untrusted high-confidence error, launcher `eval`, documentation) were fixed before packaging |

## Not yet established

GPU smoke, validation, freeze and the test campaign must run on the
workstation. Whether a given v3r1 stage is reused is decided at run time by
exact key match. The notebook prints how many stages were reused and how many
were computed.

---

# SemUAV-VLM v3r1 software verification

Release: `release_20260928_visdrone_paper_v3r1`  
Correction of v3 source archive: `25b9a80da04e95d15ddc9a46ca8de0682e377def9913c768f67b38db0b3e83ae`

## Observed failure and correction

The supplied remote v3 deployment log reports **568 tests, zero failures, three
errors, zero skipped**. All three errors were unresolved test-helper imports.
Five checkpoint test cases were replaced by one failed module-import case;
two other test methods also failed to import helpers. The prior local 572-test
pass did not establish compatibility with the workstation's import environment.

The original error pattern was reproduced locally by loading an unrelated
installed-style `tests` package. Three imports now use the sibling-module names
established by `unittest discover -s tests`. No installed package is removed or
upgraded. The exact shadowing package on the workstation has not been inspected.

## Completed for v3r1

| Check | Observed result |
|---|---|
| Full corrected CPU suite | **574 tests passed**, 44.248 seconds |
| Import-collision regression | All 15 affected tests pass with the unrelated `tests` package loaded |
| Deployment integration | Nine tests passed, including actual CPU fixture execution, tamper rejection, and preserving preexisting v3 source/receipt files |
| Notebook | 59 cells; all code cells parse; v3r1 kernel, release, receipt, run and cache namespaces agree |
| Shell scripts | `bash -n` passed for deployer and launcher |
| Scientific source comparison | All `semuav` modules byte-identical to v3 except explicit v3r1 raw-recovery allowlist and its error message |
| Registered configuration | Only study ID, run-cache namespace and trusted-code cache namespace changed |
| Predecessor compatibility | Raw inference, models, measurement, prompts, canary and requirements remain byte-identical to the authenticated v2r13 inventory |
| Original v3 delivery | Original outer archive remains unchanged: `9a4c3f0cd8f9f39988d4ff3454262e6c1de554f654cd6cd68f60f1856d9fd34b` |

Full-suite command from the corrected source root:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests -q
```

Local runtime: Python 3.12.14, FFmpeg 6.1.1. The count increased from 572 to
574 because of the shadow-package regression and the separate-v3r1-deployment
preservation regression. Subtests and the 15 nested shadow-package checks are
not counted as additional independent top-level unittest methods.

`LOCAL_SOFTWARE_ACCEPTANCE.json` and `.log` in the delivery are produced after
source sealing by the deployment module's actual isolated CPU-test harness.
They are local software evidence. The workstation must execute its full gate
in its existing Python 3.10 candidate environment before it writes its own
`DEPLOYMENT_READY.json` and prints `V3R1_SOURCE_DEPLOYMENT_COMPLETE`.

## Preserved v3 implementation

The odd-dimension H264/H265 padding and receiver geometry restoration, native
receiver row checkpoints, synchronous experiment-result/figure callbacks,
matched-support Pareto plots and final report checks are unchanged. The original
v3 verification included real codec round trips (including 1360 x 765) and all
11 figure types using non-scientific fixtures. V3r1's complete suite continues
to exercise the codec, recovery, notebook and scientific compatibility tests.
The inherited v3 changes and limits are recorded in `V3_AUDIT_AND_CHANGES.md`.

## Workstation evidence still required

- Successful v3r1 source deployment against private receipts and 132 raw shards.
- Fresh three-model runtime smoke and scientific-use environment seal.
- CUDA/NVML execution on physical GPU 1 in the preserved candidate environment.
- Validation Experiments 1–7, protocol freeze, authorized untouched test, and
  the final report derived from measured results.

Local deployment tests use temporary fixtures and mock unavailable candidate
runtime facts. No remote deployment, GPU smoke, scientific results, improved
accuracy, hypothesis support or publication acceptance are claimed here.
The full gate, source hashes, raw-recovery hashes and one-time test controls
remain mandatory. Follow `VISDRONE_PAPER_RUN_GUIDE.md`.

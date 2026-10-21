# Run SemUAV-VLM v3r2 on the NYUAD workstation

v3r2 is the pre-test successor of v3r1. It does three things:

- It moves the four receipted v3r1 in-kernel corrections into source.
- It registers the pre-test calibration-gate amendment
  (`PRE_TEST_CALIBRATION_GATE_AMENDMENT_20261013_V3R2.md`).
- It reuses exact, hash-verified v3r1 validation computations wherever the
  producer code is unchanged.

The sequence is:

**deploy → prepare (smoke) → validation → freeze → test once → report.**

Deploying v3r2 leaves the v3r1 source, run, receipts and results untouched.
v3r2 uses its own source sibling, receipt directory, kernel and run name
(`semuav_visdrone_v3r2`).

## 0. Before you start

- Get supervisor sign-off on the amendment record. Freezing the protocol
  binds it.
- In the v3r1 Jupyter, **shut down the v3r1 kernel**
  (Kernel → Shut Down Kernel).
- Stop the v3r1 Jupyter server: press Ctrl+C twice in its tmux pane.

The deploy script refuses to run while the v3r1 kernel is alive, because it
inventories the v3r1 run.

## 1. Transfer (Windows PowerShell)

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\SemUAV_V3R2_Release.zip" -DestinationPath "$env:USERPROFILE\Downloads" -Force -ErrorAction Stop
scp -r "$env:USERPROFILE\Downloads\SemUAV_V3R2_Release" nyuad-lab:~/
ssh nyuad-lab
```

## 2. Deploy (Ubuntu, about 15–30 minutes)

```bash
bash "$HOME/SemUAV_V3R2_Release/deploy_semuav_visdrone_v3r2_20261013.sh"
```

The script does the following, in order:

1. Checks the delivery checksums.
2. Deploys the sealed source and the v3r2 kernel.
3. Hashes the reusable v3r1 validation artifacts into
   `private/visdrone_v3r2_deployment/V3R1_VALIDATION_IMPORT_SOURCE.json` and
   pins that file in the deployment receipt.
4. Runs the full CPU test suite.
5. Checks that the v3r2 configuration equals v3r1 except for release
   identity.

Continue only if the last line reads `V3R2_DEPLOYMENT_AND_PREFLIGHT_COMPLETE`.
On any `STOP`, keep the full message and do not edit receipts or sources.

## 3. Run phases headlessly (recommended; survives a closed laptop)

Inside tmux, `run_semuav_v3r2.sh` does the same as editing the ten controls
and choosing **Restart Kernel and Run All**, one phase after another:

- It sets the controls in the deployed notebook, checked against the sealed copy.
- It executes every cell in the v3r2 kernel.
- After every cell, it saves the executed notebook and a log to
  `~/semuav_v3r2_outputs/`.

```bash
export TMUX_TMPDIR='/data/Lane Detection Datasets/musta/semuav_vlm/tmp/tmux'
mkdir -p "$TMUX_TMPDIR"
tmux new-session -s semuav-v3r2
bash "$HOME/SemUAV_V3R2_Release/run_semuav_v3r2.sh" prepare validation
```

Detach with Ctrl+B, then D. You can then close the laptop.

- To check progress at any time: `bash ~/SemUAV_V3R2_Release/status_semuav_v3r2.sh`
- When the log shows `FINISHED validation OK`, pack the results with
  `bash ~/SemUAV_V3R2_Release/bundle_semuav_v3r2_validation.sh`. This creates
  `~/semuav_v3r2_validation_bundle.tar.gz`.

The validation phase now ends with a **paper-figures cell**. It redraws
Experiments 1–7 with class-balanced metrics and 95% confidence intervals from
the run's own results. The figures are shown in the executed notebook and saved
under the run's `paper_figures_validation` output folder.

Use section 4 below only if you prefer to run interactively. Never run both
at once.

## 3b. Optional: start Jupyter for interactive use

```bash
export TMUX_TMPDIR='/data/Lane Detection Datasets/musta/semuav_vlm/tmp/tmux'
mkdir -p "$TMUX_TMPDIR"
tmux new-session -s semuav-visdrone-v3r2
bash "$HOME/SemUAV_V3R2_Release/launch_semuav_visdrone_v3r2_20261013.sh"
```

Detach with Ctrl+B, then D. Then:

1. Open the usual SSH tunnel from Windows (port 8888).
2. Open the token URL in your browser.
3. Open `SemUAV_VLM_VisDrone_Paper_Study.ipynb`.
4. Select the kernel **Python (SemUAV-VLM 2026-10-13 v3r2)**.

## 4. Edit only the ten control values; then Kernel → Restart Kernel and Run All Cells

Each phase below changes only the right-hand-side values of the existing ten
lines in the parameter cell. Edit the values in place; do not paste new lines
or add cells. Save the notebook before restarting.

### 4a. Prepare: fresh three-model smoke (about 30–60 minutes)

```python
PHASE = "prepare"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v3r2"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = True
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

All of these must appear:

- `MODEL_RUNTIME_SMOKE_3_OF_3_PASS`
- `TRUSTED_CODE_CACHE_FROZEN`
- `INFERENCE_ENVIRONMENT_SEALED_FOR_SCIENTIFIC_USE`

### 4b. Validation, reusing v3r1 (about 1–2.5 days)

```python
PHASE = "validation"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v3r2"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = True
BUILD_FEATURE_CACHE = True
```

Watch the output for these markers:

- `V3R1_VALIDATION_IMPORT_PINNED`, near the start.
- `[v3r1-import] reused …` lines.
- `V3R1_VALIDATION_REUSE_AFTER_EXPERIMENTS_1_2`, which shows how many stages
  were reused and how many were computed.
- `V3R2_CONTROLLER … calibration_trusted= False best_fixed= …`

Perception inference, the detector stage, native receiver and feature caches,
and the Experiment 2 frontier should be reused. Experiment 3 shards are
reused only where their inputs reproduce byte-identically; otherwise they
are recomputed, which takes about 1.5 days. Experiment 4 (about half a day)
is always recomputed. Either way the results are correct.

### 4c. Freeze (about 1–3 hours; reuses everything from 4b)

```python
PHASE = "validation"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = True
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v3r2"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

Expected output:

- `CALIBRATION_GATE_FAIL_ROUTED_BY_AMENDMENT`. The calibration FAIL stays on
  record, and H4 becomes NOT_ESTIMABLE.
- A frozen protocol.

Back up the run's `results` folder and the frozen protocol before continuing.

### 4d. Untouched test, one attempt (about 12–15 days)

```python
PHASE = "test"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = True
RUN_NAME = "semuav_visdrone_v3r2"
RESUME = False
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

Never rerun the test as troubleshooting. The OPEN ledger consumes the attempt.
Native receiver measurement on the 960 test temporal frames dominates the
runtime. Experiments 3 and 4 on 15 sequences × 30 traces add about 3–4 days.

### 4e. Report (a few hours)

```python
PHASE = "report"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v3r2"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

## Monitoring and reconnecting

To reattach to the session:

```bash
export TMUX_TMPDIR='/data/Lane Detection Datasets/musta/semuav_vlm/tmp/tmux'
tmux attach-session -t semuav-visdrone-v3r2
```

The run folder is
`/data/Lane Detection Datasets/musta/semuav_vlm/.cache/visdrone_paper_v3r2/runs/semuav_visdrone_v3r2-full-*`.
Its `status.sqlite` lists every stage. Reused stages carry the message
`IMPORTED_V3R1 …`.

## Keep v3r1 in place

Do not move or delete the v3r1 run folder (`.cache/visdrone_paper_v3r1/...`)
until the v3r2 report is complete. Every v3r2 launch reads its
`run_manifest.json` to reproduce the same data and model locations.

## What must not change

Do not alter any of the following to obtain a favourable result:

- models, prompts or samples
- thresholds or gates
- the amendment

A null result or `NOT_ESTIMABLE` is kept as is.

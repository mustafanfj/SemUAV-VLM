# SemUAV-VLM VisDrone paper study v2r13: exact NYUAD run guide

This is the sole operator guide for source release
`release_20260921_visdrone_paper_v2r13`. It starts a new run after v2r12
stopped in the native Experiment 2 visual receiver on an inconsistent
low-evidence detector packet.
It preserves the v2r9, v2r10, v2r11, and v2r12 runs. It may
copy only the deployment-frozen raw v2r9 inference shards after exact identity,
cache-key, hash, content, and coverage verification.

Read first:

- [`PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md`](PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md);
- [`PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md`](PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md);
- [`PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md`](PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md);
- [`PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md`](PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md);
- [`PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md`](PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md);
- [`PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md`](PRE_VALIDATION_MODEL_SET_AMENDMENT_20260914_V2R6.md);
- [`VALIDATION_REPORT.md`](VALIDATION_REPORT.md), the current v2r13 recovery
  boundary and scientific-state report; and
- this guide.

## 1. Scientific state and amended design

v2r12 passed its fresh 3/3 smoke and scientific-use seal, then stopped during
native Experiment 2 visual receiver execution. A priority detection could
exceed the registered 0.05 priority threshold while remaining below the
unchanged 0.35 sufficient-evidence threshold. `DetectorOnlyBackend.infer`
then combined `sufficient_evidence=false` and `REVIEW` with a positive event,
which the strict semantic validator rejected. v2r13 assigns the required
`insufficient_evidence` event when evidence is false. Thresholds, actions,
models, and scientific design remain unchanged. A fresh 3/3 smoke and seal
are required under the new source identity.

v2r9 completed an exact 3/3 model smoke and scientific-use seal. It completed
the validation input gate and all registered raw Experiment 1 detector/VLM
inference: the general detector plus three VLM routes, deterministic compact
selection, and the temporal detector plus selected compact route. It stopped
when Experiment 2 decoded its first `l0_cbor` heartbeat. L0 intentionally has
no `recommended_action`; the v2 receiver incorrectly treated every semantic
variant as action-bearing and rejected the missing field.

v2r10 advanced the native fusion policy to
`semuav.visdrone-native-fusion.v3`. A strictly validated L0 heartbeat now
returns the registered `NO_ACTION`; L1/L2 still require an exact registered
action, and malformed or unknown variants remain fail-closed. This is a
downstream receiver correction, not a change to a model, prompt, payload,
sample, endpoint, threshold, or hypothesis. v2r13 inherits that correction
unchanged.

v2r10 then completed its deployment, fresh 3/3 smoke, and environment seal,
but stopped in software acceptance before Experiment 1. Its unit-test child
had inherited the parent-only recovery authorization. v2r11 removed exactly
those three recovery variables from the child, and v2r13 inherits that
isolation unchanged. v2r11 then stopped before runtime smoke because its
immediate-predecessor boundary was checked against the older v2r9 schema;
v2r12 corrected that routing defect, and v2r13 preserves its historical
provenance layers. No v2r10, v2r11, or v2r12 derived scientific result or
downstream cache is reused.

To avoid repeating hours of identical inference, the deployer reverifies the
existing exact v2r9 recovery inventory frozen under the v2r10 deployment.
Validation may import only 132 raw Experiment 1 shard
files containing 4,152 rows. v2r13 reconstructs each predecessor cache key,
rehashes and validates the source, and writes a byte-identical copy under the
new v2r13 cache key. All summaries, calibration, selection, semantic packets,
native/feature caches, Experiment 2 and later experiments are recomputed. The
scientific contract, including the inherited temporal-validation capacity and
cell gate, is unchanged:

| Item | Exact v2r13 value |
|---|---:|
| Temporal validation sequences | 7 |
| Observations per sequence | 64 |
| Source stride | 1 |
| Replay rate | 5 Hz |
| Validation traces | 12 |
| Fully paired temporal-validation cells | 84 (= 7 × 12) |
| Static validation minimum | 15 groups, unchanged |
| Temporal test sequences | 15, unchanged |
| Held-out test traces | 30, unchanged |
| Planned temporal test cells | 450, unchanged |

The v2r13 model population remains exactly:

1. `qwen3_vl_2b` — compact selection candidate;
2. `smolvlm2_2b` — compact selection candidate; and
3. `qwen3_vl_8b_upper_bound` — non-selectable upper-bound reference.

Moondream remains excluded under the v2r6 model-set amendment. The
v2r5 Starmie tokenizer cache is historical failed-run evidence only. v2r13 does not acquire, export, bind, or
verify a Moondream tokenizer. Do not rerun v2r5 or copy any of its PASS rows.
Do not copy a v2r9 smoke row or treat a recovered raw shard as a v2r13 result.
v2r13 must run all three retained models again for its development smoke and
freshly recompute the exact seven-sequence validation capacity. The recovery
only avoids repeating the already-completed raw Experiment 1 inference.

### Fastest valid route to results

1. Deploy v2r13 and save its printed deployment-receipt SHA-256.
2. Run `prepare` once and require the fresh v2r13 3/3 smoke and seal.
3. Run `validation` once. The 132 raw shards should verify/copy quickly; the
   derived Experiment 1 products, native/feature caches, and Experiments 2–7
   still execute normally.
4. Freeze and back up the validated protocol. Development results are then
   available, but final paper figures are not yet confirmatory results.
5. After supervisor approval, run the untouched test exactly once, then run
   `report` to create the final tables and figures.

Do not start a second notebook/kernel, delete either run tree, or manually copy
cache files. Those actions defeat the authenticated recovery path and can cost
more time than they save.

## 2. Fixed identities

| Item | Exact value |
|---|---|
| Host/user | [redacted lab workstation] |
| Study root | `/data/Lane Detection Datasets/musta/semuav_vlm` |
| Release | `$STUDY_ROOT/releases/release_20260921_visdrone_paper_v2r13/SemUAV-VLM-VisDrone-Paper-Study` |
| Run/output namespace | `semuav_visdrone_v2r13` |
| Reused candidate | `$STUDY_ROOT/venvs/release_20260912_visdrone_inference_candidate_v2r3` |
| Kernelspec | `semuav-vlm-20260921-v2r13` |
| Kernel display | `Python (SemUAV-VLM 2026-09-21 v2r13)` |
| Trusted-code cache | `$STUDY_ROOT/.cache/huggingface/modules-v2r13` |
| Deployment receipt | `$STUDY_ROOT/private/visdrone_v2r13_deployment/DEPLOYMENT_READY.json` |
| Receipt environment variable | `SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256` |
| GPU | Physical 1 exposed alone as logical `cuda:0` |
| Required smoke | Fresh exact 3/3 PASS under v2r13 |
| Authorized recovery | 132 v2r9 raw Experiment 1 shards / 4,152 rows only |

The candidate remains v2r3. Never rename or mutate it and never install,
remove, or upgrade a package during this study.

## 3. Upload and deploy

All four predecessor runs must be quiescent. Close only the positively
identified v2r12, v2r11, v2r10, and v2r9 tmux sessions; this terminates old
Jupyter/kernel processes but does not delete or alter their run files:

```bash
set -Eeuo pipefail

STUDY_ROOT='/data/Lane Detection Datasets/musta/semuav_vlm'
export TMUX_TMPDIR="$STUDY_ROOT/tmp/tmux"

for SESSION in \
  semuav-visdrone-v2r12 \
  semuav-visdrone-v2r11 \
  semuav-visdrone-v2r10 \
  semuav-visdrone-v2r9
do
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux list-panes -t "$SESSION" \
      -F 'closing pane=#{pane_id} pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead}'
    tmux kill-session -t "$SESSION"
    printf '%s=CLOSED\n' "$SESSION"
  else
    printf '%s=ALREADY_ABSENT\n' "$SESSION"
  fi
done

sleep 2
ss -H -ltnp 'sport = :8888' 2>&1 || true
nvidia-smi -i 1 \
  --query-compute-apps=pid,process_name,used_memory \
  --format=csv,noheader
```

The final two commands must print no listener and no compute process. If they
print anything, identify it before proceeding; do not kill an arbitrary PID.

The final handoff will supply SHA-256 values for:

- `SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip`; and
- `deploy_semuav_visdrone_v2r13_20260921.sh`.

Do not run this section until those two authenticated files are supplied. From
Windows PowerShell:

```powershell
$DownloadRoot = Join-Path $HOME 'Downloads'
$Archive = Join-Path $DownloadRoot 'SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip'
$Deployer = Join-Path $DownloadRoot 'deploy_semuav_visdrone_v2r13_20260921.sh'
Get-FileHash -Algorithm SHA256 $Archive
Get-FileHash -Algorithm SHA256 $Deployer
scp $Archive nyuad-lab:~/
scp $Deployer nyuad-lab:~/
```

After both hashes match the handoff, SSH to the lab and run:

```bash
set -Eeuo pipefail

STUDY_ROOT='/data/Lane Detection Datasets/musta/semuav_vlm'
ARCHIVE_NAME='SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip'
DEPLOYER_NAME='deploy_semuav_visdrone_v2r13_20260921.sh'

mkdir -p "$STUDY_ROOT/incoming"
install -m 0644 "$HOME/$ARCHIVE_NAME" "$STUDY_ROOT/incoming/$ARCHIVE_NAME"
install -m 0755 "$HOME/$DEPLOYER_NAME" "$STUDY_ROOT/incoming/$DEPLOYER_NAME"
sha256sum -- "$STUDY_ROOT/incoming/$ARCHIVE_NAME" "$STUDY_ROOT/incoming/$DEPLOYER_NAME"
bash "$STUDY_ROOT/incoming/$DEPLOYER_NAME"
```

The deployer is source-only. It must not install packages, use CUDA, load model
weights, read validation/test data, run a smoke, or authorize science. Save the
printed `SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256` exactly.

## 4. Confirm a clean v2r13 launch boundary

After deployment, the v2r13 session, TCP port 8888, and physical GPU 1 must
still be clean. Never kill an arbitrary PID or another user's process.

```bash
(
set -Eeuo pipefail

STUDY_ROOT='/data/Lane Detection Datasets/musta/semuav_vlm'
export TMUX_TMPDIR="$STUDY_ROOT/tmp/tmux"
mkdir -p "$TMUX_TMPDIR"
chmod 700 "$TMUX_TMPDIR"

if tmux has-session -t semuav-visdrone-v2r13 2>/dev/null; then
  echo 'STOP: inspect the existing v2r13 tmux session'
  tmux list-panes -t semuav-visdrone-v2r13 \
    -F 'pane=#{pane_id} pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead}'
  exit 1
else
  echo 'V2R13_TMUX_SESSION=ABSENT'
fi

if ss -H -ltnp 'sport = :8888' | grep -q .; then
  echo 'STOP: port 8888 is occupied'
  ss -H -ltnp 'sport = :8888'
  exit 1
else
  echo 'PORT_8888=FREE'
fi

GPU_STATE="$(
  nvidia-smi -i 1 \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>&1
)" || {
  echo 'STOP: physical GPU 1 occupancy query failed'
  printf '%s\n' "$GPU_STATE"
  exit 1
}
if [ -n "$GPU_STATE" ]; then
  echo 'STOP: physical GPU 1 has an existing compute process'
  printf '%s\n' "$GPU_STATE"
  exit 1
else
  echo 'PHYSICAL_GPU_1=FREE'
fi
)
```

If the port or GPU is occupied, identify its owner and full command first.

## 5. Start Jupyter in a fresh tmux session

```bash
set -Eeuo pipefail

STUDY_ROOT='/data/Lane Detection Datasets/musta/semuav_vlm'
export TMUX_TMPDIR="$STUDY_ROOT/tmp/tmux"
mkdir -p "$TMUX_TMPDIR"
chmod 700 "$TMUX_TMPDIR"
if tmux has-session -t semuav-visdrone-v2r13 2>/dev/null; then
  echo 'STOP: v2r13 tmux session appeared after the clean-boundary check' >&2
  exit 1
fi
if ss -H -ltnp 'sport = :8888' | grep -q .; then
  echo 'STOP: port 8888 became occupied after the clean-boundary check' >&2
  exit 1
fi
GPU_STATE="$(
  nvidia-smi -i 1 \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader 2>&1
)" || {
  echo 'STOP: physical GPU 1 occupancy recheck failed' >&2
  printf '%s\n' "$GPU_STATE" >&2
  exit 1
}
if [ -n "$GPU_STATE" ]; then
  echo 'STOP: physical GPU 1 became occupied after the clean-boundary check' >&2
  printf '%s\n' "$GPU_STATE" >&2
  exit 1
fi
tmux new-session -d -s semuav-visdrone-v2r13 '/bin/bash --noprofile --norc'
tmux attach-session -t semuav-visdrone-v2r13
```

Inside the new pane, paste:

```bash
set -Eeuo pipefail

unset BASH_ENV ENV LD_PRELOAD LD_LIBRARY_PATH PYTHONPATH PYTHONHOME
unset SEMUAV_QWEN_REVISION SEMUAV_SMOLVLM_REVISION
unset SEMUAV_UPPER_BOUND_REVISION SEMUAV_MOONDREAM_REVISION
unset SEMUAV_MOONDREAM_TOKENIZER_LOCK
unset SEMUAV_ENABLE_V2R9_EXP1_RECOVERY
unset SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT
unset SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256
export PYTHONNOUSERSITE=1

STUDY_ROOT='/data/Lane Detection Datasets/musta/semuav_vlm'
REPO_ROOT="$STUDY_ROOT/releases/release_20260921_visdrone_paper_v2r13/SemUAV-VLM-VisDrone-Paper-Study"
VENV_ROOT="$STUDY_ROOT/venvs/release_20260912_visdrone_inference_candidate_v2r3"
REUSED_V1="$STUDY_ROOT/outputs/visdrone_paper/semuav_visdrone_v1/frozen_inputs"

read -r -p 'Paste SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256: ' \
  SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256
export SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256
case "$SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256" in
  *[!0-9a-f]*|'') echo 'STOP: invalid v2r13 receipt SHA-256' >&2; return 1 2>/dev/null || exit 1 ;;
esac
[ "${#SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256}" -eq 64 ] || {
  echo 'STOP: receipt SHA-256 must contain 64 characters' >&2
  return 1 2>/dev/null || exit 1
}

export PATH="$VENV_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
cd "$REPO_ROOT"

export TMPDIR="$STUDY_ROOT/tmp"
export XDG_CACHE_HOME="$STUDY_ROOT/.cache/xdg"
export XDG_CONFIG_HOME="$STUDY_ROOT/.config"
export HF_HOME="$STUDY_ROOT/.cache/huggingface"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export HF_MODULES_CACHE="$HF_HOME/modules-v2r13"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$STUDY_ROOT/.cache/torch"
export TORCH_EXTENSIONS_DIR="$STUDY_ROOT/.cache/torch_extensions"
export CUDA_CACHE_PATH="$STUDY_ROOT/.cache/cuda"
export TRITON_CACHE_DIR="$STUDY_ROOT/.cache/triton"
export YOLO_CONFIG_DIR="$STUDY_ROOT/.config/ultralytics"
export PIP_CACHE_DIR="$STUDY_ROOT/.cache/pip"
export JUPYTER_DATA_DIR="$STUDY_ROOT/.cache/jupyter/data"
export JUPYTER_CONFIG_DIR="$STUDY_ROOT/.cache/jupyter/config-v2r13"
export JUPYTER_RUNTIME_DIR="$STUDY_ROOT/.cache/jupyter/runtime-v2r13"
export IPYTHONDIR="$STUDY_ROOT/.cache/ipython-v2r13"
export MPLCONFIGDIR="$STUDY_ROOT/.cache/matplotlib-v2r13"

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$HF_HOME" "$HF_MODULES_CACHE" "$HF_DATASETS_CACHE" \
  "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" \
  "$TRITON_CACHE_DIR" "$YOLO_CONFIG_DIR" "$PIP_CACHE_DIR" \
  "$JUPYTER_DATA_DIR" "$JUPYTER_CONFIG_DIR" "$JUPYTER_RUNTIME_DIR" \
  "$IPYTHONDIR" "$MPLCONFIGDIR"
chmod 700 "$JUPYTER_RUNTIME_DIR"

export CUDA_VISIBLE_DEVICES=1
export SEMUAV_DEVICE='cuda:0'
export SEMUAV_STUDY_ROOT="$STUDY_ROOT"
export SEMUAV_PROJECT_ROOT="$REPO_ROOT"
export SEMUAV_DATA_ROOT="$STUDY_ROOT/data"
export SEMUAV_OUTPUT_ROOT="$STUDY_ROOT/outputs/visdrone_paper"
export SEMUAV_CACHE_ROOT="$STUDY_ROOT/.cache/visdrone_paper_v2r13"
export SEMUAV_RUN_NAME='semuav_visdrone_v2r13'
export SEMUAV_REUSED_V1_FROZEN_INPUT_DIR="$REUSED_V1"

export YOLO_AUTOINSTALL=false
export PIP_NO_INDEX=1
export PIP_NO_INPUT=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SEMUAV_MODELS_LOCAL_ONLY=1

test "$(command -v python)" = "$VENV_ROOT/bin/python"
test "$(python -I -B -c 'import sys; print(sys.prefix)')" = "$VENV_ROOT"
test "$HF_MODULES_CACHE" = "$STUDY_ROOT/.cache/huggingface/modules-v2r13"
test "$HF_HUB_CACHE" = "$STUDY_ROOT/.cache/huggingface"
test "$HUGGINGFACE_HUB_CACHE" = "$HF_HUB_CACHE"
test "$TRANSFORMERS_CACHE" = "$HF_HUB_CACHE"
test "$(command -v tc)" = '/usr/sbin/tc'
python -m pip check
echo 'V2R13_JUPYTER_BOOTSTRAP=PASS'

jupyter lab --no-browser --ip=127.0.0.1 --port=8888 \
  --ServerApp.port_retries=0
```

In Windows PowerShell, open the tunnel:

```powershell
ssh -o ExitOnForwardFailure=yes -N -L 8888:127.0.0.1:8888 nyuad-lab
```

Open <http://127.0.0.1:8888/lab>, select
**Python (SemUAV-VLM 2026-09-21 v2r13)**, and open the paper notebook.

## 6. Fresh development smoke

Change only these ten assignment right-hand sides in the parameter cell:

```python
PHASE = "prepare"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v2r13"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = True
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

Save, then use **Kernel → Restart Kernel and Run All Cells** once. Continue
only after `MODEL_RUNTIME_SMOKE_3_OF_3_PASS`,
`TRUSTED_CODE_CACHE_FROZEN`, and
`INFERENCE_ENVIRONMENT_SEALED_FOR_SCIENTIFIC_USE` all print. A 2/3 result or skipped upper bound keeps validation and test closed.

## 7. Run validation

After the fresh v2r13 seal, change only the ten values:

```python
PHASE = "validation"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v2r13"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = True
BUILD_FEATURE_CACHE = True
```

Save and **Restart Kernel and Run All Cells** once. Require the exact-capacity
marker before any temporal sample write. Before experiments begin, require all
three markers below with exact values:

```text
V2R13_TEMPORAL_FEASIBILITY_CONTRACT_VERIFIED validation_sequences=7 validation_traces=12 validation_paired_cells=84 test_sequences=15
V2R13_TEMPORAL_ELIGIBLE_CAPACITY_VERIFIED eligible_validation_sequences=7
V2R13_TEMPORAL_CARTESIAN_DESIGN_VERIFIED validation_sequences=7 validation_traces=12 validation_cells=84 test_sequences=15 test_traces=30 test_cells=450
```

General Experiment 1 has exactly `detector_only`, Qwen3-VL-2B, SmolVLM2,
and the 8B upper bound. Only the two compact models are selectable.

The notebook automatically enables the authenticated v2r9 raw-shard recovery
only in `validation`. Do not set recovery environment variables manually. If
any one of the 132 expected shards is missing, changed, incompatible, or has
the wrong row coverage, the run stops instead of silently mixing evidence.
Seeing the Experiment 1 summary after the verified imports is expected; it is
freshly recomputed from the validated rows. The time-consuming work that still
remains is native/feature-cache construction and Experiments 2–7.

Do not change a model, prompt, sample, threshold, score weight, tie-break,
campaign cell, or hypothesis gate after seeing validation outcomes. A null,
negative, failed preregistered hypothesis, or `NOT_ESTIMABLE` outcome remains a
valid scientific result and is not permission to tune the protocol.

## 8. Freeze, back up, test once, and report

Only after all validation gates pass, freeze with:

```python
PHASE = "validation"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = True
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v2r13"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

Verify the manifest-driven backup and obtain supervisor approval before test.
Then run the one-time test exactly once with:

```python
PHASE = "test"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = True
RUN_NAME = "semuav_visdrone_v2r13"
RESUME = False
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

The OPEN ledger consumes the attempt even if the run disconnects or fails.
After a `COMPLETE` ledger, report with:

```python
PHASE = "report"
EXECUTE_PHASE = True
FREEZE_PROTOCOL = False
AUTHORIZE_TEST_ONCE = False
RUN_NAME = "semuav_visdrone_v2r13"
RESUME = True
DEVICE = "cuda:0"
RUN_MODEL_SMOKE = False
BUILD_NATIVE_RECEIVER_CACHE = False
BUILD_FEATURE_CACHE = False
```

## 9. Recovery and hard stops

Reconnect only to the same session:

```bash
export TMUX_TMPDIR='/data/Lane Detection Datasets/musta/semuav_vlm/tmp/tmux'
tmux attach-session -t semuav-visdrone-v2r13
```

Stop on any identity/hash mismatch, unexpected model, smoke failure, absent
seal, temporal count other than validation 7/12/84 and test 15/30/450, cache
mutation, changed protocol, early test access, or second test authorization.
Any source correction requires another new release and run identity.

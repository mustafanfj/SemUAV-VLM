"""Extract compact, figure-ready data from a SemUAV run (read-only).

Usage (lab server, candidate Python):
    "$PY" extract_figdata.py test            # newest v3r2 run, test outputs
    "$PY" extract_figdata.py validation      # newest v3r2 run, validation outputs
    "$PY" extract_figdata.py test <run_root> # explicit run directory
    "$PY" extract_figdata.py validation <run_root> <out_dir>   # used by the notebook
Writes ~/semuav_figdata_<phase>.tar.gz, or <out_dir>/semuav_figdata when an
output directory is given. Nothing in the run is modified.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # run with python -I

import csv, gzip, json, shutil, sys, tarfile
PHASE = sys.argv[1] if len(sys.argv) > 1 else "test"
if PHASE not in {"validation", "test"}:
    raise SystemExit("phase must be 'validation' or 'test'")
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/data/Lane Detection Datasets/musta/semuav_vlm")
if len(sys.argv) > 2:
    RUN = Path(sys.argv[2])
else:
    runs = sorted((ROOT / ".cache/visdrone_paper_v3r2/runs").glob("semuav_visdrone_v3r2-full-*"))
    if len(runs) != 1:
        raise SystemExit(f"expected exactly one v3r2 run directory, found {len(runs)}; pass it explicitly")
    RUN = runs[0]
RES = RUN / "results"
print("run:", RUN)
if PHASE == "test" and not (RUN / "test_access_ledger.json").is_file():
    raise SystemExit("no test ledger in this run; the test campaign has not run")
# Result names differ between phases; figure scripts read the validation-style names.
NAMES = {
    "validation": {"exp2": "validation_exp2_frontier.csv.gz", "exp3": "validation_exp3_summary.csv.gz",
                   "exp6": "validation_exp6_fault_summary.csv.gz",
                   "partitions": "validation_policy_partition_index.csv",
                   "temporal_manifest": "validation_temporal_policy_manifest.csv.gz"},
    "test": {"exp2": "test_exp2_representation_frontier.csv.gz", "exp3": "test_exp3_static_summary.csv.gz",
             "exp6": "test_receiver_robustness_summary.csv.gz",
             "partitions": "test_policy_partition_index.csv",
             "temporal_manifest": "test_temporal_policy_manifest.csv.gz"},
}[PHASE]
OUT_PARENT = Path(sys.argv[3]) if len(sys.argv) > 3 else None
OUT = (OUT_PARENT / "semuav_figdata") if OUT_PARENT else Path.home() / f"semuav_figdata_{PHASE}"
SMALL = 20e6
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

def header(p):
    opener = gzip.open if str(p).endswith(".gz") else open
    with opener(p, "rt", newline="") as h:
        return next(csv.reader(h))

def as_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    m = s.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    return pd.to_numeric(s, errors="coerce").fillna(m)

columns = {}
# 1) small result files, verbatim
for p in sorted(RES.rglob("*")):
    if p.is_file() and p.stat().st_size < SMALL:
        dst = OUT / "results" / p.relative_to(RES)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        if p.suffix in {".csv", ".gz"} and ".csv" in p.name:
            columns[str(p.relative_to(RES))] = header(p)
if PHASE == "test":
    aliases = {NAMES["exp3"]: "test_exp3_summary.csv.gz", NAMES["exp6"]: "test_exp6_fault_summary.csv.gz",
               "validation_selection.json": "test_selection.json"}
    for source, alias in aliases.items():
        if (RES / source).is_file():
            shutil.copy2(RES / source, OUT / "results" / alias)
print("copied small result files")

# 2) Experiment 2 static frontier -> per (stream, sample) means
src = RES / NAMES["exp2"]
hdr = header(src); columns[src.name] = hdr
keys = [c for c in ("mode", "variant", "source_model", "sample_id") if c in hdr]
cand = [c for c in hdr if c not in keys and any(k in c for k in ("bytes", "correct", "latency", "energy", "_ms", "_j"))]
probe = pd.read_csv(src, usecols=keys + cand, nrows=5000, low_memory=False)
vals = [c for c in cand if as_num(probe[c]).notna().mean() > 0.5]
acc = None
for i, ch in enumerate(pd.read_csv(src, usecols=keys + vals, chunksize=400_000, low_memory=False)):
    for c in vals:
        ch[c] = as_num(ch[c])
    g = ch.groupby(keys, dropna=False)[vals]
    part = pd.concat({"sum": g.sum(min_count=1), "n": g.count()}, axis=1)
    acc = part if acc is None else acc.add(part, fill_value=0)
    if i % 10 == 0:
        print(f"  exp2 frontier chunk {i}", flush=True)
means = acc["sum"] / acc["n"]
means.columns = [f"mean_{c}" for c in means.columns]
means["rows"] = acc["n"].max(axis=1)
means.reset_index().to_csv(OUT / "exp2_frontier_by_sample.csv.gz", index=False)
print("exp2 aggregated:", len(means), "stream-sample cells")

# 3) Experiment 4/5 trial partitions -> cell table, controller diagnostics, one showcase episode
index = pd.read_csv(RES / NAMES["partitions"])
parts = [Path(str(p)) for p in index["artifact_path"]]
missing = [p for p in parts if not p.is_file()]
if missing:
    raise SystemExit(f"{len(missing)} {PHASE} policy partitions are missing, e.g. {missing[0]}")
print(f"{PHASE} policy partitions:", len(parts))
phdr = header(parts[0]); columns["04_policy_partition (example)"] = phdr
cell_keys = [c for c in ("policy", "trace_id", "visual_group_id", "mission_profile", "replicate_seed") if c in phdr]
metric_cols = [c for c in ("correct_timely_action", "correct_action", "delivered_by_deadline", "critical",
                           "critical_miss", "false_alert", "total_wire_bytes", "energy_j",
                           "full_message_received") if c in phdr]
diag_cols = [c for c in phdr if c in ("calibrated_correctness_probability", "decision_risk",
             "decision_uncertainty", "decision_mission_priority", "decision_remaining_deadline_ms",
             "decision_context_available", "evidence_escalated")]
ep_keys = [c for c in ("trace_id", "replicate_seed", "episode_id") if c in phdr]
cells, modes, diag, switches = [], [], [], {}
for f in parts:
    use = list(dict.fromkeys(cell_keys + metric_cols + diag_cols + ep_keys + ["selected_mode", "decision_us"]))
    d = pd.read_csv(f, usecols=[c for c in use if c in phdr], low_memory=False)
    for c in metric_cols + diag_cols:
        d[c] = as_num(d[c])
    g = d.groupby(cell_keys, dropna=False)
    cells.append(pd.concat([g[metric_cols].sum(min_count=1), g.size().rename("decisions")], axis=1).reset_index())
    modes.append(d.groupby(cell_keys + ["selected_mode"], dropna=False).size().rename("n").reset_index())
    if diag_cols:
        diag.append(d[["policy"] + diag_cols].sample(min(len(d), 3000), random_state=0))
    pf = d[d["policy"].astype(str).eq("proposed_full")].sort_values(ep_keys + ["decision_us"])
    for k, e in pf.groupby(ep_keys):
        switches[k] = int((e["selected_mode"].astype(str) != e["selected_mode"].astype(str).shift()).sum() - 1)
pd.concat(cells).groupby(cell_keys, dropna=False).sum(min_count=1).reset_index().to_csv(OUT / "exp4_cells.csv.gz", index=False)
pd.concat(modes).groupby(cell_keys + ["selected_mode"], dropna=False)["n"].sum().reset_index().to_csv(OUT / "exp4_mode_counts.csv.gz", index=False)
if diag:
    pd.concat(diag).to_csv(OUT / "exp4_decision_signal_sample.csv.gz", index=False)
best = max(switches, key=switches.get)
print("showcase episode", dict(zip(ep_keys, best)), "proposed_full switches:", switches[best])
episode = []
for f in parts:
    d = pd.read_csv(f, low_memory=False)
    m = np.logical_and.reduce([d[k].astype(str).eq(str(v)) for k, v in zip(ep_keys, best)])
    if m.any():
        episode.append(d[m])
episode = pd.concat(episode)
episode.to_csv(OUT / "showcase_episode_all_policies.csv.gz", index=False)

# 4) showcase frames and link trace
manifest_path = RES / NAMES["temporal_manifest"]
if not manifest_path.is_file():
    found = sorted(RES.rglob(f"{PHASE}*temporal*sample*.csv.gz")) + sorted(RES.rglob("test_registered_temporal_sample/*.csv.gz"))
    if not found:
        raise SystemExit(f"no {PHASE} temporal manifest found under {RES}")
    manifest_path = found[0]
print("temporal manifest:", manifest_path.name)
man = pd.read_csv(manifest_path, low_memory=False)
if "temporal_observation_index" not in man.columns:
    order = [c for c in ("sequence_id", "frame_index", "capture_time_us") if c in man.columns]
    man = man.sort_values(order)
    man["temporal_observation_index"] = man.groupby(order[0]).cumcount() if order else range(len(man))
ids = set(episode["sample_id"].astype(str))
sel = man[man["sample_id"].astype(str).isin(ids)]
sel.to_csv(OUT / "showcase_manifest_rows.csv", index=False)
img_cols = [c for c in man.columns if "image_path" in c or c.endswith("_path")]
(OUT / "showcase_frames").mkdir()
try:
    from PIL import Image
    for _, row in sel.iterrows():
        for c in img_cols:
            p = Path(str(row[c]))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and p.is_file():
                with Image.open(p) as im:
                    im = im.convert("RGB"); im.thumbnail((960, 960))
                    im.save(OUT / "showcase_frames" / f"{row['sample_id']}.jpg", quality=88)
                break
except Exception as exc:
    print("frame copy skipped:", exc)
tm = pd.read_csv(RES / "registered_trace_manifest.csv", low_memory=False)
trace_id = str(dict(zip(ep_keys, best)).get("trace_id"))
rows = tm[tm.astype(str).apply(lambda r: trace_id in " ".join(r.values), axis=1)]
for c in tm.columns:
    for v in rows[c].astype(str):
        p = Path(v)
        if p.is_file() and p.stat().st_size < SMALL:
            shutil.copy2(p, OUT / f"showcase_trace_{p.name}")

(OUT / "COLUMNS.json").write_text(json.dumps(columns, indent=1))
if OUT_PARENT is not None:
    print(f"DONE -> {OUT}, frames: {len(list((OUT/'showcase_frames').glob('*.jpg')))}")
    raise SystemExit(0)
with tarfile.open(Path.home() / f"semuav_figdata_{PHASE}.tar.gz", "w:gz") as t:
    t.add(OUT, arcname="semuav_figdata")
size = (Path.home() / f"semuav_figdata_{PHASE}.tar.gz").stat().st_size / 1e6
print(f"DONE -> ~/semuav_figdata_{PHASE}.tar.gz ({size:.0f} MB), frames: {len(list((OUT/'showcase_frames').glob('*.jpg')))}")

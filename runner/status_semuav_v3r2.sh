#!/usr/bin/env bash
# Show v3r2 progress: latest log lines, stage counts and v3r1 reuse. Read-only; safe at any time.
OUT="$HOME/semuav_v3r2_outputs"
RUNS='/data/Lane Detection Datasets/musta/semuav_vlm/.cache/visdrone_paper_v3r2/runs'
echo "== $(date) =="
LOG="$(ls -t "$OUT"/*.log 2>/dev/null | head -1)"
if [ -n "$LOG" ]; then echo "-- latest log: $LOG"; grep -E '^\[|START|FINISHED|FAILED|PAPER_FIGURES|V3R1_|V3R2_|PASS|STOP' "$LOG" | tail -n 15; fi
for RUN in "$RUNS"/semuav_visdrone_v3r2-full-*; do
    [ -f "$RUN/status.sqlite" ] || continue
    echo "-- run: $(basename "$RUN")"
    /usr/bin/python3 -I -B - "$RUN/status.sqlite" <<'PY'
import sqlite3, sys
c = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
rows = c.execute("SELECT name, status, message, ended_utc FROM stages").fetchall()
fam = {}
for name, status, message, _ in rows:
    key = name.split("_shard_")[0].split("_policy_partition_")[0][:40]
    f = fam.setdefault(key, {"COMPLETE": 0, "RUNNING": 0, "other": 0, "reused": 0})
    f[status if status in ("COMPLETE", "RUNNING") else "other"] += 1
    f["reused"] += bool(message and str(message).startswith("IMPORTED_V3R1"))
for key, f in sorted(fam.items()):
    print(f"  {key:42s} complete={f['COMPLETE']:4d} running={f['RUNNING']} reused_from_v3r1={f['reused']}")
last = sorted(rows, key=lambda r: r[3] or "")[-3:]
print("  latest:", [(r[0], r[1], r[3]) for r in last])
PY
done
pgrep -f run_notebook_v3r2.py >/dev/null && echo "runner: RUNNING" || echo "runner: not running"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/GPU /'

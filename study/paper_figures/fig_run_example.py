import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # run with python -I
import sys, ast
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image
from common import *

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); style()
ep = pd.read_csv(DATA / "showcase_episode_all_policies.csv.gz", low_memory=False)
man = pd.read_csv(DATA / "showcase_manifest_rows.csv", low_memory=False).sort_values("temporal_observation_index")
ep["ok"] = as_num(ep.correct_timely_action).astype(bool)
ep["t"] = ep.capture_us / 1e6
ROWS = [("heuristic_lookahead_reference", "Lookahead ref.*"), ("best_fixed_validation", best_fixed_info(DATA)["short"]),
        ("proposed_full", "Proposed (full)"), ("fixed_l5_image_or_clip", "Fixed L5 image")]
pf = ep[ep.policy.eq("proposed_full")].sort_values("t")
dt = float(np.median(np.diff(pf.t)))
T_END = pf.t.max() + dt

fig = plt.figure(figsize=(DCOL, 3.75))
gs = GridSpec(3, 6, figure=fig, height_ratios=[0.78, 0.62, 1.55], hspace=0.38, wspace=0.05)
pick = np.linspace(0, len(man) - 1, 6).round().astype(int)
for k, i in enumerate(pick):
    ax = fig.add_subplot(gs[0, k]); row = man.iloc[i]
    f = DATA / "showcase_frames" / f"{row.sample_id}.jpg"
    im = Image.open(f); W, H = im.size
    ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_color(GHOST); sp.set_linewidth(0.5)
    x0, y0, x1, y1 = ast.literal_eval(row.registered_roi_xyxy)
    ax.add_patch(Rectangle((x0 * W, y0 * H), (x1 - x0) * W, (y1 - y0) * H, fill=False, ec="white", lw=0.8, ls=(0, (3, 2))))
    a, b, c, d = ast.literal_eval(row.evidence_roi)
    ax.add_patch(Rectangle((a * W, b * H), (c - a) * W, (d - b) * H, fill=False, ec="#eda100", lw=1.1))
    ax.set_title(f"t = {row.temporal_observation_index * dt:.1f} s", fontsize=6.5, pad=2, color=INK2)
    if k == 0:
        ax.text(-0.04, 1.0, "(a)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")

ax_l = fig.add_subplot(gs[1, :])
ax_l.step(pf.t, pf.decision_estimated_service_rate_bps / 1e6, where="post", color=INK, lw=0.9)
ax_l.set_yscale("log"); ax_l.set_xlim(0, T_END); ax_l.set_ylabel("Est. uplink\n(Mb/s)", fontsize=7)
ax_l.tick_params(labelbottom=False); ax_l.text(-0.02, 1.0, "(b)", transform=ax_l.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")
ax_l.set_title("Dashed: registered priority ROI; yellow: gold evidence object.  Below: sender's causal link estimate (registered steady-mid trace); gold action = RESPOND in all 64 frames",
               fontsize=6.5, color=INK2, loc="left", pad=2)

ax = fig.add_subplot(gs[2, :], sharex=ax_l)
for r, (pol, name) in enumerate(ROWS):
    d = ep[ep.policy.eq(pol)].sort_values("t"); y = len(ROWS) - 1 - r
    for t, mode, ok in zip(d.t, d.selected_mode, d.ok):
        ax.add_patch(Rectangle((t, y - 0.32), dt * 0.9, 0.5, color=MODE_COLOR.get(mode, GHOST), lw=0))
        if not ok:
            ax.plot(t + dt * 0.45, y - 0.43, marker="x", ms=3.2, mew=0.9, color="#e34948")
    kb = d.total_wire_bytes.mean() / 1e3
    ax.text(T_END + 0.15, y - 0.07, f"{d.ok.mean():.0%} correct & timely\n{kb:.1f} kB/decision", fontsize=6.3,
            va="center", color=PROPOSED if pol == "proposed_full" else INK2)
ax.set_yticks(range(len(ROWS)), [n for _, n in ROWS][::-1]); ax.set_ylim(-0.75, len(ROWS) - 0.4)
ax.set_xlabel("Capture time (s), 5 Hz"); ax.grid(axis="y", visible=False)
ax.text(-0.02, 1.0, "(c)", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")
used = [m for m in MODE_ORDER if m in set(ep[ep.policy.isin([p for p, _ in ROWS])].selected_mode)]
handles = [Rectangle((0, 0), 1, 1, color=MODE_COLOR[m]) for m in used]
handles.append(Line2D([], [], marker="x", ls="", color="#e34948", mew=0.9, ms=4))
ax.legend(handles, [MODE_LABEL[m] for m in used] + ["not correct & timely"], loc="upper left",
          bbox_to_anchor=(0, -0.28), ncol=len(handles), fontsize=6.3, handlelength=1.0, columnspacing=1.0)
fig.subplots_adjust(right=0.86)
save(fig, OUT, "fig7_run_example")
status = ep[ep.policy.eq("proposed_full") & ~ep.ok].apply_status.value_counts()
print("proposed failures by status:", status.to_dict())

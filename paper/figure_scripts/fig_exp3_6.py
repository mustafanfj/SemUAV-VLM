import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # run with python -I
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, FuncFormatter
from common import *
from stats import balanced_timely

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); PHASE = sys.argv[3] if len(sys.argv) > 3 else "validation"
style(); R = DATA / "results"

# ---------------- Fig. 3: transport surface (Experiment 3) ----------------
e3 = pd.read_csv(R / f"{PHASE}_exp3_summary.csv.gz")
e3 = e3[~e3.variant.str.endswith("__detector_only")]
spec = json.load(open(R / f"{PHASE}_policy_spec.json"))
REP = {f["mode"]: f["variant"] for f in spec["fixed"]}          # the registered fixed variant of each level
for _mode in MODE_ORDER:
    if not e3.variant.eq(REP.get(_mode, "")).any():
        REP[_mode] = e3[e3["mode"].eq(_mode)].groupby("variant").mean_wire_bytes.mean().idxmin()
def wmean(d, col):
    return (d[col] * d.observation_rows).sum() / d.observation_rows.sum()
# Message size = mean wire bytes where nothing is refused (fastest rate, largest buffer, clean link).
_free = e3[e3.loss.eq(0) & e3.jitter_ms.eq(0) & e3.rate_mbps.eq(e3.rate_mbps.max()) & e3.queue_kb.eq(e3.queue_kb.max())]
SIZE_KB = {m: float(_free[_free.variant.eq(v)].mean_wire_bytes.mean()) / 1e3 for m, v in REP.items()}
def vname(v):
    m = {"l0_cbor": "CBOR", "l1_cbor": "CBOR", "l2_cbor": "CBOR", "cbor+jpeg_q50_roi": "CBOR+JPEG ROI",
         "registered_feature_int4": "INT4", "registered_feature_int8": "INT8"}
    if v in m:
        return m[v]
    c, q, sc = v.split("_")
    return f"{c.upper() if c != 'webp' else 'WebP'} q{q[1:]} ×{sc[1:]}"
fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.7), sharey=True)
clean = e3[e3.loss.eq(0) & e3.jitter_ms.eq(0)]
curves = {}
for mode in MODE_ORDER:
    d = clean[clean.variant.eq(REP[mode])]
    s = d.groupby("rate_mbps").apply(lambda g: wmean(g, "deadline_success_probability"))
    kb = SIZE_KB[mode]
    curves[mode] = (s, kb)
    size = f"{kb:.1f} kB" if kb < 10 else f"{kb:.0f} kB"
    a.plot(s.index, s.values, marker="o", ms=3.2, color=MODE_COLOR[mode],
           lw=1.6 if mode in ("L1_ALERT",) else 1.1, zorder=4 if mode == "L1_ALERT" else 3,
           label=f"{MODE_LABEL[mode]} ({vname(REP[mode])}, {size})")
a.set_xscale("log"); a.set_xticks([0.1, 0.25, 0.5, 1, 2, 5, 10, 20], ["0.1", "0.25", "0.5", "1", "2", "5", "10", "20"])
a.set_xlim(0.08, 25)
a.set_xlabel("Uplink rate (Mb/s), loss-free link"); a.set_ylabel("Deadline success probability")
a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); a.set_ylim(0, 1.04); panel(a, "a")
_h3, _l3 = a.get_legend_handles_labels()
_buf = float(clean.queue_kb.min())
_small = {m: v for m, v in curves.items() if v[1] < _buf}
_sem = [m for m in _small if m != "L5_IMAGE_OR_CLIP"]
if _sem:
    _ok = sorted(set.intersection(*[set(curves[m][0].index[curves[m][0] >= 0.95]) for m in _sem]))
    _l1 = curves["L1_ALERT"][0]
    txt = f"Semantic payloads (≤ {max(curves[m][1] for m in _sem):.0f} kB): ≥ 95% on time from {_ok[0]:g} Mb/s" if _ok else ""
    if (_l1 >= 0.995).all():
        txt += f"\nL1 alert: {_l1.min():.0%} on time at every rate"
    txt = txt.replace(": ≥", ":\n≥").replace("\nL1 alert:", ";\nL1 alert:").replace(" on time at every rate", " at every rate")
    a.text(2.2, 0.215, txt, fontsize=6.1, color=INK, va="top", linespacing=1.15)
_big = [m for m, v in curves.items() if v[1] > _buf]
if _big:
    m4 = "L4_FEATURES" if "L4_FEATURES" in _big else _big[0]
    s4 = curves[m4][0]
    names = " and ".join(MODE_LABEL[m].split()[0] for m in _big)
    a.annotate(f"{names} messages larger than the\nsmallest send buffer ({_buf:.0f} kB) are\nrefused there: "
               f"{MODE_LABEL[m4].split()[0]} plateaus\nat {s4.max():.0%} even at 20 Mb/s",
               xy=(10.0, float(s4.loc[10.0])), xytext=(2.2, 0.50), fontsize=6.1, color=INK, va="top", linespacing=1.15,
               arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, shrinkB=4))
one = e3[e3.rate_mbps.eq(1.0) & e3.jitter_ms.eq(0)]
for mode in MODE_ORDER:
    d = one[one.variant.eq(REP[mode])]
    for lm, ls in (("iid", "-"), ("gilbert_elliott", (0, (3, 2)))):
        s = d[d.loss_model.eq(lm)].groupby("loss").apply(lambda g: wmean(g, "deadline_success_probability"))
        b.plot(s.index * 100, s.values, ls=ls, marker="o", ms=2.6, color=MODE_COLOR[mode],
               lw=1.5 if mode == "L1_ALERT" else 1.0, zorder=4 if mode == "L1_ALERT" else 3)
b.plot([], [], color=INK2, ls="-", label="i.i.d. loss"); b.plot([], [], color=INK2, ls=(0, (3, 2)), label="Gilbert–Elliott bursts")
b.legend(loc="lower left", fontsize=6.3); b.set_xlabel("Packet-loss rate at 1 Mb/s (%)"); b.set_xticks([0, 1, 5, 15]); panel(b, "b")
_l1b = one[one.variant.eq(REP["L1_ALERT"]) & one.loss.eq(one.loss.max())]
_iid = wmean(_l1b[_l1b.loss_model.eq("iid")], "deadline_success_probability")
_ge = wmean(_l1b[_l1b.loss_model.eq("gilbert_elliott")], "deadline_success_probability")
b.text(15, 1.03, f"L1 at {one.loss.max():.0%} loss: {_iid:.0%} (i.i.d.), {_ge:.0%} (bursts)",
       fontsize=6.3, color=MODE_COLOR["L1_ALERT"], ha="right", va="bottom")
fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.87))
fig.legend(_h3, _l3, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=6.3, columnspacing=1.2,
           handlelength=1.4, handletextpad=0.3)
save(fig, OUT, "fig3_transport_surface")
pd.DataFrame([{"mode": m, "variant": REP[m], "message_kb": SIZE_KB[m], **{f"success_at_{r:g}Mbps": v for r, v in curves[m][0].items()}}
              for m in MODE_ORDER]).to_csv(OUT / "table_exp3_rate.csv", index=False)

# ---------------- Fig. 4: controller benchmark (Experiments 4/5) ----------------
cells = pd.read_csv(DATA / "exp4_cells.csv.gz")
policies = sorted(cells.policy.unique())
res = {p: balanced_timely(cells, p) for p in policies}
T = pd.DataFrame({p: {f"{k}_{s}": v[i] for k, v in r.items() for i, s in enumerate(("pt", "lo", "hi"))}
                  for p, r in res.items()}).T
T.index.name = "policy"; T.to_csv(OUT / "table_exp4_5.csv")
mc = pd.read_csv(DATA / "exp4_mode_counts.csv.gz")
mixall = mc.pivot_table(index="policy", columns="selected_mode", values="n", aggfunc="sum").fillna(0)
mixall = mixall.div(mixall.sum(axis=1), axis=0).round(6)
# Behavioural duplicates: same outcome metrics AND same representation mix.
key = T[["balanced_pt", "kb_pt", "energy_pt", "miss_pt"]].round(6).astype(str).agg("|".join, axis=1) + "|" + \
      mixall.reindex(T.index).astype(str).agg("|".join, axis=1)
dup_groups = T.assign(_k=key).reset_index().groupby("_k").policy.apply(list)
BF = best_fixed_info(DATA)
NAME = {"proposed_full": "Proposed controller", "best_fixed_validation": BF["label"],
        "heuristic_lookahead_reference": "Lookahead reference*", "fixed_l5_image_or_clip": "Fixed L5 image",
        "fixed_l3_semantics_roi": "Fixed L3", "fixed_l2_semantics": "Fixed L2", "fixed_l0_heartbeat": "Fixed L0",
        "fixed_l4_features": "Fixed L4 features", "channel_only_hysteresis": "Channel-only", "channel_plus_mission": "Channel+mission",
        "queue_deadline_greedy": "Queue/deadline greedy", "risk_aware_no_uncertainty": "Risk-aware, no uncert.",
        "mission_only": "Mission-only", "adaptive_image_codec": "Adaptive image codec", "adaptive_feature_rate": "Adaptive feature rate",
        "ablate_receiver_state": "w/o receiver state", "ablate_mission_risk": "w/o mission risk",
        "ablate_evidence_escalation": "w/o evidence escal.", "ablate_hysteresis": "w/o hysteresis", "always_vlm_full": "always-VLM",
        "adaptive_semantic_rate": "Adaptive semantic rate", "fixed_l1_alert": "Fixed L1", "detector_only_semantics": "Detector-only semantics"}
PREF = ("proposed_full", "heuristic_lookahead_reference", "fixed_l2_semantics", "fixed_l4_features", "fixed_l5_image_or_clip",
        "fixed_l3_semantics_roi", "fixed_l0_heartbeat", "mission_only", "adaptive_image_codec")
rep_of = {}
for grp in dup_groups:
    pick = next((p for p in PREF if p in grp), grp[0])
    rep_of[pick] = grp
twin = len(rep_of.get("proposed_full", [])) - 1
SHORT = dict(NAME); SHORT["proposed_full"] = f"Proposed (= best fixed {BF['short'].split('(')[1].rstrip(')')})" if "best_fixed_validation" in rep_of.get("proposed_full", []) else "Proposed"
SHORT["fixed_l2_semantics"] = "Fixed L2"; SHORT["fixed_l4_features"] = "Fixed L4"
pd.DataFrame([{"representative": p, "identical_policies": ";".join(sorted(g))} for p, g in rep_of.items()]).to_csv(
    OUT / "table_exp4_identical_groups.csv", index=False)
fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.9), gridspec_kw={"width_ratios": [1.55, 1]})
LAB = {"proposed_full": (0, 17, "center"), "heuristic_lookahead_reference": (-9, 8, "right"), "fixed_l2_semantics": (6, -9, "left"),
       "fixed_l5_image_or_clip": (6, 10, "left"), "adaptive_image_codec": (7, -1, "left"), "fixed_l4_features": (-6, -12, "right"),
       "mission_only": (-6, 7, "right"), "fixed_l3_semantics_roi": (5, -10, "left"), "fixed_l0_heartbeat": (0, -12, "center")}
for p, grp in rep_of.items():
    r = T.loc[p]
    if r.kb_pt <= 0:
        continue
    hl = p in ("proposed_full", "heuristic_lookahead_reference")
    col = PROPOSED if p == "proposed_full" else (INK if hl else GHOST)
    a.plot([r.kb_pt, r.kb_pt], [r.balanced_lo, r.balanced_hi], color=col, lw=1.0 if hl else 0.8, zorder=2)
    a.plot([r.kb_lo, r.kb_hi], [r.balanced_pt] * 2, color=col, lw=1.0 if hl else 0.8, zorder=2)
    a.scatter(r.kb_pt, r.balanced_pt, s=46 if hl else 20, color=col, edgecolor="white", lw=0.6, zorder=6 if hl else 4,
              marker="D" if p == "heuristic_lookahead_reference" else "o")
    if p in LAB:
        dx, dy, ha = LAB[p]
        a.annotate(SHORT.get(p, p), (r.kb_pt, r.balanced_pt), xytext=(dx, dy), textcoords="offset points", fontsize=6.4,
                   color=PROPOSED if p == "proposed_full" else (INK if hl else INK2), ha=ha, va="center",
                   fontweight="bold" if p == "proposed_full" else "normal")
_p, _h, _l5 = T.loc["proposed_full"], T.loc["heuristic_lookahead_reference"], T.loc["fixed_l5_image_or_clip"]
if _h.balanced_pt - _p.balanced_pt >= 0.02:
    a.annotate("", xy=(4.4, _h.balanced_pt), xytext=(4.4, _p.balanced_pt), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    a.text(4.7, (_h.balanced_pt + _p.balanced_pt) / 2, f"+{100*(_h.balanced_pt-_p.balanced_pt):.0f} pts with one-step\nlookahead (non-deployable)",
           fontsize=6.3, va="center", color=INK)
a.annotate(f"+{100*(_p.balanced_pt-_l5.balanced_pt):.0f} pts over fixed L5 image\nat {_l5.kb_pt/_p.kb_pt:.0f}× fewer bytes, "
           f"{_l5.energy_pt/_p.energy_pt:.1f}× less energy",
           xy=(_p.kb_pt * 1.05, _p.balanced_pt - 0.01), xytext=(0.95, 0.075), fontsize=6.4, color=PROPOSED,
           arrowprops=dict(arrowstyle="-|>", color=PROPOSED, lw=0.7, shrinkB=5, connectionstyle="arc3,rad=-0.2"))
_l0 = T.loc["fixed_l0_heartbeat"].balanced_pt
a.axhline(_l0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
a.text(95, _l0 + 0.015, "never-alert floor (fixed L0)", fontsize=6.1, color=MUTED, ha="right")
a.set_xscale("log"); a.set_xlim(0.6, 110); a.set_ylim(0, 0.95)
a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a.set_xlabel("Wire kB per decision (log)"); a.set_ylabel("Balanced correct-and-timely rate"); panel(a, "a")

show = ["heuristic_lookahead_reference", "proposed_full", "mission_only", "fixed_l4_features", "fixed_l5_image_or_clip"]
mix = mixall.loc[show[::-1]]
left = np.zeros(len(mix))
for mode in MODE_ORDER:
    if mode in mix and mix[mode].sum() > 0:
        v = mix[mode].to_numpy()
        b.barh(range(len(mix)), v, left=left, color=MODE_COLOR[mode], edgecolor="white", lw=0.8, height=0.62,
               label=MODE_LABEL[mode])
        for i, (x0, w) in enumerate(zip(left, v)):
            if w > 0.12:
                b.text(x0 + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=6, color="white")
        left += v
b.set_yticks(range(len(mix)), [SHORT.get(p, p) for p in mix.index]); b.set_xlim(0, 1)
b.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); b.set_xlabel("Share of decisions by payload level")
b.grid(axis="y", visible=False); panel(b, "b")
b.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.04), ncol=3, fontsize=6.1, borderaxespad=0, columnspacing=0.6,
         handlelength=1.0, handletextpad=0.3)
fig.tight_layout(w_pad=1.0); save(fig, OUT, "fig4_controller_benchmark")

# ---------------- Fig. 5: safety endpoints ----------------
order = T.loc[[p for p in rep_of if T.loc[p].kb_pt > 0]].sort_values("miss_pt").index
fig, a = plt.subplots(figsize=(COL, 2.75))
y = np.arange(len(order))[::-1]
for yi, p in zip(y, order):
    r = T.loc[p]
    col = PROPOSED if p == "proposed_full" else INK2
    a.plot([r.miss_lo, r.miss_hi], [yi, yi], color=col, lw=1.0 if p == "proposed_full" else 0.9)
    a.scatter(r.miss_pt, yi, s=26 if p == "proposed_full" else 20, color=col, zorder=3,
              marker="D" if p == "heuristic_lookahead_reference" else "o")
    a.text(1.03, yi, f"FA {r.fa_pt:.0%}", fontsize=6, color=INK2 if r.fa_pt > 0.001 else MUTED, va="center",
           transform=a.get_yaxis_transform())
a.text(1.03, y[0] + 0.45, "False-alert\nrate", fontsize=5.8, color=MUTED, va="bottom", transform=a.get_yaxis_transform())
dep = [p for p in order if p != "heuristic_lookahead_reference"]
if dep and dep[0] == "proposed_full":
    yp = y[list(order).index("proposed_full")]
    a.text(0.02, yp - 0.5, "lowest miss rate of all deployable policies, 0 false alerts",
           fontsize=5.9, color=PROPOSED, va="center")
a.set_yticks(y, [SHORT.get(p, p) for p in order]); a.set_xlim(0, 1)
a.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a.set_xlabel("Critical-miss rate on critical episodes (95% CI)"); a.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, OUT, "fig5_safety")

# ---------------- Fig. 6: overconfidence (why confidence cannot drive the controller) ----------------
sel = json.load(open(R / f"{PHASE}_selection.json"))
gate = pd.DataFrame(sel["calibration_gate"]).set_index("model_id")
det = pd.read_csv(R / f"{PHASE}_exp1_perception_detail.csv.gz", low_memory=False,
                  usecols=["model_id", "split", "confidence", "action_correct"])
for c in ("confidence", "action_correct"):
    det[c] = pd.to_numeric(det[c].replace({"True": 1, "False": 0}), errors="coerce")
cal = det[det.split.eq("validation")].groupby("model_id").agg(conf=("confidence", "mean"), acc=("action_correct", "mean"))
MODELS = ["detector_only", "qwen3_vl_2b", "smolvlm2_2b", "qwen3_vl_8b_upper_bound"]
ML = ["Detector", "Qwen3-VL-2B", "SmolVLM2-2.2B", "Qwen3-VL-8B"]
fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.0), gridspec_kw={"width_ratios": [1.2, 1]})
yy = np.arange(len(MODELS))[::-1]
for yi, m in zip(yy, MODELS):
    r = cal.loc[m]
    a.plot([r.acc, r.conf], [yi, yi], color=GHOST, lw=3.2, solid_capstyle="round", zorder=1)
    a.scatter(r.acc, yi, s=30, color=INK, zorder=3, label="Action accuracy" if yi == yy[0] else None)
    a.scatter(r.conf, yi, s=30, facecolor="white", edgecolor=INK, lw=1.0, zorder=3, label="Mean stated confidence" if yi == yy[0] else None)
    a.text((r.acc + r.conf) / 2, yi + 0.28, f"+{100*(r.conf-r.acc):.0f} pts", ha="center", fontsize=6.3, color="#c0392b")
a.set_yticks(yy, ML); a.set_xlim(0, 1.02); a.set_ylim(-0.6, len(MODELS) - 0.2)
a.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); a.set_xlabel("Rate on the VisDrone validation set (126 frames per model)")
a.grid(axis="y", visible=False); panel(a, "a")
a.legend(loc="lower left", bbox_to_anchor=(0, 1.03), ncol=2, fontsize=6.3, borderaxespad=0, handletextpad=0.3)
for yi, m in zip(yy, MODELS):
    r = gate.loc[m]
    b.plot([r.ece, r.ece_group_bootstrap_upper], [yi, yi], color=INK2, lw=0.9)
    b.scatter(r.ece, yi, s=24, color=INK, zorder=3)
    t = sel["calibration"][m]["temperature"]
    b.text(0.57, yi, f"T = {t:.1f} (fit bound)" if abs(np.log(t) - 3) < 1e-3 else f"T = {t:.2f}", fontsize=6.2, va="center", color=INK2)
for thr, lab in ((0.05, "0.05"), (0.10, "0.10 (priority)")):
    b.axvline(thr, color=MUTED, lw=0.7, ls=(0, (3, 2)))
b.text(0.108, yy[-1] - 0.05, "registered gates:\n0.05 (all), 0.10 (priority)", fontsize=5.8, color=MUTED, va="center")
b.set_yticks(yy, []); b.set_xlim(0, 0.8); b.set_ylim(-0.6, len(MODELS) - 0.2)
b.set_xlabel("ECE after temperature scaling (point, 95% upper)"); b.grid(axis="y", visible=False); panel(b, "b")
fig.tight_layout(w_pad=1.0); save(fig, OUT, "fig6_calibration_collapse")
cal.join(gate[["ece", "ece_group_bootstrap_upper"]]).to_csv(OUT / "table_calibration.csv")

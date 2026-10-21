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
REP = {"L0_HEARTBEAT": "l0_cbor", "L1_ALERT": "l1_cbor", "L2_SEMANTICS": "l2_cbor",
       "L3_SEMANTICS_ROI": "cbor+jpeg_q50_roi", "L4_FEATURES": "registered_feature_int4"}
l5 = e3[e3["mode"].eq("L5_IMAGE_OR_CLIP") & ~e3.variant.str.contains("lossless")].groupby("variant").mean_wire_bytes.mean()
REP["L5_IMAGE_OR_CLIP"] = l5.idxmin()          # the lightest registered image variant
for _mode in MODE_ORDER:                       # fall back to the lightest variant if a name is absent
    if not e3.variant.eq(REP[_mode]).any():
        REP[_mode] = e3[e3["mode"].eq(_mode)].groupby("variant").mean_wire_bytes.mean().idxmin()
def wmean(d, col):
    return (d[col] * d.observation_rows).sum() / d.observation_rows.sum()
fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.3), sharey=True)
clean = e3[e3.loss.eq(0) & e3.jitter_ms.eq(0)]
curves = {}
for mode in MODE_ORDER:
    d = clean[clean.variant.eq(REP[mode])]
    s = d.groupby("rate_mbps").apply(lambda g: wmean(g, "deadline_success_probability"))
    kb = d.mean_wire_bytes.mean() / 1e3
    curves[mode] = (s, kb)
    a.plot(s.index, s.values, marker="o", ms=3.2, color=MODE_COLOR[mode],
           label=f"{MODE_LABEL[mode]} ({kb:.1f} kB)" if kb < 10 else f"{MODE_LABEL[mode]} ({kb:.0f} kB)")
a.set_xscale("log"); a.set_xticks([0.1, 0.25, 0.5, 1, 2, 5, 10, 20], ["0.1", "0.25", "0.5", "1", "2", "5", "10", "20"])
a.set_xlim(0.08, 25)
a.set_xlabel("Uplink rate (Mb/s), loss-free link"); a.set_ylabel("Deadline success probability")
a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); a.set_ylim(0, 1.03); panel(a, "a")
a.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, fontsize=6.3, borderaxespad=0, columnspacing=0.8,
         handlelength=1.4, handletextpad=0.3)
# Data-driven callouts: each is drawn only when the data support it.
_min_buffer_kb = float(clean.queue_kb.min()) if "queue_kb" in clean else 32.0
for _mode, (_s, _kb) in curves.items():
    if _kb > _min_buffer_kb and len(_s) and _s.max() < 0.9:
        _x = 5.0 if 5.0 in _s.index else _s.index[len(_s) // 2]
        a.annotate(f"{MODE_LABEL[_mode].split()[0]} ({_kb:.0f} kB) is capped by send-buffer\nadmission "
                   f"({_min_buffer_kb:.0f} kB buffer), not link rate",
                   xy=(_x, float(_s.loc[_x])), xytext=(0.105, 0.36), fontsize=6.6, color=INK,
                   arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, shrinkB=4))
        break
_small = [(_s, _kb) for _s, _kb in curves.values() if _kb < 10]
if _small:
    _rates = sorted(set.intersection(*[set(_s.index[_s >= 0.95]) for _s, _ in _small]))
    if _rates:
        a.text(0.27, 0.86, f"≤ {max(_kb for _, _kb in _small):.0f} kB payloads: ≥ 95% on time from {_rates[0]:g} Mb/s",
               fontsize=6.6, color=INK)
one = e3[e3.rate_mbps.eq(1.0) & e3.jitter_ms.eq(0)]
for mode in MODE_ORDER:
    d = one[one.variant.eq(REP[mode])]
    for lm, ls in (("iid", "-"), ("gilbert_elliott", (0, (3, 2)))):
        s = d[d.loss_model.eq(lm)].groupby("loss").apply(lambda g: wmean(g, "deadline_success_probability"))
        b.plot(s.index * 100, s.values, ls=ls, marker="o", ms=2.6, color=MODE_COLOR[mode])
b.plot([], [], color=INK2, ls="-", label="i.i.d. loss"); b.plot([], [], color=INK2, ls=(0, (3, 2)), label="Gilbert–Elliott bursts")
b.legend(loc="lower left", fontsize=6.3); b.set_xlabel("Packet-loss rate at 1 Mb/s (%)"); b.set_xticks([0, 1, 5, 15]); panel(b, "b")
fig.tight_layout(w_pad=1.0); save(fig, OUT, "fig3_transport_surface")

# ---------------- Fig. 4: controller benchmark (Experiments 4/5) ----------------
cells = pd.read_csv(DATA / "exp4_cells.csv.gz")
policies = sorted(cells.policy.unique())
res = {p: balanced_timely(cells, p) for p in policies}
T = pd.DataFrame({p: {f"{k}_{s}": v[i] for k, v in r.items() for i, s in enumerate(("pt", "lo", "hi"))}
                  for p, r in res.items()}).T
T.index.name = "policy"; T.to_csv(OUT / "table_exp4_5.csv")
# collapse exact behavioural duplicates so the plot shows each distinct policy once
T["_b"] = T.balanced_pt.round(3); T["_k"] = T.kb_pt.round(1); T["_e"] = T.energy_pt.round(2)
dup_groups = (T.reset_index().groupby(["_b", "_k", "_e"]).policy.apply(list))
BF = best_fixed_info(DATA)
NAME = {"proposed_full": "Proposed (full)", "best_fixed_validation": BF["label"],
        "heuristic_lookahead_reference": "Lookahead reference*", "fixed_l5_image_or_clip": "Fixed L5 image",
        "fixed_l3_semantics_roi": "Fixed L3", "fixed_l2_semantics": "Fixed L2", "fixed_l0_heartbeat": "Fixed L0",
        "fixed_l4_features": "Fixed L4", "channel_only_hysteresis": "Channel-only", "channel_plus_mission": "Channel+mission",
        "queue_deadline_greedy": "Queue/deadline greedy", "risk_aware_no_uncertainty": "Risk-aware, no uncert.",
        "mission_only": "Mission-only", "adaptive_image_codec": "Adaptive image codec", "adaptive_feature_rate": "Adaptive feature rate",
        "ablate_receiver_state": "w/o receiver state", "ablate_mission_risk": "w/o mission risk", "ablate_evidence_escalation": "w/o evidence escal."}
rep_of = {}
for grp in dup_groups:
    pick = next((p for p in ("proposed_full", "best_fixed_validation", "fixed_l2_semantics") if p in grp), grp[0])
    rep_of[pick] = grp
fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
for p, grp in rep_of.items():
    r = T.loc[p]
    if r.kb_pt <= 0:
        continue
    hl = p in ("proposed_full", "best_fixed_validation", "heuristic_lookahead_reference")
    col = PROPOSED if p == "proposed_full" else (BF["color"] if p == "best_fixed_validation" else (INK if hl else GHOST))
    a.plot([r.kb_pt, r.kb_pt], [r.balanced_lo, r.balanced_hi], color=col if hl else GHOST, lw=0.9, zorder=2)
    a.plot([r.kb_lo, r.kb_hi], [r.balanced_pt] * 2, color=col if hl else GHOST, lw=0.9, zorder=2)
    a.scatter(r.kb_pt, r.balanced_pt, s=40 if hl else 18, color=col, edgecolor="white", lw=0.6, zorder=6 if hl else 4,
              marker="D" if p == "heuristic_lookahead_reference" else "o")
    lbl = NAME.get(p, p)
    if p == "proposed_full" and len(grp) > 1:
        _same = sorted(({"always_vlm_full": "always-VLM", "ablate_hysteresis": "w/o hysteresis"}.get(q, NAME.get(q, q))
                        for q in grp if q != p), key=lambda name: (name != "always-VLM", name))
        lbl += "\n(= " + ",\n    ".join(", ".join(_same[i:i + 2]) for i in range(0, len(_same), 2)) + ")"
    off = {"proposed_full": (-8, -16), "best_fixed_validation": (7, 0), "heuristic_lookahead_reference": (-9, 7),
           "fixed_l2_semantics": (7, -7), "fixed_l5_image_or_clip": (6, 9), "fixed_l0_heartbeat": (6, -8)}.get(p)
    if off:
        a.annotate(lbl, (r.kb_pt, r.balanced_pt), xytext=off, textcoords="offset points", fontsize=6.4,
                   color=INK if hl else INK2, ha="right" if off[0] < 0 else "left", va="center")
_b, _h, _p = T.loc["best_fixed_validation"], T.loc["heuristic_lookahead_reference"], T.loc["proposed_full"]
_top = _p if _p.balanced_pt > _b.balanced_pt else _b          # best deployable operating point
if _h.balanced_pt - _top.balanced_pt >= 0.02:
    a.annotate("", xy=(3.0, _h.balanced_pt), xytext=(3.0, _top.balanced_pt),
               arrowprops=dict(arrowstyle="<->", color=INK, lw=0.8))
    a.text(3.15, (_h.balanced_pt + _top.balanced_pt) / 2,
           f"+{100*(_h.balanced_pt-_top.balanced_pt):.0f} pts headroom\nfor adaptive control", fontsize=6.6, va="center", color=INK)
if _p.balanced_pt - _b.balanced_pt >= 0.02:
    a.text(_p.kb_pt * 1.12, _p.balanced_pt - 0.045,
           f"+{100*(_p.balanced_pt-_b.balanced_pt):.0f} pts over the best fixed\npolicy at {_p.kb_pt/_b.kb_pt:.1f}× its bytes",
           fontsize=6.4, color=PROPOSED, va="top")
elif _p.kb_pt / _b.kb_pt >= 1.5:
    a.text(_b.kb_pt * 1.12, _b.balanced_pt - 0.045, f"{_p.kb_pt/_b.kb_pt:.1f}× fewer bytes than\nthe proposed controller",
           fontsize=6.4, color=BF["color"], va="top")
a.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2))); a.text(65, 0.515, "chance", fontsize=6.3, color=MUTED, ha="right")
a.text(0.98, 0.02, "gray: other registered baselines and ablations", transform=a.transAxes, ha="right", va="bottom", fontsize=6, color=MUTED)
a.set_xscale("log"); a.set_xlim(0.6, 70); a.set_ylim(0, 0.95)
a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a.set_xlabel("Wire kB per decision (log)"); a.set_ylabel("Cohort-balanced correct-and-timely rate"); panel(a, "a")

mc = pd.read_csv(DATA / "exp4_mode_counts.csv.gz")
show = ["heuristic_lookahead_reference", "best_fixed_validation", "fixed_l2_semantics", "proposed_full", "fixed_l5_image_or_clip"]
mix = mc[mc.policy.isin(show)].pivot_table(index="policy", columns="selected_mode", values="n", aggfunc="sum").fillna(0)
mix = mix.div(mix.sum(axis=1), axis=0).loc[show[::-1]]
left = np.zeros(len(mix))
for mode in MODE_ORDER:
    if mode in mix:
        v = mix[mode].to_numpy()
        b.barh(range(len(mix)), v, left=left, color=MODE_COLOR[mode], edgecolor="white", lw=0.8, height=0.62,
               label=MODE_LABEL[mode])
        for i, (x0, w) in enumerate(zip(left, v)):
            if w > 0.12:
                b.text(x0 + w / 2, i, f"{w:.0%}", ha="center", va="center", fontsize=6, color="white")
        left += v
b.set_yticks(range(len(mix)), [NAME.get(p, p) for p in mix.index]); b.set_xlim(0, 1)
b.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); b.set_xlabel("Share of decisions")
b.grid(axis="y", visible=False); panel(b, "b")
b.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.04), ncol=3, fontsize=6.2, borderaxespad=0, columnspacing=0.6,
         handlelength=1.0, handletextpad=0.3)
fig.tight_layout(w_pad=1.0); save(fig, OUT, "fig4_controller_benchmark")

# ---------------- Fig. 5: safety endpoints ----------------
order = T.loc[[p for p in rep_of if T.loc[p].kb_pt > 0]].sort_values("miss_pt").index
fig, a = plt.subplots(figsize=(COL, 2.75))
y = np.arange(len(order))[::-1]
for yi, p in zip(y, order):
    r = T.loc[p]
    col = PROPOSED if p == "proposed_full" else (BF["color"] if p == "best_fixed_validation" else INK2)
    a.plot([r.miss_lo, r.miss_hi], [yi, yi], color=col, lw=0.9)
    a.scatter(r.miss_pt, yi, s=22, color=col, zorder=3, marker="D" if p == "heuristic_lookahead_reference" else "o")
    if r.fa_pt > 0.001:
        a.text(1.02, yi, f"FA {r.fa_pt:.0%}", fontsize=6, color=INK2, va="center", transform=a.get_yaxis_transform())
a.set_yticks(y, [NAME.get(p, p) for p in order]); a.set_xlim(0, 1)
a.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a.set_xlabel("Critical-miss rate (critical episodes, 95% CI)"); a.grid(axis="y", visible=False)
fig.tight_layout(); save(fig, OUT, "fig5_safety")

# ---------------- Fig. 6: why the controller collapses (calibration) ----------------
sel = json.load(open(R / f"{PHASE}_selection.json"))
gate = pd.DataFrame(sel["calibration_gate"]).set_index("model_id")
sig = pd.read_csv(DATA / "exp4_decision_signal_sample.csv.gz")
pf = sig[sig.policy.eq("proposed_full")]
ctrl = sel.get("policy_spec", {}) if isinstance(sel.get("policy_spec"), dict) else {}
thr_unc = 0.35
fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.1), gridspec_kw={"width_ratios": [1.4, 1]})
bins = np.linspace(0, 0.6, 31)
a.hist(pf.decision_uncertainty.dropna(), bins=bins, color=PROPOSED, edgecolor="white", lw=0.5)
a.axvline(thr_unc, color=INK, lw=0.9, ls=(0, (3, 2)))
frac = (pf.decision_uncertainty >= thr_unc).mean()
a.set_ylim(0, a.get_ylim()[1] * 1.45)
a.text(thr_unc - 0.008, a.get_ylim()[1] * 0.97, f"VLM/escalation\ntrigger ({thr_unc})", ha="right", va="top", fontsize=6.4)
_u = pf.decision_uncertainty.dropna()
_bands = int((np.histogram(_u, bins=bins)[0] > 0.01 * len(_u)).sum()) if len(_u) else 0
_span = f"three narrow bands\n" if _bands == 3 else ""
a.text(0.60, a.get_ylim()[1] * 0.97, f"All decisions in {_span}({_u.min():.2f}–{_u.max():.2f});\n{frac:.0%} at or above the trigger", ha="right", va="top", fontsize=6.4, color=PROPOSED)
a.set_xlabel("Calibrated uncertainty seen by the controller (1 − p̂)"); a.set_ylabel("Decisions (sample)"); panel(a, "a")
MODELS = ["detector_only", "qwen3_vl_2b", "smolvlm2_2b", "qwen3_vl_8b_upper_bound"]
ML = ["Detector", "Qwen-2B", "Smol-2.2B", "Qwen-8B"]
yy = np.arange(len(MODELS))[::-1]
for yi, m in zip(yy, MODELS):
    r = gate.loc[m]
    b.plot([r.ece, r.ece_group_bootstrap_upper], [yi, yi], color=INK2, lw=0.9)
    b.scatter(r.ece, yi, s=24, color=INK, zorder=3)
    t = sel["calibration"][m]["temperature"]
    b.text(0.56, yi, f"T = {t:.2f} (bound)" if abs(np.log(t) - 3) < 1e-3 else f"T = {t:.2f}", fontsize=6.3, va="center", color=INK2)
b.axvline(0.10, color=MUTED, lw=0.7, ls=(0, (3, 2))); b.text(0.105, yy[0] + 0.3, "0.10", fontsize=6, color=MUTED, va="center")
b.set_yticks(yy, ML); b.set_xlim(0, 0.78); b.set_ylim(-0.9, len(MODELS) - 0.4)
b.set_xlabel("Expected calibration error (point, 95% upper)"); b.grid(axis="y", visible=False); panel(b, "b")
fig.tight_layout(w_pad=1.0); save(fig, OUT, "fig6_calibration_collapse")

keep = ["balanced", "pos", "neg", "kb", "energy", "miss", "fa"]
print(T[[f"{k}_pt" for k in keep]].round(3).sort_values("balanced_pt").to_string())
print("uncertainty>=0.35 fraction:", round(frac, 4), "n=", len(pf))

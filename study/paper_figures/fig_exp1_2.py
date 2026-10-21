import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # run with python -I
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter, PercentFormatter
from common import *
from stats import macro_recall_boot

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); PHASE = sys.argv[3] if len(sys.argv) > 3 else "validation"
style()
R = DATA / "results"
MODEL_LABEL = {"detector_only": "Detector only", "qwen3_vl_2b": "Qwen3-VL-2B",
               "smolvlm2_2b": "SmolVLM2-2.2B", "qwen3_vl_8b_upper_bound": "Qwen3-VL-8B\n(upper bound)"}
MODELS = list(MODEL_LABEL)

det = pd.read_csv(R / f"{PHASE}_exp1_perception_detail.csv.gz", low_memory=False)
for c in ("action_correct", "reached_token_cap", "full_path_latency_ms", "output_tokens"):
    det[c] = as_num(det[c])
rows = []
for m in MODELS:
    d = det[det.model_id.eq(m)]
    pt, ci, per = macro_recall_boot(d, correct="action_correct")
    rows.append({"model": m, "macro": pt, "lo": ci[0], "hi": ci[1], **per,
                 "acc": d.action_correct.mean(),
                 "lat50": d.full_path_latency_ms.median(), "lat05": d.full_path_latency_ms.quantile(.05),
                 "lat95": d.full_path_latency_ms.quantile(.95), "cap": d.reached_token_cap.mean(),
                 "tok": d.output_tokens.median()})
E1 = pd.DataFrame(rows); E1.to_csv(OUT / "table_exp1.csv", index=False) if OUT.exists() else None

# ---------------- Fig. 1: perception quality and cost ----------------
fig, (a, b) = plt.subplots(1, 2, figsize=(DCOL, 2.25), gridspec_kw={"width_ratios": [1.25, 1]})
y = np.arange(len(MODELS))[::-1]
for yi, r in zip(y, E1.itertuples()):
    a.plot([r.lo, r.hi], [yi, yi], color=INK2, lw=1.0, solid_capstyle="butt", zorder=2)
    a.scatter(r.macro, yi, s=34, color=INK, zorder=4, label="Class-balanced recall (95% CI)" if yi == y[0] else None)
    for k in CLASS_ORDER:
        a.scatter(getattr(r, k), yi + 0.22, s=16, marker=CLASS_MARK[k], facecolor="white",
                  edgecolor=INK2, lw=0.7, zorder=3, label=f"Recall: {k.replace('_', ' ').lower()}" if yi == y[0] else None)
a.axvline(1 / 3, color=MUTED, lw=0.7, ls=(0, (3, 2)))
a.text(1 / 3 + 0.01, y[-1] - 0.55, "chance (1/3)", color=MUTED, fontsize=6.5, va="center")
a.set_yticks(y, [MODEL_LABEL[m] for m in MODELS]); a.set_ylim(-0.8, len(MODELS) - 0.3)
a.set_xlim(0, 1); a.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a.set_xlabel("Action recall"); a.grid(axis="y", visible=False); panel(a, "a")
a.legend(loc="lower left", bbox_to_anchor=(0, 1.06), fontsize=6.3, handletextpad=0.2, borderaxespad=0, ncol=4, columnspacing=0.8)

for yi, r in zip(y, E1.itertuples()):
    b.plot([r.lat05, r.lat95], [yi, yi], color=INK2, lw=1.0)
    b.scatter(r.lat50, yi, s=34, color=INK, zorder=3)
    note = f"{r.cap:.0%} hit token cap" if r.cap > 0.01 else ""
    if note:
        b.text(max(r.lat95, r.lat50) * 1.6, yi, note, va="center", fontsize=6.5, color=INK2)
b.set_xscale("log"); b.set_yticks(y, []); b.set_ylim(-0.8, len(MODELS) - 0.3)
b.set_xlim(10, 6e4)
b.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
b.set_xlabel("Full-path latency per frame (ms, median and 5–95th pct.)")
b.grid(axis="y", visible=False); panel(b, "b")
fig.tight_layout(w_pad=1.2)
save(fig, OUT, "fig1_perception")

# ---------------- Fig. 2: representation frontier ----------------
fr = pd.read_csv(DATA / "exp2_frontier_by_sample.csv.gz")
meta = det[det.model_id.eq("detector_only")][["observation_id", "gold_action", "visual_group_id"]]
fr = fr.merge(meta, left_on="sample_id", right_on="observation_id", how="inner")
fr["correct"] = fr["mean_native_receiver_correct"]
full_n = fr.groupby(["mode", "variant", "source_model"]).sample_id.nunique()
keep = full_n[full_n.eq(full_n.max())].index            # matched cohort: variants covering every sample
fr = fr.set_index(["mode", "variant", "source_model"]).loc[keep].reset_index()
fr = fr[~fr.variant.str.endswith("__detector_only")]
pts = []
for (mode, var, src), d in fr.groupby(["mode", "variant", "source_model"]):
    pt, ci, per = macro_recall_boot(d, reps=2000)
    pts.append({"mode": mode, "variant": var, "bytes": d.mean_wire_bytes.mean(), "macro": pt,
                "lo": ci[0], "hi": ci[1], "acc": d.correct.mean(), **per})
P = pd.DataFrame(pts).sort_values("bytes")
P.to_csv(OUT / "table_exp2.csv", index=False)
best = -1; front = []
for r in P.itertuples():
    if r.macro > best + 1e-12:
        front.append(r); best = r.macro
F = pd.DataFrame(front)

# Single wide panel in the style of the original study preview: colored payload levels,
# numbered Pareto points with a key, legend outside, class-balanced metric with CIs.
fig, a = plt.subplots(figsize=(DCOL, 3.0))
for mode in MODE_ORDER:
    d = P[P["mode"].eq(mode)]
    big = mode != "L5_IMAGE_OR_CLIP"
    a.scatter(d.bytes, d.macro, s=46 if big else 26, color=MODE_COLOR[mode], edgecolor="white", lw=0.7,
              zorder=3 if big else 2, label=MODE_LABEL[mode])
for r in F.itertuples():
    a.plot([r.bytes, r.bytes], [r.lo, r.hi], color=INK2, lw=0.8, zorder=1, alpha=0.8)
a.step(F.bytes, F.macro, where="post", color=INK, lw=1.3, zorder=4)
a.scatter(F.bytes, F.macro, s=22, color=INK, zorder=5, label="Pareto frontier (95% CI)")
def nice(v):
    m = {"l0_cbor": "L0 CBOR", "l1_cbor": "L1 CBOR", "l2_cbor": "L2 CBOR", "cbor+jpeg_q50_roi": "L3 CBOR+ROI",
         "registered_feature_int4": "L4 INT4", "registered_feature_int8": "L4 INT8", "lossless_png_reference": "PNG lossless"}
    if v in m:
        return m[v]
    c, q, sc = v.split("_")
    return f"{c.upper() if c != 'webp' else 'WebP'} q{q[1:]} ×{sc[1:]}"
for i, r in enumerate(F.itertuples(), 1):
    a.annotate(str(i), (r.bytes, r.macro), xytext=(0, 6), textcoords="offset points", ha="center",
               fontsize=6.8, fontweight="bold", color=INK, zorder=6)
for v, off in {"registered_feature_int4": (0, -10), "registered_feature_int8": (0, -10),
               "cbor+jpeg_q50_roi": (0, -10), "lossless_png_reference": (0, 7), "l2_cbor": (-4, -11)}.items():
    r = P[P.variant.eq(v)].iloc[0]
    a.annotate(nice(v), (r.bytes, r.macro), xytext=off, textcoords="offset points", ha="center", fontsize=6.4, color=INK2)
a.axhline(1 / 3, color=MUTED, lw=0.7, ls=(0, (3, 2)))
a.text(3.2e6, 1 / 3 + 0.008, "chance (3 classes)", fontsize=6.4, color=MUTED, ha="right")
a.set_xscale("log"); a.set_xlim(4.5e2, 3.5e6); a.set_ylim(0.28, 0.72)
a.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
a.set_xlabel("Wire bytes per observation (log scale)"); a.set_ylabel("Class-balanced receiver recall")
a.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.8, title="Payload level", title_fontsize=7,
         alignment="left", borderaxespad=0)
_l1 = P[P.variant.eq("l1_cbor")].iloc[0]; _best = F.iloc[-1]
a.annotate(f"L1 alert: {_l1.macro/_best.macro:.0%} of the best image's quality\nat {_l1.bytes/_best.bytes:.1%} of its bytes ({_l1.bytes/1e3:.1f} kB vs {_best.bytes/1e3:.0f} kB)",
           xy=(_l1.bytes, _l1.macro), xytext=(1.6e3, 0.665), fontsize=7, color=INK, ha="left",
           arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.8, shrinkA=2, shrinkB=5, connectionstyle="arc3,rad=0.25"),
           bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GHOST, lw=0.6))
key = "Pareto points: " + " · ".join(f"{i}. {nice(r.variant)}" for i, r in enumerate(F.itertuples(), 1))
fig.text(0.0, -0.02, key, fontsize=6.4, color=INK2, ha="left", va="top", wrap=True)
fig.tight_layout()
save(fig, OUT, "fig2_representation_frontier")

# Companion per-class view (Fig. 2 inset / supplementary): why raw accuracy misleads
show = ["l1_cbor", "cbor+jpeg_q50_roi", "registered_feature_int4", "registered_feature_int8"]
show += [F[F["mode"].eq("L5_IMAGE_OR_CLIP")].iloc[-1].variant, "lossless_png_reference"]
S = P.set_index("variant").loc[show]
fig, b = plt.subplots(figsize=(COL, 2.3))
yy = np.arange(len(S))[::-1]
for k, dx, col in zip(CLASS_ORDER, (-0.24, 0, 0.24), (INK, INK2, GHOST)):
    b.barh(yy + dx, S[k], height=0.22, color=col, label=k.replace("_", " ").lower())
b.scatter(S.acc, yy, marker="|", s=90, color="#e34948", zorder=4, label="raw accuracy")
b.set_yticks(yy, [nice(v) for v in S.index]); b.set_xlim(0, 1)
b.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0)); b.set_xlabel("Per-class recall")
b.grid(axis="y", visible=False)
b.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=4, fontsize=6.2, borderaxespad=0, columnspacing=0.7, handlelength=1.0)
fig.tight_layout()
save(fig, OUT, "fig2b_per_class_recall")
print(E1[["model", "macro", "lo", "hi", "acc", "lat50", "cap", "tok"]].round(3).to_string())
print(F[["mode", "variant", "bytes", "macro", "lo", "hi", "acc"]].round(3).to_string())

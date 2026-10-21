"""Shared style, palette and statistics for the SemUAV-VLM IEEE figures."""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # run with python -I

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE two-column geometry (inches)
COL = 3.5
DCOL = 7.16

INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8984"; GRID = "#e4e3df"; GHOST = "#c9c8c3"
# Fixed categorical order (validated: dataviz reference palette, slots 1-6), one hue per payload level
MODE_ORDER = ["L0_HEARTBEAT", "L1_ALERT", "L2_SEMANTICS", "L3_SEMANTICS_ROI", "L4_FEATURES", "L5_IMAGE_OR_CLIP"]
MODE_COLOR = dict(zip(MODE_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
MODE_LABEL = {"L0_HEARTBEAT": "L0 heartbeat", "L1_ALERT": "L1 alert", "L2_SEMANTICS": "L2 semantics",
              "L3_SEMANTICS_ROI": "L3 semantics+ROI", "L4_FEATURES": "L4 features", "L5_IMAGE_OR_CLIP": "L5 image"}
PROPOSED = "#4a3aa7"   # slot 7 violet: the proposed controller only
CLASS_ORDER = ["RESPOND", "REVIEW", "NO_ACTION"]
CLASS_MARK = {"RESPOND": "o", "REVIEW": "s", "NO_ACTION": "^"}

def style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
        "legend.frameon": False, "savefig.dpi": 600, "pdf.fonttype": 42, "ps.fonttype": 42,
        "lines.linewidth": 1.2, "figure.dpi": 150,
    })

def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def panel(ax, letter):
    ax.text(-0.02, 1.0, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, fontweight="bold")

def as_num(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    m = s.astype(str).str.strip().str.lower().map({"true": 1.0, "false": 0.0})
    return pd.to_numeric(s, errors="coerce").fillna(m)

def cluster_boot(df, cluster, stat, reps=2000, seed=20260823):
    """Percentile CI from resampling whole clusters with replacement."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby(cluster)]
    k = len(groups)
    vals = []
    for _ in range(reps):
        idx = rng.integers(0, k, k)
        vals.append(stat(pd.concat([groups[i] for i in idx])))
    return np.nanpercentile(vals, [2.5, 97.5])

def twoway_weights(n_a, n_b, reps, seed):
    """Two-way (crossed) cluster bootstrap weights, as in the registered H2 estimator."""
    rng = np.random.default_rng(seed)
    wa = rng.multinomial(n_a, np.full(n_a, 1 / n_a), size=reps)
    wb = rng.multinomial(n_b, np.full(n_b, 1 / n_b), size=reps)
    return wa, wb


def best_fixed_info(data_dir):
    """Mode, label and colour of the frozen best-fixed comparator (validation policy spec)."""
    import json
    spec_path = Path(data_dir) / "results" / "validation_policy_spec.json"
    mode = "L1_ALERT"
    try:
        mode = json.loads(spec_path.read_text())["best_fixed"]["mode"]
    except (OSError, KeyError, ValueError):
        pass
    short = MODE_LABEL.get(mode, mode)
    return {"mode": mode, "label": f"Best fixed ({short})", "short": f"Best fixed ({short.split()[0]})",
            "color": MODE_COLOR.get(mode, INK)}

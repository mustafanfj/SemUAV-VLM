import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # run with python -I
import numpy as np
import pandas as pd
from common import CLASS_ORDER

def macro_recall_boot(df, correct="correct", cls="gold_action", cluster="visual_group_id",
                      reps=4000, seed=20260823):
    """Macro (class-balanced) recall with a visual-group cluster bootstrap.

    Returns point estimate, CI, and per-class point recalls."""
    classes = [c for c in CLASS_ORDER if c in set(df[cls])]
    clusters = sorted(df[cluster].unique())
    ci = {c: i for i, c in enumerate(clusters)}
    S = np.zeros((len(clusters), len(classes))); N = np.zeros_like(S)
    for (g, k), sub in df.groupby([cluster, cls]):
        if k in classes:
            S[ci[g], classes.index(k)] = sub[correct].sum(); N[ci[g], classes.index(k)] = len(sub)
    point_k = S.sum(0) / N.sum(0)
    rng = np.random.default_rng(seed)
    W = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=reps)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = np.nanmean((W @ S) / (W @ N), axis=1)
    return float(point_k.mean()), np.nanpercentile(boot, [2.5, 97.5]), dict(zip(classes, point_k))

def twoway_cells(cells, policy, num_fn, reps=4000, seed=20260823, a="visual_group_id", b="trace_id"):
    """Crossed visual-group x trace bootstrap for a ratio-of-sums metric.

    num_fn(frame) -> dict name -> per-cell numerator/denominator arrays combined in metric()."""
    sub = cells[cells.policy.eq(policy)]
    A = sorted(sub[a].unique()); B = sorted(sub[b].unique())
    ia = sub[a].map({v: i for i, v in enumerate(A)}).to_numpy()
    ib = sub[b].map({v: i for i, v in enumerate(B)}).to_numpy()
    comps = num_fn(sub)                       # name -> per-row values
    tensors = {}
    for name, vals in comps.items():
        T = np.zeros((len(A), len(B)))
        np.add.at(T, (ia, ib), vals)
        tensors[name] = T
    rng = np.random.default_rng(seed)
    wa = rng.multinomial(len(A), np.full(len(A), 1 / len(A)), size=reps)
    wb = rng.multinomial(len(B), np.full(len(B), 1 / len(B)), size=reps)
    weighted = {n: np.einsum("ri,ij,rj->r", wa, T, wb) for n, T in tensors.items()}
    point = {n: T.sum() for n, T in tensors.items()}
    return point, weighted

def balanced_timely(cells, policy, reps=4000, seed=20260823):
    """Cohort-balanced correct-and-timely rate: mean of critical-episode and non-critical-episode rates."""
    def comps(f):
        crit = f.critical.to_numpy(); dec = f.decisions.to_numpy()
        pos_ok = (f.critical - f.critical_miss).to_numpy()
        return {"pos_ok": pos_ok, "pos_n": crit,
                "neg_ok": (f.correct_timely_action - (f.critical - f.critical_miss)).to_numpy(),
                "neg_n": dec - crit, "bytes": f.total_wire_bytes.to_numpy(), "dec": dec,
                "energy": f.energy_j.to_numpy(), "fa": f.false_alert.to_numpy(),
                "miss": f.critical_miss.to_numpy()}
    p, w = twoway_cells(cells, policy, comps, reps, seed)
    def metric(d):
        with np.errstate(invalid="ignore", divide="ignore"):
            return {"balanced": 0.5 * (d["pos_ok"] / d["pos_n"] + d["neg_ok"] / d["neg_n"]),
                    "pos": d["pos_ok"] / d["pos_n"], "neg": d["neg_ok"] / d["neg_n"],
                    "kb": d["bytes"] / d["dec"] / 1e3, "energy": d["energy"] / d["dec"],
                    "miss": d["miss"] / d["pos_n"], "fa": d["fa"] / d["neg_n"]}
    pt = metric(p); bt = metric(w)
    return {k: (float(pt[k]), *np.nanpercentile(bt[k], [2.5, 97.5])) for k in pt}

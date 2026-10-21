import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # run with python -I
import sys
from pathlib import Path
import pandas as pd
from common import best_fixed_info
DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); PHASE = sys.argv[3] if len(sys.argv) > 3 else "validation"
T = pd.read_csv(OUT / "table_exp4_5.csv", index_col=0)
rows = [("heuristic_lookahead_reference", "Lookahead reference$^\\dagger$"),
        ("proposed_full", "Proposed controller (= best fixed L1)$^\\ddagger$"),
        ("fixed_l2_semantics", "Fixed L2 semantics$^\\S$"), ("fixed_l0_heartbeat", "Fixed L0 heartbeat (never alert)"),
        ("fixed_l5_image_or_clip", "Fixed L5 image (AVIF q90 $\\times$0.25)"), ("adaptive_image_codec", "Adaptive image codec"),
        ("fixed_l4_features", "Fixed L4 features (INT4)$^\\P$"), ("mission_only", "Mission-only"),
        ("fixed_l3_semantics_roi", "Fixed L3 semantics + ROI")]
f = lambda r, k, pct=True: (f"{100*r[k+'_pt']:.1f} [{100*r[k+'_lo']:.1f}, {100*r[k+'_hi']:.1f}]" if pct
                            else f"{r[k+'_pt']:.2f} [{r[k+'_lo']:.2f}, {r[k+'_hi']:.2f}]")
L = ["\\begin{table*}[t]", "\\centering", "\\caption{Closed-loop policy benchmark (" + PHASE +
     " split; 7 sequences $\\times$ 12 traces $\\times$ 3 mission profiles $\\times$ 5 seeds, 80{,}640 decisions per policy). "
     "Point estimates with 95\\% two-way cluster-bootstrap intervals over visual sequences and traces.}",
     "\\label{tab:controller}", "\\small", "\\begin{tabular}{lcccc}", "\\toprule",
     "Policy & Balanced correct \\& timely (\\%) & Critical miss (\\%) & Wire kB/decision & GPU-board J/decision \\\\", "\\midrule"]
for p, name in rows:
    r = T.loc[p]
    L.append(f"{name} & {f(r,'balanced')} & {f(r,'miss')} & {f(r,'kb',False)} & {f(r,'energy',False)} \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\\\[2pt]\\footnotesize $^\\dagger$Non-deployable: uses privileged next-step outcome information. "
      "$^\\ddagger$Makes decisions identical to 12 other policies: best fixed (L1), fixed L1, always-VLM, channel-only, channel+mission, "
      "queue/deadline greedy, risk-aware without uncertainty, adaptive semantic rate, and the four ablations (evidence escalation, hysteresis, "
      "mission risk, receiver state). "
      "$^\\S$Identical to detector-only semantics. $^\\P$Identical to adaptive feature rate.", "\\end{table*}"]
(OUT / "table_controller.tex").write_text("\n".join(L) + "\n")
x = pd.read_csv(DATA / "results" / f"{PHASE}_exp6_fault_summary.csv.gz")
pc = [c for c in x.columns if "pass" in c and "criter" not in c]
L = ["\\begin{table}[t]", "\\centering", "\\caption{Registered protocol fault campaign (Experiment 6).}", "\\label{tab:faults}",
     "\\footnotesize", "\\begin{tabular}{lc}", "\\toprule", "Injected fault & Registered outcome \\\\", "\\midrule"]
for r in x.itertuples():
    L.append(f"{str(r.case).replace('_', ' ').capitalize()} & {int(r.passed_runs)}/{int(r.runs)} seeds pass \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
(OUT / "table_faults.tex").write_text("\n".join(L) + "\n")
print(open(OUT / "table_controller.tex").read()); print(pc)

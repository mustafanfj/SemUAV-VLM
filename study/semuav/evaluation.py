from __future__ import annotations

import dataclasses
import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .registry import StudyGateError


PRIMARY_ENDPOINT = "correct receiver action before the absolute mission deadline"


def add_primary_endpoint(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"gold_action", "receiver_action", "action_ready_us", "deadline_us", "payload_valid", "state_valid"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Endpoint input missing columns: {sorted(missing)}")
    result = frame.copy()
    result["delivered_by_deadline"] = (
        result["action_ready_us"].notna()
        & (result["action_ready_us"] <= result["deadline_us"])
        & result["payload_valid"].fillna(False).astype(bool)
        & result["state_valid"].fillna(False).astype(bool)
    )
    result["correct_action"] = result["receiver_action"] == result["gold_action"]
    result["correct_timely_action"] = result["correct_action"] & result["delivered_by_deadline"]
    result["false_alert"] = (
        result["delivered_by_deadline"]
        & result["receiver_action"].eq("RESPOND")
        & ~result["gold_action"].eq("RESPOND")
    )
    result["critical_miss"] = result["gold_action"].eq("RESPOND") & ~result["correct_timely_action"]
    return result


def assert_paired_design(frame: pd.DataFrame, policy_col: str = "policy") -> None:
    # Pair at the actual trial/observation level, while retaining
    # visual_group_id as the cluster for inference.  A visual group may contain
    # multiple frames and must not be mistaken for one unique trial row.
    trial_id = "sample_id" if "sample_id" in frame.columns else "visual_group_id"
    keys = [trial_id, "mission_profile", "trace_id", "replicate_seed"]
    missing = set(keys + [policy_col]).difference(frame.columns)
    if missing:
        raise ValueError(f"Paired design missing columns: {sorted(missing)}")
    counts = frame.groupby(keys, dropna=False)[policy_col].nunique()
    expected = frame[policy_col].nunique()
    if not (counts == expected).all():
        raise StudyGateError("Every policy must receive exactly the same visual-mission-trace-seed tuples")
    duplicates = frame.duplicated(keys + [policy_col])
    if duplicates.any():
        raise StudyGateError("Duplicate policy rows violate the paired design")


@dataclasses.dataclass(frozen=True)
class BootstrapEffect:
    method: str
    reference: str
    endpoint: str
    estimate: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    repetitions: int
    visual_clusters: int
    trace_clusters: int


def two_way_clustered_paired_bootstrap(
    frame: pd.DataFrame,
    method: str,
    reference: str,
    endpoint: str = "correct_timely_action",
    repetitions: int = 10_000,
    seed: int = 20260823,
    confidence: float = 0.95,
) -> BootstrapEffect:
    assert_paired_design(frame)
    subset = frame[frame["policy"].isin([method, reference])]
    trial_id = "sample_id" if "sample_id" in subset.columns else "visual_group_id"
    index = [trial_id, "visual_group_id", "mission_profile", "trace_id", "replicate_seed"]
    index = list(dict.fromkeys(index))
    pivot = subset.pivot(index=index, columns="policy", values=endpoint).reset_index()
    if method not in pivot or reference not in pivot or pivot[[method, reference]].isna().any().any():
        raise StudyGateError("Missing paired outcomes")
    pivot["delta"] = pivot[method].astype(float) - pivot[reference].astype(float)
    visual = pivot["visual_group_id"].astype(str).unique()
    traces = pivot["trace_id"].astype(str).unique()
    # Macro-average frames/observations inside each independent visual group so
    # long videos do not receive greater weight than short videos.
    values = (
        pivot.groupby(["visual_group_id", "trace_id"], as_index=False)["delta"]
        .mean()
    )
    # The point estimate must use the same cluster-level estimand as the
    # bootstrap distribution; a frame-weighted estimate paired with a
    # cluster-weighted interval would be internally inconsistent.
    estimate = float(values["delta"].mean())
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    # A two-way cluster draw gives every observed visual-by-trace cell the
    # product of its independently sampled visual and trace multiplicities.
    # Expressing that operation as two small matrix contractions avoids a
    # Python loop over every observed cell for every bootstrap repetition.  A
    # separate presence matrix is essential for sparse designs: unobserved
    # cells must contribute neither a zero outcome nor exposure.
    values["visual_group_id"] = values["visual_group_id"].astype(str)
    values["trace_id"] = values["trace_id"].astype(str)
    visual_index = {label: index for index, label in enumerate(map(str, visual))}
    trace_index = {label: index for index, label in enumerate(map(str, traces))}
    value_matrix = np.zeros((len(visual), len(traces)), dtype=float)
    presence_matrix = np.zeros((len(visual), len(traces)), dtype=float)
    visual_codes = values["visual_group_id"].map(visual_index).to_numpy(np.int64)
    trace_codes = values["trace_id"].map(trace_index).to_numpy(np.int64)
    np.add.at(
        value_matrix,
        (visual_codes, trace_codes),
        values["delta"].to_numpy(float),
    )
    np.add.at(presence_matrix, (visual_codes, trace_codes), 1.0)

    rng = np.random.default_rng(seed)
    draws = np.full(repetitions, np.nan, dtype=float)
    # Bound the two multiplicity arrays to roughly two million entries per
    # batch, keeping memory predictable even for unusually large manifests.
    weight_entries_per_rep = max(1, len(visual) + len(traces))
    batch_size = max(1, min(repetitions, 2_000_000 // weight_entries_per_rep))
    for start in range(0, repetitions, batch_size):
        stop = min(repetitions, start + batch_size)
        size = stop - start
        visual_weights = np.empty((size, len(visual)), dtype=np.int64)
        trace_weights = np.empty((size, len(traces)), dtype=np.int64)
        # These cluster-count vectors are multinomial weights.  Draw the two
        # cluster dimensions in the historical per-repetition order so a
        # frozen seed continues to reproduce the exact established analysis,
        # while all work over observed cells remains batched below.
        for offset in range(size):
            visual_weights[offset] = np.bincount(
                rng.choice(len(visual), len(visual), replace=True),
                minlength=len(visual),
            )
            trace_weights[offset] = np.bincount(
                rng.choice(len(traces), len(traces), replace=True),
                minlength=len(traces),
            )
        numerators = np.einsum(
            "bi,ij,bj->b",
            visual_weights,
            value_matrix,
            trace_weights,
            optimize=True,
        )
        denominators = np.einsum(
            "bi,ij,bj->b",
            visual_weights,
            presence_matrix,
            trace_weights,
            optimize=True,
        )
        valid = denominators > 0
        batch_draws = np.full(size, np.nan, dtype=float)
        batch_draws[valid] = numerators[valid] / denominators[valid]
        draws[start:stop] = batch_draws
    draws = draws[np.isfinite(draws)]
    if len(draws) < repetitions * 0.99:
        raise StudyGateError("Too many invalid bootstrap replicates; confirm fully crossed design")
    alpha = 1.0 - confidence
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    # Centering each observed cell at the registered point estimate commutes
    # with the weighted mean, so the null distribution is exactly draws minus
    # estimate (including for sparse designs with varying denominators).
    null_draws = draws - estimate
    p = (1 + np.sum(np.abs(null_draws) >= abs(estimate))) / (len(null_draws) + 1)
    return BootstrapEffect(method, reference, endpoint, estimate, float(low), float(high), float(min(1, p)), len(draws), len(visual), len(traces))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[name] = running
    return adjusted


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def resource_summary(frame: pd.DataFrame) -> pd.DataFrame:
    delivered = frame[frame["action_ready_us"].notna()].copy()
    latency = delivered.assign(latency_ms=(delivered["action_ready_us"] - delivered["capture_us"]) / 1000)
    aggregations = frame.groupby("policy").agg(
        trials=("correct_timely_action", "size"),
        correct_timely_rate=("correct_timely_action", "mean"),
        mean_wire_bytes=("total_wire_bytes", "mean"),
        mean_energy_j=("energy_j", "mean"),
        delivery_rate=("delivered_by_deadline", "mean"),
    )
    conditional = []
    for policy, group in frame.groupby("policy"):
        noncritical = group[~group["gold_action"].eq("RESPOND")]
        critical = group[group["gold_action"].eq("RESPOND")]
        conditional.append({
            "policy": policy,
            "false_alert_rate": float(noncritical["false_alert"].mean()) if len(noncritical) else np.nan,
            "critical_miss_rate": float(critical["critical_miss"].mean()) if len(critical) else np.nan,
        })
    aggregations = aggregations.join(pd.DataFrame(conditional).set_index("policy"))
    if not latency.empty:
        quantiles = latency.groupby("policy")["latency_ms"].quantile([0.5, 0.95, 0.99]).unstack()
        quantiles.columns = ["latency_p50_ms", "latency_p95_ms", "latency_p99_ms"]
        aggregations = aggregations.join(quantiles)
    return aggregations.reset_index()


def resource_rate_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute episode-duration-normalized two-way resource rates.

    Budgets in the preregistration are per second.  Each
    policy/trace/seed/episode (when ``episode_id`` is supplied) is charged from
    its first capture through one final observed capture interval, then totals
    are pooled by elapsed exposure rather than averaged per frame.  Grouping by
    the explicit episode identifier prevents independently rebased episode
    clocks from collapsing onto a single exposure window.
    """
    required = {
        "policy", "trace_id", "replicate_seed", "capture_us",
        "total_wire_bytes", "energy_j",
    }
    if missing := required.difference(frame.columns):
        raise ValueError(f"Resource-rate input missing columns: {sorted(missing)}")
    episode_keys = ["policy", "trace_id", "replicate_seed"]
    if "episode_id" in frame.columns:
        if frame["episode_id"].isna().any():
            raise ValueError("Resource-rate episode_id cannot be missing when supplied")
        episode_keys.append("episode_id")

    # Preserve measured energy components alongside the registered combined
    # compute total.  This keeps the sender/ground breakdown auditable while
    # budget eligibility remains defined by ``energy_j_per_s``.
    component_columns = sorted(
        column for column in frame.columns
        if column != "energy_j"
        and (column.endswith("_energy_j") or column.endswith("_energy_j_charged"))
    )
    episodes: list[dict[str, Any]] = []
    for key, group in frame.groupby(episode_keys, dropna=False, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        key_values = dict(zip(episode_keys, key))
        captures = np.sort(group["capture_us"].astype(np.int64).unique())
        if len(captures) >= 2:
            interval_us = int(np.median(np.diff(captures)))
            duration_us = int(captures[-1] - captures[0] + max(1, interval_us))
        else:
            duration_us = 1_000_000
        episodes.append({
            **key_values,
            "duration_s": duration_us / 1e6,
            "total_wire_bytes": float(group["total_wire_bytes"].sum()),
            "energy_j": float(group["energy_j"].sum()),
            **{
                column: float(pd.to_numeric(group[column], errors="raise").sum())
                for column in component_columns
            },
        })
    episode_frame = pd.DataFrame(episodes)
    rows = []
    for policy, group in episode_frame.groupby("policy", sort=True):
        duration = float(group["duration_s"].sum())
        total_wire_bytes = float(group["total_wire_bytes"].sum())
        total_energy_j = float(group["energy_j"].sum())
        row = {
            "policy": policy,
            "episodes": len(group),
            "exposure_s": duration,
            "total_wire_bytes": total_wire_bytes,
            "energy_j": total_energy_j,
            "wire_bytes_per_s": total_wire_bytes / duration,
            "energy_j_per_s": total_energy_j / duration,
        }
        for column in component_columns:
            total = float(group[column].sum())
            row[column] = total
            row[f"{column}_per_s"] = total / duration
        rows.append(row)
    return pd.DataFrame(rows)


def budget_compliance(
    frame: pd.DataFrame,
    wire_budget_bytes_per_s: float,
    energy_budget_j_per_s: float,
    tolerance: float = 0.05,
) -> pd.DataFrame:
    if wire_budget_bytes_per_s <= 0 or energy_budget_j_per_s <= 0:
        raise ValueError("Frozen wire and energy budgets must be positive")
    rates = resource_rate_summary(frame)
    rates["wire_cap_bytes_per_s"] = (1 + tolerance) * wire_budget_bytes_per_s
    rates["energy_cap_j_per_s"] = (1 + tolerance) * energy_budget_j_per_s
    rates["wire_pass"] = rates["wire_bytes_per_s"] <= rates["wire_cap_bytes_per_s"]
    rates["energy_pass"] = rates["energy_j_per_s"] <= rates["energy_cap_j_per_s"]
    rates["eligible"] = rates["wire_pass"] & rates["energy_pass"]
    return rates


def pareto_front(frame: pd.DataFrame, maximize: str, minimize: Sequence[str]) -> pd.DataFrame:
    data = frame.reset_index(drop=True).copy()
    dominated = np.zeros(len(data), dtype=bool)
    for i, row in data.iterrows():
        for j, challenger in data.iterrows():
            if i == j:
                continue
            no_worse = challenger[maximize] >= row[maximize] and all(challenger[col] <= row[col] for col in minimize)
            strictly = challenger[maximize] > row[maximize] or any(challenger[col] < row[col] for col in minimize)
            if no_worse and strictly:
                dominated[i] = True
                break
    data["pareto_optimal"] = ~dominated
    return data


def gate_summary(
    frame: pd.DataFrame,
    annotation_report: Mapping[str, Any],
    calibration_report: Mapping[str, float],
    primary_budget_bytes: float,
    primary_budget_energy_j: float,
    full_policy: str = "proposed_full",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    def add(gate: str, passed: bool, detail: str) -> None:
        rows.append({"gate": gate, "status": "PASS" if passed else "FAIL", "detail": detail})
    add("annotation_overall", annotation_report.get("overall_kappa", 0) >= 0.80, f"kappa={annotation_report.get('overall_kappa')}")
    add("annotation_rare", annotation_report.get("minimum_rare_kappa", 0) >= 0.70, f"min rare kappa={annotation_report.get('minimum_rare_kappa')}")
    add("calibration_overall", calibration_report.get("ece", 1) <= 0.05, f"ECE={calibration_report.get('ece')}")
    add("calibration_critical", calibration_report.get("critical_ece", 1) <= 0.10, f"critical ECE={calibration_report.get('critical_ece')}")
    policy = frame[frame["policy"] == full_policy]
    budget = budget_compliance(
        policy, primary_budget_bytes, primary_budget_energy_j
    ) if not policy.empty else pd.DataFrame()
    budget_row = budget.iloc[0] if len(budget) else None
    add("wire_budget", bool(budget_row is not None and budget_row.wire_pass), f"bytes/s={budget_row.wire_bytes_per_s if budget_row is not None else 'missing'}")
    add("energy_budget", bool(budget_row is not None and budget_row.energy_pass), f"J/s={budget_row.energy_j_per_s if budget_row is not None else 'missing'}")
    noncritical = policy[policy["gold_action"] != "RESPOND"] if not policy.empty else policy
    far = float(noncritical["false_alert"].mean()) if len(noncritical) else float("nan")
    add("false_alert_limit", np.isfinite(far) and far <= 0.05, f"rate={far:.4f}" if np.isfinite(far) else "missing")
    return pd.DataFrame(rows)


def minimum_sample_gate(manifest: pd.DataFrame, trace_manifest: pd.DataFrame) -> pd.DataFrame:
    counts = manifest.groupby("cohort")["visual_group_id"].nunique().to_dict() if "cohort" in manifest else {}
    safety = manifest[manifest["cohort"].eq("safety")] if "cohort" in manifest else manifest.iloc[0:0]
    critical = safety.loc[safety.get("critical", False).astype(bool), "visual_group_id"].nunique() if "critical" in safety else 0
    noncritical = safety.loc[~safety.get("critical", pd.Series(False, index=safety.index)).astype(bool), "visual_group_id"].nunique()
    test_traces = trace_manifest.loc[trace_manifest["split"].eq("test"), "trace_id"].nunique() if len(trace_manifest) else 0
    families = trace_manifest.loc[trace_manifest["split"].eq("test"), "family"].nunique() if len(trace_manifest) else 0
    checks = [
        ("natural_test_visual_groups", counts.get("natural_test", 0), 500),
        ("safety_critical_positive_groups", critical, 200),
        ("safety_noncritical_groups", noncritical, 500),
        ("ood_visual_groups", counts.get("ood", 0), 150),
        ("held_out_trace_realizations", test_traces, 30),
        ("held_out_trace_families", families, 6),
    ]
    return pd.DataFrame([{"gate": name, "observed": int(value), "required": required, "passed": value >= required} for name, value, required in checks])


def validation_power_simulation(
    discordant_rate: float,
    visual_clusters: int,
    trace_clusters: int,
    effect: float = 0.06,
    null_margin: float = 0.03,
    alpha: float = 0.05,
    simulations: int = 5000,
    seed: int = 20260823,
) -> float:
    """Conservative power screen for proving an effect exceeds a margin."""
    if not 0 < discordant_rate <= 1:
        raise ValueError("discordant_rate must lie in (0,1]")
    if not -1 <= effect <= 1 or not -1 <= null_margin <= 1 or not 0 < alpha < 1:
        raise ValueError("effect/margin must be in [-1,1] and alpha in (0,1)")
    effective_n = max(1, min(visual_clusters, visual_clusters * trace_clusters / (visual_clusters + trace_clusters)))
    standard_error = math.sqrt(max(1e-9, discordant_rate - effect * effect) / effective_n)
    rng = np.random.default_rng(seed)
    estimates = rng.normal(effect, standard_error, simulations)
    critical = null_margin + norm.ppf(1 - alpha) * standard_error
    return float(np.mean(estimates > critical))


def hypothesis_report(
    effects: Sequence[BootstrapEffect],
    adjusted_p: Mapping[str, float] | None = None,
    *,
    margin: float = 0.03,
) -> pd.DataFrame:
    rows = []
    for effect in effects:
        key = f"{effect.method}_vs_{effect.reference}"
        rows.append({
            **dataclasses.asdict(effect),
            "holm_p": adjusted_p.get(key) if adjusted_p else None,
            "margin": float(margin),
            "exceeds_margin": effect.ci_low > margin,
        })
    return pd.DataFrame(rows)

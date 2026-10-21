"""Publication figures for the SemUAV-VLM study.

The public entry point, :func:`generate_publication_figures`, accepts the
dataframes that are available at the current study phase.  Every figure is
fail-closed: a plot is skipped, with an explicit manifest reason, when its
source frame or required measured columns are absent.  No placeholder values
or synthetic rows are introduced by this module.

Figures are returned as live Matplotlib objects for inline notebook display and
are also written as 300-dpi PNG, vector PDF, and vector SVG files.  Development
and integration-fixture outputs receive a prominent NON-SCIENTIFIC watermark.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np
import pandas as pd
import seaborn as sns


FIGURE_MANIFEST_SCHEMA = "semuav.figure-manifest.v1"
OUTPUT_FORMATS = ("png", "pdf", "svg")
OUTPUT_DPI = 300
HYPOTHESIS_EFFECT_COLUMNS = (
    "hypothesis",
    "component_hypothesis",
    "comparison",
    "endpoint",
    "estimate",
    "ci_low",
    "ci_high",
    "holm_adjusted_p",
    "p_value_provenance",
    "effect_available",
    "source_phase",
    "source_table",
    "effect_provenance",
    "execution_status",
    "claim_supported",
    "effect_unit",
    "direction",
    "positive_favours",
    "unavailable_reason",
)

# Okabe-Ito: a well-tested colour-vision-deficiency-safe qualitative palette.
COLORBLIND_PALETTE = (
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#D55E00",  # vermillion
    "#F0E442",  # yellow
    "#000000",  # black
)

_FIXTURE_MARKERS = (
    "generated_fixture",
    "non_scientific",
    "non-scientific",
    "synthetic_integration",
    "mock_",
)


class _SkipFigure(RuntimeError):
    """Expected, auditable reason that a requested figure cannot be made."""

    def __init__(self, reason: str, sources: Sequence[str] = ()) -> None:
        super().__init__(reason)
        self.reason = str(reason)
        self.sources = tuple(str(source) for source in sources)


@dataclass(frozen=True)
class _FigureSpec:
    figure_id: str
    title: str
    builder: Callable[[Mapping[str, pd.DataFrame]], tuple[Figure, tuple[str, ...]]]


def _normalise_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _pretty(value: Any) -> str:
    text = str(value).replace("_", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text if re.match(r"^L[0-5]\b", text) else text.title()


def _wrapped(value: Any, width: int = 24) -> str:
    return "\n".join(textwrap.wrap(_pretty(value), width=width))


def _finite_numeric(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric))


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Return a stable hash of frame contents, labels, order, and dtypes."""

    payload = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [str(dtype) for dtype in frame.dtypes],
        "index_names": [str(name) if name is not None else None for name in frame.index.names],
        "data": json.loads(
            frame.to_json(
                orient="split",
                date_format="iso",
                date_unit="ns",
                double_precision=15,
                default_handler=lambda value: str(value),
            )
        ),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _select_frame(
    frames: Mapping[str, pd.DataFrame],
    aliases: Sequence[str],
    *,
    purpose: str,
) -> tuple[str, pd.DataFrame]:
    for alias in aliases:
        if alias in frames:
            frame = frames[alias]
            if frame.empty:
                raise _SkipFigure(
                    f"source frame '{alias}' is empty", (alias,),
                )
            return alias, frame
    normalised = {_normalise_key(key): key for key in frames}
    for alias in aliases:
        match = normalised.get(_normalise_key(alias))
        if match is not None:
            frame = frames[match]
            if frame.empty:
                raise _SkipFigure(
                    f"source frame '{match}' is empty", (match,),
                )
            return match, frame
    accepted = ", ".join(aliases)
    raise _SkipFigure(
        f"no {purpose} source frame is available; accepted keys: {accepted}",
    )


def _require_columns(name: str, frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise _SkipFigure(
            f"source frame '{name}' is missing required columns: {', '.join(missing)}",
            (name,),
        )


def _first_column(
    name: str,
    frame: pd.DataFrame,
    candidates: Sequence[str],
    *,
    role: str,
) -> str:
    for column in candidates:
        if column in frame.columns:
            return column
    raise _SkipFigure(
        f"source frame '{name}' has no {role} column; expected one of: "
        + ", ".join(candidates),
        (name,),
    )


def _nonempty_numeric(
    name: str,
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = _finite_numeric(result[column])
    result = result.dropna(subset=list(columns))
    if result.empty:
        raise _SkipFigure(
            f"source frame '{name}' has no finite rows for: {', '.join(columns)}",
            (name,),
        )
    return result


def _new_figure(
    *,
    nrows: int = 1,
    ncols: int = 1,
    width: float = 8.4,
    height: float = 5.1,
) -> tuple[Figure, Any]:
    with sns.axes_style("whitegrid"):
        figure, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(width, height),
            constrained_layout=True,
            squeeze=False,
        )
    return figure, axes


def _finish_axis(axis: Any, *, grid_axis: str = "y") -> None:
    axis.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.grid(False, axis="x" if grid_axis == "y" else "y")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(labelsize=8.5)
    axis.xaxis.label.set_size(9.5)
    axis.yaxis.label.set_size(9.5)
    axis.title.set_size(10.5)


def _annotate_bars(axis: Any, *, fraction: bool = False) -> None:
    for container in axis.containers:
        labels = []
        for patch in container:
            value = patch.get_height()
            if not np.isfinite(value):
                labels.append("")
            elif fraction:
                labels.append(f"{value:.0%}")
            elif abs(value) >= 10_000:
                labels.append(f"{value / 1_000:.0f}k")
            elif abs(value) >= 100:
                labels.append(f"{value:.0f}")
            else:
                labels.append(f"{value:.2g}")
        try:
            axis.bar_label(container, labels=labels, padding=2, fontsize=7)
        except (AttributeError, ValueError):
            continue


def _figure_dataset_balance(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("dataset_manifest", "manifest", "data_manifest", "dataset", "splits"),
        purpose="dataset/split",
    )
    _require_columns(name, frame, ("split",))
    plot = frame.loc[frame["split"].notna()].copy()
    if plot.empty:
        raise _SkipFigure(f"source frame '{name}' has no labelled split rows", (name,))

    grouping = next(
        (
            column
            for column in ("dataset", "dataset_name", "source_dataset", "cohort")
            if column in plot.columns and plot[column].notna().any()
        ),
        None,
    )
    include_groups = "visual_group_id" in plot.columns and plot["visual_group_id"].notna().any()
    ncols = 2 if include_groups else 1
    figure, axes = _new_figure(ncols=ncols, width=10.5 if include_groups else 7.3, height=4.6)
    palette = list(COLORBLIND_PALETTE)
    split_order = sorted(plot["split"].astype(str).unique())
    hue_order = sorted(plot[grouping].astype(str).unique()) if grouping else None
    hue_palette = (
        [palette[index % len(palette)] for index in range(len(hue_order))]
        if hue_order else None
    )
    first_count_kwargs = (
        {"hue": grouping, "hue_order": hue_order, "palette": hue_palette}
        if grouping else {"color": COLORBLIND_PALETTE[0]}
    )
    sns.countplot(
        data=plot,
        x="split",
        order=split_order,
        ax=axes[0, 0],
        **first_count_kwargs,
    )
    axes[0, 0].set_title("Observations per split")
    axes[0, 0].set_xlabel("Split")
    axes[0, 0].set_ylabel("Observations")
    axes[0, 0].tick_params(axis="x", rotation=20)
    _annotate_bars(axes[0, 0])
    _finish_axis(axes[0, 0])
    if grouping and axes[0, 0].get_legend() is not None:
        axes[0, 0].legend(title=_pretty(grouping), frameon=False, fontsize=8)

    if include_groups:
        group_columns = ["split", "visual_group_id"] + ([grouping] if grouping else [])
        group_plot = plot[group_columns].drop_duplicates()
        second_count_kwargs = (
            {"hue": grouping, "hue_order": hue_order, "palette": hue_palette}
            if grouping else {"color": COLORBLIND_PALETTE[2]}
        )
        sns.countplot(
            data=group_plot,
            x="split",
            order=split_order,
            ax=axes[0, 1],
            **second_count_kwargs,
        )
        axes[0, 1].set_title("Independent visual groups per split")
        axes[0, 1].set_xlabel("Split")
        axes[0, 1].set_ylabel("Visual groups")
        axes[0, 1].tick_params(axis="x", rotation=20)
        _annotate_bars(axes[0, 1])
        _finish_axis(axes[0, 1])
        if grouping and axes[0, 1].get_legend() is not None:
            axes[0, 1].get_legend().remove()
    figure.suptitle("Dataset and split balance", fontsize=14, fontweight="bold")
    return figure, (name,)


def _figure_exp1_metrics(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("exp1_summary", "experiment1_summary", "exp1_perception_summary"),
        purpose="Experiment 1 summary",
    )
    _require_columns(name, frame, ("model_id",))
    candidates = (
        "action_accuracy",
        "critical_action_recall",
        "semantic_valid_rate",
        "schema_valid_rate",
        "selective_action_accuracy",
        "grounded_hallucination_rate",
    )
    metrics = [column for column in candidates if column in frame.columns]
    if not metrics:
        raise _SkipFigure(
            f"source frame '{name}' contains no registered Experiment 1 accuracy/validity metrics",
            (name,),
        )
    plot = frame[["model_id", *metrics]].copy()
    for metric in metrics:
        plot[metric] = _finite_numeric(plot[metric])
    melted = plot.melt("model_id", var_name="metric", value_name="value").dropna()
    if melted.empty:
        raise _SkipFigure(f"source frame '{name}' has no finite Experiment 1 metrics", (name,))
    melted["metric_label"] = melted["metric"].map(lambda value: _wrapped(value, 19))
    figure, axes = _new_figure(width=10.5, height=5.2)
    axis = axes[0, 0]
    sns.barplot(
        data=melted,
        x="metric_label",
        y="value",
        hue="model_id",
        palette=COLORBLIND_PALETTE,
        errorbar=None,
        ax=axis,
    )
    if melted["value"].between(0, 1).all():
        axis.set_ylim(0, 1.05)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("Task, validity, and grounded-error endpoints")
    axis.set_xlabel("")
    axis.set_ylabel("Rate")
    axis.legend(
        title="Model", frameon=False, fontsize=8,
        ncols=min(2, melted["model_id"].nunique()),
        loc="upper center", bbox_to_anchor=(0.5, -0.17),
    )
    _finish_axis(axis)
    figure.suptitle("Experiment 1 — perception quality", fontsize=14, fontweight="bold")
    return figure, (name,)


def _figure_exp1_latency(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("exp1_summary", "experiment1_summary", "exp1_perception_summary"),
        purpose="Experiment 1 summary",
    )
    _require_columns(name, frame, ("model_id",))
    preferred = (
        "full_path_latency_p50_ms",
        "full_path_latency_p95_ms",
        "latency_p50_ms",
        "latency_p95_ms",
    )
    latency_columns = [column for column in preferred if column in frame.columns]
    # Do not plot both partial-path and full-path versions of the same quantile
    # when the scientifically complete full path is available.
    if "full_path_latency_p50_ms" in latency_columns:
        latency_columns = [column for column in latency_columns if column.startswith("full_path_")]
    if not latency_columns:
        raise _SkipFigure(
            f"source frame '{name}' has no Experiment 1 latency quantiles",
            (name,),
        )
    plot = frame[["model_id", *latency_columns]].copy()
    for column in latency_columns:
        plot[column] = _finite_numeric(plot[column])
    melted = plot.melt("model_id", var_name="quantile", value_name="latency_ms").dropna()
    melted = melted[melted["latency_ms"] >= 0]
    if melted.empty:
        raise _SkipFigure(f"source frame '{name}' has no finite nonnegative latency values", (name,))
    melted["quantile_label"] = melted["quantile"].map(
        lambda value: _pretty(value).replace("Full Path ", "")
    )
    figure, axes = _new_figure(width=8.5, height=4.8)
    axis = axes[0, 0]
    sns.barplot(
        data=melted,
        x="model_id",
        y="latency_ms",
        hue="quantile_label",
        palette=COLORBLIND_PALETTE,
        errorbar=None,
        ax=axis,
    )
    axis.set_title("Measured preprocessing-to-decoding latency")
    axis.set_xlabel("Model")
    axis.set_ylabel("Latency (ms)")
    axis.set_xticks(axis.get_xticks(), [_wrapped(label.get_text(), 18) for label in axis.get_xticklabels()])
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Quantile", frameon=False, fontsize=8, ncols=2,
                loc="upper center", bbox_to_anchor=(0.5, -0.24))
    _finish_axis(axis)
    figure.suptitle("Experiment 1 — latency", fontsize=14, fontweight="bold")
    return figure, (name,)


def _figure_exp2_pareto(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("exp2_frontier", "experiment2_frontier", "exp2_representation_frontier", "exp2"),
        purpose="Experiment 2 frontier",
    )
    x_column = _first_column(
        name, frame, ("wire_bytes", "total_wire_bytes", "payload_bytes"), role="wire-size",
    )
    y_column = _first_column(
        name,
        frame,
        (
            "native_receiver_correct",
            "correct_timely_action_probability",
            "correct_timely_action",
            "action_correct",
            "action_accuracy",
            "utility",
        ),
        role="receiver utility/correctness",
    )
    label_column = _first_column(
        name, frame, ("variant", "mode", "representation"), role="representation label",
    )
    grouping = "mode" if "mode" in frame.columns else label_column
    _require_columns(name, frame, ("sample_id",))
    columns = list(dict.fromkeys(["sample_id", label_column, grouping, x_column, y_column]))
    plot = _nonempty_numeric(name, frame[columns], (x_column, y_column))
    plot = plot[plot[x_column] > 0]
    if plot.empty:
        raise _SkipFigure(f"source frame '{name}' has no positive wire-size rows", (name,))
    aggregate_columns = list(dict.fromkeys([grouping, label_column]))
    # Clips exist only for MOT observations.  Comparing their means against
    # still representations averaged over DET+MOT confounds content with codec.
    # Use the exact intersection of observations across every shown candidate.
    supports = [set(group["sample_id"].astype(str)) for _, group in plot.groupby(
        aggregate_columns, sort=True, dropna=False,
    )]
    paired_ids = set.intersection(*supports) if supports else set()
    if not paired_ids:
        raise _SkipFigure(
            f"source frame '{name}' has no common paired sample cohort across representations",
            (name,),
        )
    plot = plot[plot["sample_id"].astype(str).isin(paired_ids)]
    if plot.duplicated(["sample_id", *aggregate_columns]).any():
        raise _SkipFigure(
            f"source frame '{name}' contains duplicate sample/representation rows",
            (name,),
        )
    summary = (
        plot.groupby(aggregate_columns, sort=True, dropna=False)[[x_column, y_column]]
        .mean()
        .reset_index()
        .sort_values([x_column, y_column], ascending=[True, False], kind="stable")
    )
    if summary.empty:
        raise _SkipFigure(f"source frame '{name}' has no plottable representations", (name,))
    running_best = -np.inf
    pareto_indices: list[int] = []
    for index, row in summary.iterrows():
        value = float(row[y_column])
        if value > running_best + 1e-12:
            pareto_indices.append(index)
            running_best = value
    pareto = summary.loc[pareto_indices].sort_values(x_column)

    figure, axes = _new_figure(width=8.6, height=5.4)
    axis = axes[0, 0]
    sns.scatterplot(
        data=summary,
        x=x_column,
        y=y_column,
        hue=grouping,
        palette=COLORBLIND_PALETTE,
        s=58,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
        ax=axis,
    )
    axis.plot(
        pareto[x_column], pareto[y_column], color="#222222", linewidth=1.4,
        marker="o", markersize=4, label="Pareto frontier", zorder=2,
    )
    # A numbered key avoids overlapping long labels when several codec
    # variants lie close together along the frontier.
    frontier_labels = []
    for number, row in enumerate(pareto.itertuples(index=False), start=1):
        axis.annotate(
            str(number), (getattr(row, x_column), getattr(row, y_column)),
            xytext=(0, 7 + 10 * (number % 2)), textcoords="offset points",
            ha="center", fontsize=7, fontweight="bold",
        )
        frontier_labels.append(f"{number}. {_pretty(getattr(row, label_column))}")
    if frontier_labels:
        figure.text(
            0.02, 0.01, textwrap.fill("Pareto points: " + " · ".join(frontier_labels), width=110),
            ha="left", va="bottom", fontsize=6.7,
        )
    positive = summary[x_column]
    if positive.max() / positive.min() >= 10:
        axis.set_xscale("log")
    if summary[y_column].between(0, 1).all():
        axis.set_ylim(0, 1.03)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title(
        "Matched-cohort receiver correctness versus representation size\n"
        f"Same {len(paired_ids):,} observations for every variant; means are descriptive"
    )
    axis.set_xlabel(_pretty(x_column) + (" (log scale)" if axis.get_xscale() == "log" else ""))
    axis.set_ylabel(_pretty(y_column))
    axis.legend(title=_pretty(grouping), frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    _finish_axis(axis, grid_axis="both")
    # Reserve deterministic room for the numbered key under the axis.
    figure.set_layout_engine(None)
    footer_lines = max(1, len(textwrap.wrap("Pareto points: " + " · ".join(frontier_labels), width=110)))
    figure.set_size_inches(10.2, 5.6 + 0.14 * footer_lines)
    figure.subplots_adjust(left=0.09, right=0.77, top=0.80, bottom=0.16 + 0.018 * footer_lines)
    figure.suptitle("Experiment 2 — representation Pareto frontier", fontsize=14, fontweight="bold")
    return figure, (name,)


def _figure_exp3_deadline_heatmap(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("exp3_summary", "experiment3_summary", "exp3_static_summary", "exp3_transport"),
        purpose="Experiment 3 transport",
    )
    _require_columns(name, frame, ("rate_mbps",))
    y_column = _first_column(
        name, frame, ("deadline_ms", "rtt_ms", "loss"), role="heatmap condition",
    )
    value_column = _first_column(
        name,
        frame,
        (
            "deadline_success_probability",
            "correct_timely_action_probability",
            "timely_delivery",
            "correct_timely_action",
        ),
        role="deadline-success",
    )
    plot = _nonempty_numeric(name, frame[["rate_mbps", y_column, value_column]], ("rate_mbps", y_column, value_column))
    grouped = plot.groupby([y_column, "rate_mbps"], dropna=False)[value_column].mean().reset_index()
    matrix = grouped.pivot(index=y_column, columns="rate_mbps", values=value_column).sort_index().sort_index(axis=1)
    if matrix.empty:
        raise _SkipFigure(f"source frame '{name}' has no complete Experiment 3 grid cells", (name,))
    figure, axes = _new_figure(width=max(7.0, 0.7 * len(matrix.columns) + 4), height=max(4.6, 0.5 * len(matrix.index) + 2.4))
    axis = axes[0, 0]
    annotate = matrix.size <= 36
    sns.heatmap(
        matrix,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=annotate,
        fmt=".2f",
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": _pretty(value_column)},
        ax=axis,
    )
    axis.set_title("Mean deadline success across available representations and link factors")
    axis.set_xlabel("Uplink rate (Mbps)")
    axis.set_ylabel(_pretty(y_column))
    axis.tick_params(axis="x", rotation=0)
    axis.tick_params(axis="y", rotation=0)
    figure.suptitle("Experiment 3 — deadline-success surface", fontsize=14, fontweight="bold")
    return figure, (name,)


def _exp4_summary(name: str, frame: pd.DataFrame) -> tuple[pd.DataFrame, str, str, str]:
    _require_columns(name, frame, ("policy",))
    utility = _first_column(
        name,
        frame,
        ("correct_timely_rate", "correct_timely_action", "utility", "delivery_rate"),
        role="policy utility",
    )
    wire = _first_column(
        name,
        frame,
        ("mean_wire_bytes", "total_wire_bytes", "wire_bytes_per_s", "wasted_uplink_wire_bytes"),
        role="wire resource",
    )
    energy = _first_column(
        name,
        frame,
        ("mean_energy_j", "energy_j", "energy_j_per_s", "total_energy_j"),
        role="energy resource",
    )
    columns = ["policy", utility, wire, energy]
    plot = frame[columns].copy()
    plot = _nonempty_numeric(name, plot, (utility, wire, energy))
    summary = plot.groupby("policy", sort=True)[[utility, wire, energy]].mean().reset_index()
    return summary, utility, wire, energy


def _figure_exp4_utility_resources(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("exp4_policy", "experiment4_policy", "exp4_trials", "exp4_summary", "exp4_resource_summary"),
        purpose="Experiment 4 policy",
    )
    summary, utility, wire, energy = _exp4_summary(name, frame)
    order = summary.sort_values(utility, kind="stable")["policy"].tolist()
    figure, axes = _new_figure(ncols=3, width=13.3, height=max(4.9, 0.34 * len(order) + 2.4))
    for index, (column, title, colour) in enumerate(
        (
            (utility, "Mission utility", COLORBLIND_PALETTE[2]),
            (wire, "Communication resource", COLORBLIND_PALETTE[0]),
            (energy, "Energy resource", COLORBLIND_PALETTE[1]),
        )
    ):
        axis = axes[0, index]
        sns.barplot(data=summary, y="policy", x=column, order=order, color=colour, errorbar=None, ax=axis)
        axis.set_title(title)
        axis.set_xlabel(_pretty(column))
        axis.set_ylabel("Policy" if index == 0 else "")
        if index > 0:
            axis.tick_params(labelleft=False)
        if column == utility and summary[column].between(0, 1).all():
            axis.set_xlim(0, 1.03)
            axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        positive = summary.loc[summary[column] > 0, column]
        if len(positive) and summary[column].max() / positive.min() >= 50:
            axis.set_xscale("log")
        _finish_axis(axis, grid_axis="x")
    figure.suptitle("Experiment 4 — policy utility and resource use", fontsize=14, fontweight="bold")
    return figure, (name,)


def _figure_exp4_mode_mix(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("exp4_policy", "experiment4_policy", "exp4_trials", "exp4_controller_summary", "exp4_summary"),
        purpose="Experiment 4 controller",
    )
    _require_columns(name, frame, ("policy",))
    if "selected_mode" in frame.columns:
        valid = frame.loc[frame["selected_mode"].notna(), ["policy", "selected_mode"]]
        if valid.empty:
            raise _SkipFigure(f"source frame '{name}' has no selected modes", (name,))
        mix = pd.crosstab(valid["policy"], valid["selected_mode"], normalize="index")
    else:
        frequency_columns = [column for column in frame if column.startswith("mode_frequency_")]
        if not frequency_columns:
            raise _SkipFigure(
                f"source frame '{name}' needs selected_mode or mode_frequency_* columns",
                (name,),
            )
        mix = frame.set_index("policy")[frequency_columns].copy()
        mix.columns = [column.removeprefix("mode_frequency_").upper() for column in mix.columns]
        mix = mix.apply(pd.to_numeric, errors="coerce")
        mix = mix.loc[mix.notna().any(axis=1)].fillna(0)
    if mix.empty:
        raise _SkipFigure(f"source frame '{name}' has no plottable mode frequencies", (name,))
    canonical = [f"L{number}_{suffix}" for number, suffix in enumerate(
        ("HEARTBEAT", "ALERT", "SEMANTICS", "SEMANTICS_ROI", "FEATURES", "IMAGE_OR_CLIP")
    )]
    ordered_columns = [column for column in canonical if column in mix.columns]
    ordered_columns += [column for column in mix.columns if column not in ordered_columns]
    mix = mix[ordered_columns]
    totals = mix.sum(axis=1)
    mix = mix.loc[totals > 0].div(totals[totals > 0], axis=0)
    if mix.empty:
        raise _SkipFigure(f"source frame '{name}' contains only zero mode frequencies", (name,))
    figure, axes = _new_figure(width=10.0, height=max(4.8, 0.42 * len(mix) + 2.5))
    axis = axes[0, 0]
    mix.plot(kind="barh", stacked=True, color=COLORBLIND_PALETTE[: len(mix.columns)], width=0.78, ax=axis)
    axis.set_xlim(0, 1)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("Representation selected at each controller decision")
    axis.set_xlabel("Decision share")
    axis.set_ylabel("Policy")
    axis.legend(title="Mode", frameon=False, fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    _finish_axis(axis, grid_axis="x")
    figure.suptitle("Experiment 4 — adaptive representation mix", fontsize=14, fontweight="bold")
    return figure, (name,)


def _figure_exp5_safety(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        ("exp5_summary", "experiment5_summary", "exp5_risk_uncertainty_summary", "exp5_detail"),
        purpose="Experiment 5 safety",
    )
    _require_columns(name, frame, ("subset", "policy", "critical_miss_rate", "false_alert_rate"))
    metrics = ["critical_miss_rate", "false_alert_rate"]
    if "high_confidence_semantic_error_rate" in frame.columns:
        metrics.append("high_confidence_semantic_error_rate")
    working = frame[["subset", "policy", *metrics]].copy()
    for metric in metrics:
        working[metric] = _finite_numeric(working[metric])
    if not working[metrics].notna().any().any():
        raise _SkipFigure(f"source frame '{name}' has no finite safety rates", (name,))
    figure, axes = _new_figure(ncols=len(metrics), width=4.4 * len(metrics), height=max(4.7, 0.38 * working["policy"].nunique() + 2.6))
    policy_order = sorted(working["policy"].dropna().unique())
    subset_order = sorted(working["subset"].dropna().unique())
    for index, metric in enumerate(metrics):
        matrix = (
            working.pivot_table(index="policy", columns="subset", values=metric, aggfunc="mean")
            .sort_index()
            .reindex(index=policy_order, columns=subset_order)
        )
        axis = axes[0, index]
        if not matrix.notna().any().any():
            axis.set_axis_off()
            axis.set_title(_wrapped(metric, 20))
            axis.text(0.5, 0.5, "Not estimable\n(no eligible denominator)",
                      ha="center", va="center", transform=axis.transAxes, color="#666666")
            continue
        sns.heatmap(
            matrix,
            cmap="mako_r",
            vmin=0,
            vmax=1,
            annot=matrix.size <= 30,
            fmt=".2f",
            linewidths=0.4,
            linecolor="white",
            cbar=index == len(metrics) - 1,
            cbar_kws={"label": "Rate"} if index == len(metrics) - 1 else None,
            ax=axis,
        )
        axis.set_title(_wrapped(metric, 20))
        axis.set_xlabel("Cohort")
        axis.set_ylabel("Policy" if index == 0 else "")
        axis.tick_params(axis="x", rotation=35)
        axis.tick_params(axis="y", rotation=0, labelleft=index == 0)
        axis.set_facecolor("#eeeeee")
    figure.suptitle("Experiment 5 — safety and uncertainty endpoints", fontsize=14, fontweight="bold")
    return figure, (name,)


def _figure_exp6_protocol(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        (
            "exp6_campaign_summary",
            "exp6_summary",
            "experiment6_campaign_summary",
            "exp6_robustness",
            "experiment6_robustness",
            "exp6_state_strategy_summary",
        ),
        purpose="Experiment 6 protocol",
    )
    label = _first_column(name, frame, ("case", "strategy", "gate"), role="protocol-case label")
    value = _first_column(
        name,
        frame,
        ("pass_rate", "passed", "all_passed", "valid_state_delivery_rate", "restoration_success_rate"),
        role="protocol pass-rate",
    )
    working = frame[[label, value]].copy()
    if pd.api.types.is_bool_dtype(working[value].dtype):
        working[value] = working[value].astype(float)
    working[value] = _finite_numeric(working[value])
    working = working.dropna(subset=[label, value])
    if working.empty:
        raise _SkipFigure(f"source frame '{name}' has no finite protocol outcomes", (name,))
    summary = working.groupby(label, sort=True)[value].mean().sort_values()
    colours = [
        COLORBLIND_PALETTE[2] if rate >= 1.0 - 1e-12 else COLORBLIND_PALETTE[1] if rate >= 0.8 else COLORBLIND_PALETTE[5]
        for rate in summary
    ]
    figure, axes = _new_figure(width=9.2, height=max(4.8, 0.36 * len(summary) + 2.3))
    axis = axes[0, 0]
    axis.barh([_wrapped(item, 35) for item in summary.index], summary.values, color=colours)
    axis.axvline(1.0, color="#333333", linestyle="--", linewidth=0.9)
    axis.set_xlim(0, 1.03)
    axis.xaxis.set_major_formatter(PercentFormatter(1.0))
    axis.set_title("Registered receiver/transport fault outcomes")
    axis.set_xlabel(_pretty(value))
    axis.set_ylabel(_pretty(label))
    _finish_axis(axis, grid_axis="x")
    figure.suptitle("Experiment 6 — protocol robustness", fontsize=14, fontweight="bold")
    return figure, (name,)


def _hardware_frame_is_valid(name: str, frame: pd.DataFrame) -> None:
    _require_columns(name, frame, ("hardware_id", "platform_class", "policy"))
    for column in ("hardware_id", "platform_class"):
        values = frame[column].dropna().astype(str).str.strip()
        if values.empty or values.eq("").any():
            raise _SkipFigure(f"source frame '{name}' has blank {column} values", (name,))
        lowered = " ".join(values.str.lower().unique())
        if "replace_with" in lowered or "placeholder" in lowered or "example" in lowered:
            raise _SkipFigure(
                f"source frame '{name}' contains placeholder {column} values; hardware plot is blocked",
                (name,),
            )


def _figure_exp7_readiness_hardware(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    hardware_aliases = (
        "exp7_hardware",
        "experiment7_hardware",
        "embedded_trials",
        "embedded_records",
        "embedded_metrics",
    )
    hardware_name = next((alias for alias in hardware_aliases if alias in frames), None)
    if hardware_name is not None and not frames[hardware_name].empty:
        frame = frames[hardware_name]
        _hardware_frame_is_valid(hardware_name, frame)
        metrics = [
            column
            for column in ("latency_ms", "energy_j", "total_wire_bytes", "memory_mb", "temperature_c")
            if column in frame.columns
        ]
        if len(metrics) < 2:
            raise _SkipFigure(
                f"source frame '{hardware_name}' needs at least two measured hardware metrics",
                (hardware_name,),
            )
        working = frame[["policy", *metrics]].copy()
        for metric in metrics:
            working[metric] = _finite_numeric(working[metric])
        metrics = [metric for metric in metrics if working[metric].notna().any()]
        if len(metrics) < 2:
            raise _SkipFigure(
                f"source frame '{hardware_name}' has fewer than two finite hardware metrics",
                (hardware_name,),
            )
        ncols = 2
        nrows = math.ceil(len(metrics) / ncols)
        figure, axes = _new_figure(nrows=nrows, ncols=ncols, width=10.5, height=3.8 * nrows)
        for index, metric in enumerate(metrics):
            axis = axes.flat[index]
            sns.boxplot(
                data=working,
                x=metric,
                y="policy",
                color=COLORBLIND_PALETTE[index % len(COLORBLIND_PALETTE)],
                fliersize=2,
                linewidth=1,
                ax=axis,
            )
            axis.set_title(_pretty(metric))
            axis.set_xlabel(_pretty(metric))
            axis.set_ylabel("Policy")
            _finish_axis(axis, grid_axis="x")
        for index in range(len(metrics), nrows * ncols):
            axes.flat[index].set_visible(False)
        figure.suptitle("Experiment 7 — measured target-hardware performance", fontsize=14, fontweight="bold")
        return figure, (hardware_name,)

    name, frame = _select_frame(
        frames,
        ("exp7_readiness", "experiment7_readiness", "environment_report", "readiness"),
        purpose="Experiment 7 readiness",
    )
    _require_columns(name, frame, ("item", "status"))
    statuses = frame[["item", "status"]].dropna().copy()
    statuses["status"] = statuses["status"].astype(str).str.upper()
    allowed = {"PASS", "FAIL", "SKIP"}
    unknown = sorted(set(statuses["status"]) - allowed)
    if unknown:
        raise _SkipFigure(
            f"source frame '{name}' has unsupported readiness statuses: {', '.join(unknown)}",
            (name,),
        )
    if statuses.empty:
        raise _SkipFigure(f"source frame '{name}' has no readiness checks", (name,))
    colour_map = {"PASS": COLORBLIND_PALETTE[2], "FAIL": COLORBLIND_PALETTE[5], "SKIP": "#999999"}
    statuses = statuses.sort_values(["status", "item"], kind="stable")
    values = statuses["status"].map({"FAIL": 0, "SKIP": 0.5, "PASS": 1}).astype(float)
    figure, axes = _new_figure(width=9.0, height=max(4.8, 0.32 * len(statuses) + 2.5))
    axis = axes[0, 0]
    axis.barh(
        [_wrapped(item, 34) for item in statuses["item"]],
        np.ones(len(statuses)),
        color=[colour_map[status] for status in statuses["status"]],
    )
    for position, (value, status) in enumerate(zip(values, statuses["status"])):
        axis.text(0.02, position, status, va="center", ha="left", color="white" if status != "SKIP" else "black", fontweight="bold", fontsize=8)
    axis.set_xlim(0, 1)
    axis.set_xticks([])
    axis.set_title("Runtime and external-evidence readiness checks")
    axis.set_xlabel("")
    axis.set_ylabel("Check")
    _finish_axis(axis, grid_axis="x")
    figure.suptitle("Experiment 7 — readiness", fontsize=14, fontweight="bold")
    return figure, (name,)


def _strict_optional_boolean(value: Any, *, field: str) -> bool | None:
    if value is None or pd.isna(value):
        return None
    normalized = value.strip().lower() if isinstance(value, str) else value
    mapping = {
        True: True,
        False: False,
        1: True,
        0: False,
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    if normalized not in mapping:
        raise ValueError(f"{field} must contain strict boolean values")
    return mapping[normalized]


def build_hypothesis_effects_frame(
    *,
    h1_report: pd.DataFrame | None = None,
    h2_report: pd.DataFrame | None = None,
    h3_report: pd.DataFrame | None = None,
    h4_report: pd.DataFrame | None = None,
    hypothesis_status: pd.DataFrame | None = None,
    embedded_retest_effects: pd.DataFrame | None = None,
    embedded_retest_status: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Standardize registered H1--H5 evidence without inventing an H5 effect.

    H1--H4 have scalar, paired-bootstrap estimands in the test analysis.  H5 is
    deliberately different: it is the registered conjunction of original and
    target-hardware H2--H4 support plus hardware gates.  No scalar H5 estimand
    or confidence interval is registered.  Once H5 is empirically complete,
    its verified embedded H2--H4 effects are retained as explicitly labelled
    H5 replication-component rows.  Before that point, this function emits an
    explicit H5 status row with ``effect_available=False`` and missing numeric
    values instead of converting a pass/fail status into a fake effect size.

    Embedded H2--H4 estimates are accepted only together with the status table
    produced by :func:`semuav.hardware.evaluate_embedded_retest`.  That table
    proves the supplied summary values were reconciled against the hashed raw
    target-stage trials.  Unverified embedded summaries are rejected.
    """

    named_frames = {
        "h1_report": h1_report,
        "h2_report": h2_report,
        "h3_report": h3_report,
        "h4_report": h4_report,
        "hypothesis_status": hypothesis_status,
        "embedded_retest_effects": embedded_retest_effects,
        "embedded_retest_status": embedded_retest_status,
    }
    for name, frame in named_frames.items():
        if frame is not None and not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame or None")

    status_lookup: dict[str, dict[str, Any]] = {}
    if hypothesis_status is not None and not hypothesis_status.empty:
        required = {"hypothesis", "execution_status", "claim_supported"}
        if missing := required.difference(hypothesis_status.columns):
            raise ValueError(
                "hypothesis_status is missing required columns: "
                + ", ".join(sorted(missing))
            )
        status = hypothesis_status.copy()
        status["hypothesis"] = status["hypothesis"].astype(str).str.upper().str.strip()
        if status["hypothesis"].duplicated().any():
            raise ValueError("hypothesis_status must contain at most one row per hypothesis")
        for row in status.itertuples(index=False):
            status_lookup[str(row.hypothesis)] = {
                "execution_status": str(row.execution_status),
                "claim_supported": _strict_optional_boolean(
                    row.claim_supported,
                    field=f"hypothesis_status[{row.hypothesis}].claim_supported",
                ),
            }

    def status_fields(hypothesis: str) -> dict[str, Any]:
        return status_lookup.get(
            hypothesis,
            {"execution_status": "STATUS_NOT_SUPPLIED", "claim_supported": None},
        )

    def report_row_status(hypothesis: str, row: Any) -> dict[str, Any]:
        default = status_fields(hypothesis)
        return {
            "execution_status": str(
                getattr(row, "execution_status", default["execution_status"])
            ),
            "claim_supported": _strict_optional_boolean(
                getattr(row, "claim_supported", default["claim_supported"]),
                field=f"{hypothesis} report claim_supported",
            ),
        }

    rows: list[dict[str, Any]] = []

    def append_effect(
        *,
        hypothesis: str,
        component_hypothesis: str,
        comparison: str,
        endpoint: str,
        estimate: Any,
        ci_low: Any,
        ci_high: Any,
        holm_adjusted_p: Any = np.nan,
        p_value_provenance: str = "",
        source_phase: str,
        source_table: str,
        provenance: str,
        positive_favours: str,
        row_status: Mapping[str, Any] | None = None,
        unavailable_reason: str = "",
    ) -> None:
        state = dict(row_status or status_fields(hypothesis))
        not_estimable = (
            str(state["execution_status"]).upper().strip() == "NOT_ESTIMABLE"
        )
        numeric = pd.to_numeric(
            pd.Series([estimate, ci_low, ci_high], dtype="object"), errors="coerce",
        ).to_numpy(float)
        finite = bool(np.isfinite(numeric).all()) and not not_estimable
        if finite and not numeric[1] <= numeric[0] <= numeric[2]:
            raise ValueError(
                f"{hypothesis}/{comparison} confidence interval does not contain its estimate"
            )
        adjusted_p = pd.to_numeric(
            pd.Series([holm_adjusted_p], dtype="object"), errors="coerce",
        ).to_numpy(float)[0]
        if np.isfinite(adjusted_p) and not 0.0 <= adjusted_p <= 1.0:
            raise ValueError(
                f"{hypothesis}/{comparison} Holm-adjusted p-value must be in [0, 1]"
            )
        # An unestimated component has no inferential p-value.  Some upstream
        # claim ledgers use 1.0 as a fail-closed sentinel; it must not be
        # presented as an observed p-value in the publication effect table.
        if not finite:
            adjusted_p = np.nan
        rows.append({
            "hypothesis": hypothesis,
            "component_hypothesis": component_hypothesis,
            "comparison": str(comparison),
            "endpoint": str(endpoint),
            "estimate": float(numeric[0]) if finite else np.nan,
            "ci_low": float(numeric[1]) if finite else np.nan,
            "ci_high": float(numeric[2]) if finite else np.nan,
            "holm_adjusted_p": (
                float(adjusted_p) if np.isfinite(adjusted_p) else np.nan
            ),
            "p_value_provenance": (
                str(p_value_provenance) if np.isfinite(adjusted_p) else ""
            ),
            "effect_available": finite,
            "source_phase": str(source_phase),
            "source_table": str(source_table),
            "effect_provenance": str(provenance),
            "execution_status": str(state["execution_status"]),
            "claim_supported": state["claim_supported"],
            "effect_unit": "absolute_rate_difference",
            "direction": "positive_favours_semuav",
            "positive_favours": str(positive_favours),
            "unavailable_reason": (
                "" if finite else (
                    str(unavailable_reason).strip()
                    or f"{hypothesis} estimate or confidence interval is unavailable in {source_table}"
                )
            ),
        })

    def append_unavailable(hypothesis: str, reason: str) -> None:
        state = status_fields(hypothesis)
        rows.append({
            "hypothesis": hypothesis,
            "component_hypothesis": hypothesis,
            "comparison": "",
            "endpoint": "",
            "estimate": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "holm_adjusted_p": np.nan,
            "p_value_provenance": "",
            "effect_available": False,
            "source_phase": "status",
            "source_table": "hypothesis_status" if hypothesis in status_lookup else "",
            "effect_provenance": "status_only_no_scalar_effect",
            "execution_status": str(state["execution_status"]),
            "claim_supported": state["claim_supported"],
            "effect_unit": "",
            "direction": "",
            "positive_favours": "",
            "unavailable_reason": str(reason),
        })

    if h1_report is None or h1_report.empty:
        append_unavailable("H1", "H1 test effect report is unavailable")
    else:
        required = {
            "semantic_mode", "image_variant", "utility_difference",
            "utility_ci_low", "utility_ci_high",
        }
        if missing := required.difference(h1_report.columns):
            raise ValueError(
                "h1_report is missing required effect columns: "
                + ", ".join(sorted(missing))
            )
        for row in h1_report.itertuples(index=False):
            append_effect(
                hypothesis="H1",
                component_hypothesis="H1",
                comparison=f"{row.semantic_mode}_vs_{row.image_variant}",
                endpoint="correct_timely_action_difference",
                estimate=row.utility_difference,
                ci_low=row.utility_ci_low,
                ci_high=row.utility_ci_high,
                source_phase="test",
                source_table="h1_semantic_noninferiority",
                provenance="two_way_clustered_paired_bootstrap",
                positive_favours=str(row.semantic_mode),
                row_status=report_row_status("H1", row),
                unavailable_reason=str(getattr(row, "not_estimable_reason", "")),
            )

    if h2_report is None or h2_report.empty:
        append_unavailable("H2", "H2 confirmatory effect report is unavailable")
    else:
        required = {"method", "reference", "endpoint", "estimate", "ci_low", "ci_high"}
        if missing := required.difference(h2_report.columns):
            raise ValueError(
                "h2_report is missing required effect columns: "
                + ", ".join(sorted(missing))
            )
        for row in h2_report.itertuples(index=False):
            append_effect(
                hypothesis="H2",
                component_hypothesis="H2",
                comparison=f"{row.method}_vs_{row.reference}",
                endpoint=row.endpoint,
                estimate=row.estimate,
                ci_low=row.ci_low,
                ci_high=row.ci_high,
                holm_adjusted_p=getattr(
                    row, "holm_p", getattr(row, "holm_adjusted_p", np.nan),
                ),
                p_value_provenance="holm_family_adjusted_two_sided_p",
                source_phase="test",
                source_table="h2_confirmatory_report",
                provenance="two_way_clustered_paired_bootstrap",
                positive_favours=str(row.method),
                row_status=report_row_status("H2", row),
                unavailable_reason=str(getattr(row, "not_estimable_reason", "")),
            )

    if h3_report is None or h3_report.empty:
        append_unavailable("H3", "H3 mission-awareness effect report is unavailable")
    else:
        required = {"critical_miss_reduction", "ci_low", "ci_high"}
        if missing := required.difference(h3_report.columns):
            raise ValueError(
                "h3_report is missing required effect columns: "
                + ", ".join(sorted(missing))
            )
        for row in h3_report.itertuples(index=False):
            append_effect(
                hypothesis="H3",
                component_hypothesis="H3",
                comparison="proposed_full_vs_channel_only_hysteresis",
                endpoint="critical_miss_reduction",
                estimate=row.critical_miss_reduction,
                ci_low=row.ci_low,
                ci_high=row.ci_high,
                holm_adjusted_p=getattr(row, "holm_adjusted_p", np.nan),
                p_value_provenance="holm_family_adjusted_two_sided_p",
                source_phase="test",
                source_table="h3_mission_awareness",
                provenance="two_way_clustered_paired_bootstrap",
                positive_favours="proposed_full",
                row_status=report_row_status("H3", row),
                unavailable_reason=str(getattr(row, "not_estimable_reason", "")),
            )

    h4_components = (
        {
            "component_hypothesis": "H4_MATCHED_COVERAGE_SELECTIVE_RISK",
            "endpoint": "matched_coverage_selective_risk_reduction",
            "estimate": "matched_coverage_error_reduction",
            "ci_low": "matched_coverage_ci_low",
            "ci_high": "matched_coverage_ci_high",
            "p_value": "matched_coverage_holm_adjusted_p",
            "provenance": (
                "two_way_clustered_paired_bootstrap_registered_"
                "matched_coverage_selective_risk"
            ),
        },
        {
            "component_hypothesis": "H4_HIGH_CONFIDENCE_SEMANTIC_ERROR",
            "endpoint": "high_confidence_semantic_error_reduction",
            "estimate": "high_confidence_error_reduction",
            "ci_low": "high_confidence_ci_low",
            "ci_high": "high_confidence_ci_high",
            "p_value": "high_confidence_holm_adjusted_p",
            "provenance": (
                "two_way_clustered_paired_bootstrap_registered_"
                "high_confidence_semantic_error"
            ),
        },
    )
    h4_comparison = "proposed_full_vs_ablate_evidence_escalation"
    if h4_report is None or h4_report.empty:
        # Keep both registered H4 estimands visible in the publication table,
        # even before a complete test report exists.  Collapsing these into one
        # generic H4 status row would obscure the conjunctive claim definition.
        for component in h4_components:
            append_effect(
                hypothesis="H4",
                component_hypothesis=str(component["component_hypothesis"]),
                comparison=h4_comparison,
                endpoint=str(component["endpoint"]),
                estimate=np.nan,
                ci_low=np.nan,
                ci_high=np.nan,
                holm_adjusted_p=np.nan,
                p_value_provenance="",
                source_phase="status",
                source_table=(
                    "hypothesis_status" if "H4" in status_lookup else ""
                ),
                provenance="status_only_no_scalar_effect",
                positive_favours="proposed_full",
            )
            rows[-1]["unavailable_reason"] = (
                f"{component['endpoint']} is unavailable because the H4 "
                "uncertainty-escalation effect report is unavailable"
            )
    else:
        required = {
            str(component[field])
            for component in h4_components
            for field in ("estimate", "ci_low", "ci_high", "p_value")
        }
        if missing := required.difference(h4_report.columns):
            raise ValueError(
                "h4_report is missing required component-specific effect columns: "
                + ", ".join(sorted(missing))
            )
        if len(h4_report) != 1:
            raise ValueError(
                "h4_report must contain exactly one conjunctive H4 report row"
            )
        row = next(h4_report.itertuples(index=False))
        h4_status = report_row_status("H4", row)
        not_estimable_reason = str(getattr(row, "not_estimable_reason", "")).strip()
        for component in h4_components:
            append_effect(
                hypothesis="H4",
                component_hypothesis=str(component["component_hypothesis"]),
                comparison=h4_comparison,
                endpoint=str(component["endpoint"]),
                estimate=getattr(row, str(component["estimate"])),
                ci_low=getattr(row, str(component["ci_low"])),
                ci_high=getattr(row, str(component["ci_high"])),
                holm_adjusted_p=getattr(row, str(component["p_value"])),
                p_value_provenance="holm_family_adjusted_two_sided_p",
                source_phase="test",
                source_table="h4_uncertainty_escalation",
                provenance=str(component["provenance"]),
                positive_favours="proposed_full",
                row_status=h4_status,
                unavailable_reason=not_estimable_reason,
            )

    if (embedded_retest_effects is None) != (embedded_retest_status is None):
        raise ValueError(
            "embedded_retest_effects and embedded_retest_status must be supplied together"
        )
    if embedded_retest_effects is not None and embedded_retest_status is not None:
        required_status = {
            "hypothesis", "claim_supported", "raw_recomputed", "source_kind",
        }
        if missing := required_status.difference(embedded_retest_status.columns):
            raise ValueError(
                "embedded_retest_status is missing verification columns: "
                + ", ".join(sorted(missing))
            )
        verified_status = embedded_retest_status.copy()
        verified_status["hypothesis"] = (
            verified_status["hypothesis"].astype(str).str.upper().str.strip()
        )
        if (
            set(verified_status["hypothesis"]) != {"H2", "H3", "H4"}
            or verified_status["hypothesis"].duplicated().any()
        ):
            raise ValueError("embedded_retest_status must contain exactly H2, H3, and H4")
        raw_flags = [
            _strict_optional_boolean(value, field="embedded_retest_status.raw_recomputed")
            for value in verified_status["raw_recomputed"]
        ]
        if not all(flag is True for flag in raw_flags):
            raise ValueError("embedded effects require raw_recomputed=True for H2--H4")
        if not verified_status["source_kind"].astype(str).eq(
            "hashed_raw_stage_trial_recomputation"
        ).all():
            raise ValueError(
                "embedded effects require hashed raw-stage recomputation provenance"
            )
        embedded_status_lookup = {}
        for row in verified_status.itertuples(index=False):
            embedded_status_lookup[str(row.hypothesis)] = {
                "execution_status": str(
                    getattr(row, "execution_status", "EMBEDDED_RETEST_STATUS_AVAILABLE")
                ),
                "claim_supported": _strict_optional_boolean(
                    row.claim_supported,
                    field=f"embedded_retest_status[{row.hypothesis}].claim_supported",
                ),
            }
        required_effect = {
            "hypothesis", "comparison", "endpoint", "estimate", "ci_low", "ci_high",
        }
        if missing := required_effect.difference(embedded_retest_effects.columns):
            raise ValueError(
                "embedded_retest_effects is missing required effect columns: "
                + ", ".join(sorted(missing))
            )
        embedded = embedded_retest_effects.copy()
        embedded["hypothesis"] = embedded["hypothesis"].astype(str).str.upper().str.strip()
        if not set(embedded["hypothesis"]).issubset({"H2", "H3", "H4"}):
            raise ValueError("embedded_retest_effects may contain only H2, H3, and H4")
        if embedded.duplicated(["hypothesis", "comparison"]).any():
            raise ValueError("embedded_retest_effects contains duplicate comparison rows")
        h5_state = status_fields("H5")
        h5_complete = h5_state["execution_status"] == "EMPIRICAL_COMPLETE"
        if h5_complete:
            for row in embedded.itertuples(index=False):
                append_effect(
                    hypothesis="H5",
                    component_hypothesis=str(row.hypothesis),
                    comparison=str(row.comparison),
                    endpoint=str(row.endpoint),
                    estimate=row.estimate,
                    ci_low=row.ci_low,
                    ci_high=row.ci_high,
                    holm_adjusted_p=getattr(row, "holm_adjusted_p", np.nan),
                    p_value_provenance=(
                        "holm_family_adjusted_two_sided_p"
                        if hasattr(row, "holm_adjusted_p") else ""
                    ),
                    source_phase="embedded",
                    source_table="embedded_retest_effects",
                    provenance="verified_hashed_raw_stage_trial_recomputation",
                    positive_favours="proposed_full",
                    row_status=h5_state,
                )

    if not any(row["hypothesis"] == "H5" and row["effect_available"] for row in rows):
        h5_execution = status_fields("H5")["execution_status"]
        append_unavailable(
            "H5",
            (
                "H5 has no standalone scalar estimand; its forest entries are the verified "
                "embedded H2-H4 replication components and are unavailable until H5 is "
                f"EMPIRICAL_COMPLETE (current status: {h5_execution})"
            ),
        )
    result = pd.DataFrame(rows, columns=HYPOTHESIS_EFFECT_COLUMNS)
    result["_hypothesis_order"] = (
        result["hypothesis"].str.removeprefix("H").astype(int)
    )
    result["_phase_order"] = result["source_phase"].map(
        {"test": 0, "embedded": 1, "status": 2}
    ).fillna(3)
    result = result.sort_values(
        ["_hypothesis_order", "_phase_order", "comparison"], kind="stable",
    ).drop(columns=["_hypothesis_order", "_phase_order"]).reset_index(drop=True)
    return result


def _figure_hypothesis_forest(
    frames: Mapping[str, pd.DataFrame],
) -> tuple[Figure, tuple[str, ...]]:
    name, frame = _select_frame(
        frames,
        (
            "hypothesis_effects",
            "hypothesis_report",
            "confirmatory_report",
            "embedded_retest_summary",
            "hypotheses",
        ),
        purpose="H1-H5 effect",
    )
    _require_columns(name, frame, ("hypothesis", "estimate"))
    low = _first_column(name, frame, ("ci_low", "ci_lower", "lower_ci"), role="confidence-interval lower bound")
    high = _first_column(name, frame, ("ci_high", "ci_upper", "upper_ci"), role="confidence-interval upper bound")
    comparison = next((column for column in ("comparison", "reference", "endpoint") if column in frame.columns), None)
    extra_columns = [
        column
        for column in (
            "component_hypothesis", "effect_available", "source_phase", "execution_status",
            "unavailable_reason", "endpoint", "holm_adjusted_p", "p_value_provenance",
        )
        if column in frame.columns
    ]
    columns = list(dict.fromkeys(
        ["hypothesis", "estimate", low, high, *extra_columns]
        + ([comparison] if comparison else [])
    ))
    working = frame[columns].copy()
    working["hypothesis"] = working["hypothesis"].astype(str).str.upper().str.strip()
    for column in ("estimate", low, high):
        working[column] = _finite_numeric(working[column])
    finite_effect = working[["estimate", low, high]].notna().all(axis=1)
    if "effect_available" in working:
        declared_available = working["effect_available"].map(
            lambda value: _strict_optional_boolean(value, field="effect_available")
        ).fillna(False).astype(bool)
        if (declared_available & ~finite_effect).any():
            raise _SkipFigure(
                f"source frame '{name}' declares an available effect without finite estimate/CI values",
                (name,),
            )
        available_mask = declared_available & finite_effect
    else:
        available_mask = finite_effect
    effects = working.loc[available_mask].copy()
    required_hypotheses = {"H1", "H2", "H3", "H4"}
    observed = set(effects["hypothesis"])
    missing_hypotheses = sorted(required_hypotheses - observed)
    unavailable_statuses: list[dict[str, str]] = []
    for hypothesis in missing_hypotheses:
        status_rows = working[working["hypothesis"].eq(hypothesis)]
        if status_rows.empty or "execution_status" not in status_rows:
            raise _SkipFigure(
                "confirmatory forest lacks an effect or explicit status for: " + hypothesis,
                (name,),
            )
        statuses = set(status_rows["execution_status"].astype(str).str.upper())
        if not statuses or not statuses.issubset({"NOT_ESTIMABLE", "NOT_APPLICABLE", "BLOCKED", "FAILED"}):
            raise _SkipFigure(
                "confirmatory forest has missing effects without an unavailable status for: " + hypothesis,
                (name,),
            )
        reasons = sorted(set(status_rows.get(
            "unavailable_reason", pd.Series("No estimable effect", index=status_rows.index),
        ).dropna().astype(str)))
        unavailable_statuses.append({
            "hypothesis": hypothesis,
            "status": ", ".join(sorted(statuses)),
            "reason": "; ".join(reason for reason in reasons if reason),
        })
    effects = effects[effects["hypothesis"].isin({"H1", "H2", "H3", "H4", "H5"})].copy()
    effects["hypothesis_order"] = effects["hypothesis"].str.removeprefix("H").astype(int)
    sort_columns = ["hypothesis_order"]
    if "source_phase" in effects:
        effects["source_phase_order"] = effects["source_phase"].map(
            {"test": 0, "embedded": 1}
        ).fillna(2)
        sort_columns.append("source_phase_order")
    if comparison:
        sort_columns.append(comparison)
    effects = effects.sort_values(sort_columns, ascending=True, kind="stable")
    effects["label"] = effects["hypothesis"]
    if "component_hypothesis" in effects:
        distinct_component = (
            effects["component_hypothesis"].astype(str).str.upper().str.strip()
            != effects["hypothesis"]
        )
        replicated_component = distinct_component & effects["hypothesis"].eq("H5")
        effects.loc[replicated_component, "label"] += (
            " (" + effects.loc[replicated_component, "component_hypothesis"].astype(str)
            + " replication component)"
        )
        registered_component = distinct_component & ~effects["hypothesis"].eq("H5")
        component_labels = []
        for hypothesis, component in zip(
            effects.loc[registered_component, "hypothesis"].astype(str),
            effects.loc[registered_component, "component_hypothesis"].astype(str),
        ):
            prefix = f"{hypothesis}_"
            component_labels.append(
                _pretty(component[len(prefix):] if component.startswith(prefix) else component)
            )
        effects.loc[registered_component, "label"] += (
            " (" + pd.Series(
                component_labels,
                index=effects.index[registered_component],
                dtype="object",
            ) + ")"
        )
    if "source_phase" in effects and effects["source_phase"].nunique() > 1:
        effects["label"] += " [" + effects["source_phase"].astype(str).map(_pretty) + "]"
    if comparison:
        effects["label"] += " — " + effects[comparison].astype(str).map(lambda value: _pretty(value))
    if (effects[low] > effects["estimate"]).any() or (effects[high] < effects["estimate"]).any():
        raise _SkipFigure(
            f"source frame '{name}' has confidence bounds that do not contain their estimates",
            (name,),
        )
    h5_has_effect = bool(effects["hypothesis"].eq("H5").any())
    if not h5_has_effect:
        h5_rows = working[working["hypothesis"].eq("H5")]
        h5_row = h5_rows.iloc[0] if len(h5_rows) else {}
        unavailable_statuses.append({
            "hypothesis": "H5",
            "status": str(h5_row.get("execution_status", "STATUS_NOT_SUPPLIED")),
            "reason": str(h5_row.get("unavailable_reason", "No H5 scalar effect or status row was supplied")),
        })
    rows: list[dict[str, Any]] = [
        {"hypothesis": row["hypothesis"], "label": row["label"], "effect": row}
        for _, row in effects.iterrows()
    ] + [
        {"hypothesis": row["hypothesis"],
         "label": row["hypothesis"] + (" — target-hardware replication" if row["hypothesis"] == "H5" else " — effect unavailable"),
         "status": row}
        for row in unavailable_statuses
    ]
    rows.sort(key=lambda row: int(row["hypothesis"].removeprefix("H")))
    row_count = len(rows)
    figure, axes = _new_figure(width=12.8, height=max(7.0, 0.72 * row_count + 3.0))
    figure.set_layout_engine(None)
    figure.subplots_adjust(left=0.40, right=0.97, top=0.88, bottom=0.12)
    axis = axes[0, 0]
    effect_positions = [index for index, row in enumerate(rows) if "effect" in row]
    if effect_positions:
        estimates = np.array([rows[index]["effect"]["estimate"] for index in effect_positions], dtype=float)
        lows = np.array([rows[index]["effect"][low] for index in effect_positions], dtype=float)
        highs = np.array([rows[index]["effect"][high] for index in effect_positions], dtype=float)
        axis.errorbar(
            estimates, effect_positions, xerr=np.vstack((estimates - lows, highs - estimates)),
            fmt="o", color=COLORBLIND_PALETTE[0], ecolor="#333333",
            elinewidth=1.3, capsize=3, markersize=5,
        )
    axis.axvline(0, color=COLORBLIND_PALETTE[5], linestyle="--", linewidth=1)
    tick_positions = list(range(row_count))
    labels = [_wrapped(row["label"], 48) for row in rows]
    for position, row in enumerate(rows):
        if "status" not in row:
            continue
        status = row["status"]
        prefix = "No registered scalar effect" if row["hypothesis"] == "H5" else "No estimate or confidence interval"
        reason_lines = textwrap.wrap(status["reason"], width=93)
        reason = "\n".join(reason_lines[:2])
        if len(reason_lines) > 2:
            reason += " … (see hypothesis status table)"
        axis.text(
            0.02, position, f"{prefix} · {_pretty(status['status'])}\n{reason}",
            transform=axis.get_yaxis_transform(), va="center", ha="left",
            fontsize=7.2, color="#555555",
        )
    axis.set_yticks(tick_positions, labels)
    axis.set_ylim(row_count - 0.35, -0.65)
    axis.set_title("Reported confidence intervals and explicit estimability status")
    axis.set_xlabel("Effect estimate (positive favours SemUAV-VLM)")
    axis.set_ylabel("")
    _finish_axis(axis, grid_axis="x")
    figure.suptitle("Confirmatory effects H1–H4 and H5 replication status", fontsize=14, fontweight="bold")
    return figure, (name,)


_FIGURE_SPECS = (
    _FigureSpec("dataset_split_balance", "Dataset and split balance", _figure_dataset_balance),
    _FigureSpec("exp1_perception_metrics", "Experiment 1 perception metrics", _figure_exp1_metrics),
    _FigureSpec("exp1_latency", "Experiment 1 latency", _figure_exp1_latency),
    _FigureSpec("exp2_pareto_frontier", "Experiment 2 Pareto frontier", _figure_exp2_pareto),
    _FigureSpec("exp3_deadline_heatmap", "Experiment 3 deadline heatmap", _figure_exp3_deadline_heatmap),
    _FigureSpec("exp4_utility_resources", "Experiment 4 utility and resources", _figure_exp4_utility_resources),
    _FigureSpec("exp4_mode_mix", "Experiment 4 mode mix", _figure_exp4_mode_mix),
    _FigureSpec("exp5_safety", "Experiment 5 safety", _figure_exp5_safety),
    _FigureSpec("exp6_protocol", "Experiment 6 protocol", _figure_exp6_protocol),
    _FigureSpec("exp7_readiness_hardware", "Experiment 7 readiness or hardware", _figure_exp7_readiness_hardware),
    _FigureSpec(
        "hypotheses_h1_h5_forest",
        "H1-H4 effect forest and H5 replication status",
        _figure_hypothesis_forest,
    ),
)


def _frame_contains_fixture_marker(frame: pd.DataFrame) -> bool:
    if "scientific" in frame.columns:
        values = frame["scientific"].dropna()
        if len(values) and not values.astype(bool).all():
            return True
    candidate_columns = [
        column
        for column in (
            "source_model",
            "evidence_class",
            "execution_status",
            "claim_status",
            "profile",
        )
        if column in frame.columns
    ]
    for column in candidate_columns:
        values = " ".join(frame[column].dropna().astype(str).str.lower().unique())
        if any(marker in values for marker in _FIXTURE_MARKERS):
            return True
    return False


def _add_non_scientific_watermark(figure: Figure) -> None:
    figure.text(
        0.5,
        0.50,
        "NON-SCIENTIFIC FIXTURE",
        ha="center",
        va="center",
        rotation=24,
        fontsize=29,
        fontweight="bold",
        color=COLORBLIND_PALETTE[5],
        alpha=0.16,
        zorder=1000,
        transform=figure.transFigure,
    )
    figure.text(
        0.995,
        0.005,
        "Software integration output — not empirical evidence",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555555",
        alpha=0.9,
        transform=figure.transFigure,
    )


def _atomic_save_figure(figure: Figure, destination: Path, output_format: str) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        figure.savefig(
            temporary,
            format=output_format,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            facecolor="white",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_manifests(
    output_dir: Path,
    rows: list[dict[str, Any]],
    source_hashes: Mapping[str, str],
    scientific: bool,
) -> None:
    frame = pd.DataFrame(rows)
    csv_frame = frame.copy()
    csv_frame["source_frames"] = csv_frame["source_frames"].map(
        lambda values: ";".join(values)
    )
    csv_frame["source_frame_hashes"] = csv_frame["source_frame_hashes"].map(
        lambda values: json.dumps(values, sort_keys=True, separators=(",", ":"))
    )
    _atomic_write_text(
        output_dir / "figure_manifest.csv",
        csv_frame.to_csv(index=False, lineterminator="\n"),
    )
    json_payload = {
        "schema_version": FIGURE_MANIFEST_SCHEMA,
        "scientific_requested": bool(scientific),
        "output_dpi": OUTPUT_DPI,
        "output_formats": list(OUTPUT_FORMATS),
        "source_frame_hashes": dict(sorted(source_hashes.items())),
        "figures": rows,
    }
    _atomic_write_text(
        output_dir / "figure_manifest.json",
        json.dumps(json_payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def generate_publication_figures(
    frames: Mapping[str, pd.DataFrame | None],
    output_dir: str | Path,
    scientific: bool,
    *,
    experiment: int | None = None,
    evidence_label: str | None = None,
    strict: bool = False,
) -> dict[str, Figure]:
    """Generate available figures, optionally just one experiment's figures.

    Parameters
    ----------
    frames:
        Mapping from study artifact names to dataframes.  The native notebook
        names (for example ``exp2_frontier`` and ``exp4_policy``) are accepted,
        along with documented aliases.  ``None`` values are treated as
        unavailable artifacts.
    output_dir:
        Directory in which figures and ``figure_manifest.{csv,json}`` are
        written.
    scientific:
        ``True`` only for a gate-passing empirical run.  When false—or when an
        input frame carries an explicit fixture marker—the affected figure is
        visibly watermarked.

    experiment:
        ``1`` through ``7`` selects one experiment. ``0`` selects the dataset
        overview and hypothesis forest. ``None`` preserves the full suite.
        Filtered calls merge their manifest entries with earlier experiments.
    evidence_label:
        Optional phase label drawn on exports, for example development results
        or an unfinished test preview. It never changes scientific provenance.
    strict:
        Raise after saving the manifest if an actual plotting/export error
        occurs. Missing or non-estimable source data remain explicit skips.

    Returns
    -------
    dict[str, matplotlib.figure.Figure]
        Live figures keyed by stable figure ID, suitable for notebook
        ``display``.  Skipped figures are represented only in the manifests.
    """

    if not isinstance(frames, Mapping):
        raise TypeError("frames must be a mapping of artifact names to pandas DataFrames")
    if not isinstance(scientific, (bool, np.bool_)):
        raise TypeError("scientific must be a boolean")
    if experiment is not None and (
        isinstance(experiment, bool) or not isinstance(experiment, (int, np.integer))
        or not 0 <= int(experiment) <= 7
    ):
        raise ValueError("experiment must be None or an integer from 0 through 7")
    if evidence_label is not None and not isinstance(evidence_label, str):
        raise TypeError("evidence_label must be a string or None")
    specs = tuple(spec for spec in _FIGURE_SPECS if (
        experiment is None
        or (experiment == 0 and spec.figure_id in {"dataset_split_balance", "hypotheses_h1_h5_forest"})
        or (experiment != 0 and spec.figure_id.startswith(f"exp{experiment}_"))
    ))
    available: dict[str, pd.DataFrame] = {}
    for raw_name, value in frames.items():
        name = str(raw_name)
        if value is None:
            continue
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"frames[{name!r}] must be a pandas DataFrame or None")
        available[name] = value

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_hashes = {name: _frame_sha256(frame) for name, frame in sorted(available.items())}
    generated: dict[str, Figure] = {}
    manifest_rows: list[dict[str, Any]] = []
    plotting_errors: list[str] = []
    if experiment is not None and (destination / "figure_manifest.json").is_file():
        previous = json.loads((destination / "figure_manifest.json").read_text(encoding="utf-8"))
        if previous.get("schema_version") != FIGURE_MANIFEST_SCHEMA:
            raise ValueError("Existing figure manifest has an unsupported schema")
        affected_ids = {spec.figure_id for spec in specs}
        manifest_rows = [row for row in previous["figures"] if row["figure_id"] not in affected_ids]
        source_hashes = {**previous.get("source_frame_hashes", {}), **source_hashes}

    for spec in specs:
        prior_figure_numbers = set(plt.get_fignums())
        figure: Figure | None = None
        source_names: tuple[str, ...] = ()
        written: list[Path] = []
        try:
            figure, source_names = spec.builder(available)
            fixture_detected = any(
                _frame_contains_fixture_marker(available[name]) for name in source_names
            )
            watermarked = not bool(scientific) or fixture_detected
            if watermarked:
                _add_non_scientific_watermark(figure)
            if evidence_label:
                figure.text(0.5, 1.02, evidence_label, ha="center", va="bottom",
                            fontsize=9, fontweight="bold", color="#555555",
                            transform=figure.transFigure)
            for output_format in OUTPUT_FORMATS:
                path = destination / f"{spec.figure_id}.{output_format}"
                _atomic_save_figure(figure, path, output_format)
                written.append(path)
            generated[spec.figure_id] = figure
            manifest_rows.append({
                "figure_id": spec.figure_id,
                "title": spec.title,
                "status": "generated",
                "reason": "generated from available measured columns",
                "evidence_label": evidence_label or "",
                "source_frames": list(source_names),
                "source_frame_hashes": {
                    name: source_hashes[name] for name in source_names
                },
                "scientific_requested": bool(scientific),
                "watermarked_non_scientific": bool(watermarked),
                "png_path": f"{spec.figure_id}.png",
                "pdf_path": f"{spec.figure_id}.pdf",
                "svg_path": f"{spec.figure_id}.svg",
            })
        except _SkipFigure as exc:
            for figure_number in set(plt.get_fignums()) - prior_figure_numbers:
                plt.close(figure_number)
            source_names = exc.sources
            for output_format in OUTPUT_FORMATS:
                (destination / f"{spec.figure_id}.{output_format}").unlink(missing_ok=True)
            manifest_rows.append({
                "figure_id": spec.figure_id,
                "title": spec.title,
                "status": "skipped",
                "reason": exc.reason,
                "source_frames": list(source_names),
                "source_frame_hashes": {
                    name: source_hashes[name]
                    for name in source_names
                    if name in source_hashes
                },
                "scientific_requested": bool(scientific),
                "watermarked_non_scientific": False,
                "png_path": "",
                "pdf_path": "",
                "svg_path": "",
            })
        except Exception as exc:
            for figure_number in set(plt.get_fignums()) - prior_figure_numbers:
                plt.close(figure_number)
            for output_format in OUTPUT_FORMATS:
                (destination / f"{spec.figure_id}.{output_format}").unlink(missing_ok=True)
            plotting_errors.append(f"{spec.figure_id}: {type(exc).__name__}: {exc}")
            manifest_rows.append({
                "figure_id": spec.figure_id,
                "title": spec.title,
                "status": "skipped",
                "reason": f"plotting failed safely: {type(exc).__name__}: {exc}",
                "source_frames": list(source_names),
                "source_frame_hashes": {
                    name: source_hashes[name]
                    for name in source_names
                    if name in source_hashes
                },
                "scientific_requested": bool(scientific),
                "watermarked_non_scientific": False,
                "png_path": "",
                "pdf_path": "",
                "svg_path": "",
            })

    order = {spec.figure_id: index for index, spec in enumerate(_FIGURE_SPECS)}
    manifest_rows.sort(key=lambda row: order.get(row["figure_id"], len(order)))
    _write_manifests(destination, manifest_rows, source_hashes, bool(scientific))
    if strict and plotting_errors:
        for figure in generated.values():
            plt.close(figure)
        raise RuntimeError("Publication figure export failed; see figure_manifest.json: " + "; ".join(plotting_errors))
    return generated


# A concise alias for notebook prose without weakening the stable public name.
publication_figure_suite = generate_publication_figures


__all__ = [
    "COLORBLIND_PALETTE",
    "FIGURE_MANIFEST_SCHEMA",
    "HYPOTHESIS_EFFECT_COLUMNS",
    "OUTPUT_DPI",
    "build_hypothesis_effects_frame",
    "generate_publication_figures",
    "publication_figure_suite",
]

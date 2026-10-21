"""Out-of-core reduction of dynamic policy-trial partitions.

The confirmatory dynamic campaign is intentionally much larger than the
in-memory development fixtures.  This module turns the checksummed
``(trace, seed)`` CSV partitions into the compact, cluster-level artifacts
needed by the paper analysis without ever concatenating the raw campaign.

There are three deliberately independent memory bounds:

* every source file is read with an explicit ``usecols`` and ``chunksize``;
* exact quantiles and pairing checks use a temporary SQLite database;
* H4 is spooled and ranked one trace at a time.  Only one aggregate row per
  visual-group/trace/policy cell is returned, never campaign-wide selected
  trial rows.

The reducer is deterministic under partition reordering and records enough I/O
information for a run receipt to prove those bounds were actually used.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from .registry import (
    StudyGateError,
    atomic_write_frame,
    atomic_write_json,
    hash_file,
)
from .evaluation import BootstrapEffect


REDUCER_VERSION = "semuav.policy-trial-reducer.v2"

PAIR_COLUMNS = (
    "sample_id",
    "visual_group_id",
    "mission_profile",
    "trace_id",
    "replicate_seed",
)

# The contract is the union needed for H2/H3/H4, the VisDrone Experiment 5
# table, resource accounting, and the controller-behaviour table.  Requiring
# it up front prevents a huge campaign from finishing with an unusable output.
REQUIRED_COLUMNS = (
    *PAIR_COLUMNS,
    "policy",
    "episode_id",
    "capture_us",
    "action_ready_us",
    "decision_us",
    "selected_mode",
    "variant",
    "gold_action",
    "correct_action",
    "correct_timely_action",
    "delivered_by_deadline",
    "false_alert",
    "critical_miss",
    "critical",
    "total_wire_bytes",
    "energy_j",
    "calibrated_correctness_probability",
    "high_confidence_semantic_error",
    "evidence_escalated",
    "application_payload_bytes",
    "full_message_received",
    "state_valid",
    "state_kind",
    "apply_status",
    "original_uplink_bytes",
    "retry_uplink_bytes",
    "decision_queue_pending_bytes",
    "decision_queue_outstanding_unacked_bytes",
    "decision_queue_messages",
    "decision_queue_oldest_age_ms",
    "decision_backlog_bytes",
    "decision_sender_sequence_gap",
    "episode_outage_count",
    "episode_outages_recovered",
    "outage_recovery_event",
    "outage_recovery_latency_ms",
)

OPTIONAL_ENERGY_COLUMNS = (
    "representation_energy_j",
    "detector_energy_j",
    "context_model_energy_j",
    "receiver_energy_j_charged",
)

_INTEGER_COLUMNS = {
    "replicate_seed",
    "capture_us",
    "action_ready_us",
    "decision_us",
    "application_payload_bytes",
    "original_uplink_bytes",
    "retry_uplink_bytes",
    "decision_queue_messages",
    "decision_sender_sequence_gap",
    "episode_outage_count",
    "episode_outages_recovered",
}
_REAL_COLUMNS = {
    "total_wire_bytes",
    "energy_j",
    "calibrated_correctness_probability",
    "decision_queue_pending_bytes",
    "decision_queue_outstanding_unacked_bytes",
    "decision_queue_oldest_age_ms",
    "decision_backlog_bytes",
    "outage_recovery_latency_ms",
    *OPTIONAL_ENERGY_COLUMNS,
}
_BOOLEAN_COLUMNS = {
    "correct_action",
    "correct_timely_action",
    "delivered_by_deadline",
    "false_alert",
    "critical_miss",
    "critical",
    "high_confidence_semantic_error",
    "evidence_escalated",
    "full_message_received",
    "state_valid",
    "outage_recovery_event",
}


@dataclasses.dataclass(frozen=True)
class PolicyPartition:
    """One immutable dynamic-trial partition."""

    path: Path
    sha256: str
    row_count: int
    trace_id: str | None = None
    replicate_seed: int | None = None
    stage: str | None = None


@dataclasses.dataclass(frozen=True)
class PolicyReductionResult:
    """Paths and compact frames produced by a policy reduction."""

    artifacts: Mapping[str, Path]
    frames: Mapping[str, pd.DataFrame]
    receipt_path: Path

    def artifact(self, name: str) -> Path:
        try:
            return Path(self.artifacts[name])
        except KeyError as exc:
            raise KeyError(f"Unknown policy-reduction artifact {name!r}") from exc


def _first(item: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in item and item[name] not in (None, ""):
            return item[name]
    return None


def load_verified_partition_index(
    index: str | Path | pd.DataFrame | Sequence[PolicyPartition | Mapping[str, Any]],
) -> tuple[PolicyPartition, ...]:
    """Normalize and checksum a runner partition index.

    The runner's preferred columns are ``artifact_path``,
    ``artifact_sha256`` and ``row_count``.  Historical aliases are accepted so
    resumed validation campaigns can be reduced by the same implementation.
    """

    base = Path.cwd()
    if isinstance(index, (str, Path)):
        source = Path(index).expanduser().resolve()
        if not source.is_file():
            raise StudyGateError(f"Policy partition index is missing: {source}")
        base = source.parent
        if source.suffix.lower() == ".json":
            payload = json.loads(source.read_text(encoding="utf-8"))
            records = payload.get("partitions", payload) if isinstance(payload, Mapping) else payload
        else:
            records = pd.read_csv(source, compression="infer").to_dict("records")
    elif isinstance(index, pd.DataFrame):
        records = index.to_dict("records")
    else:
        records = list(index)
    if not records:
        raise StudyGateError("Policy partition index is empty")

    partitions: list[PolicyPartition] = []
    for position, raw in enumerate(records):
        if isinstance(raw, PolicyPartition):
            part = raw
        else:
            if not isinstance(raw, Mapping):
                raise TypeError(f"Partition index row {position} is not a mapping")
            path_value = _first(raw, ("artifact_path", "artifact", "path"))
            digest = _first(raw, ("artifact_sha256", "sha256", "checksum"))
            rows = _first(raw, ("row_count", "rows"))
            if path_value is None or digest is None or rows is None:
                raise StudyGateError(
                    "Policy partition index requires artifact_path/artifact, "
                    "artifact_sha256/sha256, and row_count/rows"
                )
            candidate = Path(str(path_value)).expanduser()
            if not candidate.is_absolute():
                candidate = base / candidate
            part = PolicyPartition(
                path=candidate.resolve(),
                sha256=str(digest).lower(),
                row_count=int(rows),
                trace_id=(
                    str(raw["trace_id"]) if raw.get("trace_id") not in (None, "") else None
                ),
                replicate_seed=(
                    int(raw["replicate_seed"])
                    if raw.get("replicate_seed") not in (None, "") else None
                ),
                stage=(str(raw["stage"]) if raw.get("stage") not in (None, "") else None),
            )
        if not re.fullmatch(r"[0-9a-f]{64}", str(part.sha256).lower()):
            raise StudyGateError(f"Partition has an invalid SHA-256: {part.path}")
        if int(part.row_count) < 0:
            raise StudyGateError(f"Partition has a negative row count: {part.path}")
        if not part.path.is_file():
            raise StudyGateError(f"Policy partition is missing: {part.path}")
        observed = hash_file(part.path)
        if observed != str(part.sha256).lower():
            raise StudyGateError(f"Policy partition checksum changed: {part.path}")
        partitions.append(dataclasses.replace(part, sha256=observed))

    resolved = [part.path for part in partitions]
    if len(set(resolved)) != len(resolved):
        raise StudyGateError("Policy partition index contains duplicate artifacts")
    # Sorting makes every compact output invariant to index order.
    return tuple(sorted(partitions, key=lambda item: str(item.path)))


def _csv_columns(path: Path) -> tuple[str, ...]:
    # nrows=0 is deliberately not used: all pandas reads of the trial corpus
    # must carry both explicit usecols and chunksize, which is easy to audit.
    import csv
    import gzip

    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle), None)
    if not header:
        raise StudyGateError(f"Policy partition has no CSV header: {path}")
    if len(header) != len(set(header)):
        raise StudyGateError(f"Policy partition has duplicate CSV columns: {path}")
    return tuple(header)


def _sqlite_type(column: str) -> str:
    if column in _INTEGER_COLUMNS or column in _BOOLEAN_COLUMNS:
        return "INTEGER"
    if column in _REAL_COLUMNS:
        return "REAL"
    return "TEXT"


def _python_value(value: Any, column: str) -> Any:
    if pd.isna(value):
        return None
    if column in _BOOLEAN_COLUMNS:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes"}:
                return 1
            if lowered in {"false", "0", "no"}:
                return 0
            raise StudyGateError(f"Invalid boolean value {value!r} in {column}")
        return int(bool(value))
    if column in _INTEGER_COLUMNS:
        return int(value)
    if column in _REAL_COLUMNS:
        return float(value)
    return str(value)


def _exact_sql_quantile(
    connection: sqlite3.Connection,
    *,
    table: str,
    expression: str,
    where: str,
    parameters: Sequence[Any],
    quantile: float,
) -> float:
    """Linear exact quantile using SQLite order statistics."""

    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must lie in [0, 1]")
    count = int(connection.execute(
        f"SELECT COUNT(*) FROM {table} WHERE {where}", parameters,
    ).fetchone()[0])
    if count == 0:
        return float("nan")
    position = (count - 1) * float(quantile)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    query = (
        f"SELECT {expression} FROM {table} WHERE {where} "
        f"ORDER BY {expression} LIMIT 1 OFFSET ?"
    )
    low = float(connection.execute(query, (*parameters, lower)).fetchone()[0])
    if upper == lower:
        return low
    high = float(connection.execute(query, (*parameters, upper)).fetchone()[0])
    return low + (position - lower) * (high - low)


def _write_compact_frames(
    output_dir: Path,
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for name, frame in sorted(frames.items()):
        path = output_dir / f"{name}.csv.gz"
        atomic_write_frame(path, frame.reset_index(drop=True))
        artifacts[name] = path
    return artifacts


def _h4_trace_token(trace_id: str) -> str:
    return hashlib.sha256(trace_id.encode("utf-8")).hexdigest()[:16]


def _create_h4_spill(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS h4_trials (
            sample_id TEXT NOT NULL,
            visual_group_id TEXT NOT NULL,
            mission_profile TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            replicate_seed INTEGER NOT NULL,
            policy TEXT NOT NULL,
            selected_mode TEXT NOT NULL,
            score REAL,
            correct_action INTEGER NOT NULL,
            PRIMARY KEY (
                sample_id, visual_group_id, mission_profile,
                trace_id, replicate_seed, policy
            )
        ) WITHOUT ROWID
        """
    )
    return connection


def _reduce_h4_trace(
    database: Path,
    trace_id: str,
    policies: tuple[str, str],
    target_coverage: float,
    semantic_acceptance_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank one trace on disk and return both registered H4 endpoints.

    Matched-coverage selective risk and high-confidence semantic error are
    distinct estimands.  Their denominators and errors are retained separately
    so an empty high-confidence set cannot be mistaken for zero error.
    """

    connection = sqlite3.connect(database)
    try:
        observed = {
            str(row[0]) for row in connection.execute(
                "SELECT DISTINCT policy FROM h4_trials ORDER BY policy"
            )
        }
        if observed != set(policies):
            raise StudyGateError(
                f"H4 trace {trace_id!r} is missing policies: {sorted(set(policies) - observed)}"
            )
        rows: list[dict[str, Any]] = []
        groups = [
            str(row[0]) for row in connection.execute(
                "SELECT DISTINCT visual_group_id FROM h4_trials ORDER BY visual_group_id"
            )
        ]
        for visual_group_id in groups:
            totals: dict[str, int] = {}
            for policy in policies:
                total = int(connection.execute(
                    "SELECT COUNT(*) FROM h4_trials WHERE visual_group_id=? AND policy=?",
                    (visual_group_id, policy),
                ).fetchone()[0])
                totals[policy] = total
            if len(set(totals.values())) != 1:
                raise StudyGateError(
                    "H4 paired trial counts differ within visual-group/trace: "
                    f"{visual_group_id}/{trace_id} {totals}"
                )
            for policy in policies:
                total = totals[policy]
                target = max(1, int(math.ceil(float(target_coverage) * total)))
                eligible = int(connection.execute(
                    """
                    SELECT COUNT(*) FROM h4_trials
                    WHERE visual_group_id=? AND policy=?
                      AND selected_mode IN ('L1_ALERT', 'L2_SEMANTICS')
                      AND score BETWEEN 0.0 AND 1.0
                    """,
                    (visual_group_id, policy),
                ).fetchone()[0])
                selected = 0
                errors = 0
                if eligible >= target:
                    selected, errors = map(int, connection.execute(
                        """
                        SELECT COUNT(*), COALESCE(SUM(1 - correct_action), 0)
                        FROM (
                            SELECT correct_action
                            FROM h4_trials
                            WHERE visual_group_id=? AND policy=?
                              AND selected_mode IN ('L1_ALERT', 'L2_SEMANTICS')
                              AND score BETWEEN 0.0 AND 1.0
                            ORDER BY score DESC, sample_id, mission_profile,
                                     replicate_seed
                            LIMIT ?
                        )
                        """,
                        (visual_group_id, policy, target),
                    ).fetchone())
                high_confidence_rows, high_confidence_errors = map(
                    int,
                    connection.execute(
                        """
                        SELECT COUNT(*), COALESCE(SUM(1 - correct_action), 0)
                        FROM h4_trials
                        WHERE visual_group_id=? AND policy=?
                          AND selected_mode IN ('L1_ALERT', 'L2_SEMANTICS')
                          AND score BETWEEN ? AND 1.0
                        """,
                        (
                            visual_group_id,
                            policy,
                            float(semantic_acceptance_threshold),
                        ),
                    ).fetchone(),
                )
                matched_coverage_passed = bool(eligible >= target and selected == target)
                high_confidence_passed = bool(high_confidence_rows > 0)
                rows.append({
                    "trace_id": trace_id,
                    "visual_group_id": visual_group_id,
                    "policy": policy,
                    "total_rows": total,
                    "available_semantic_rows": eligible,
                    "target_rows": target,
                    "selected_rows": selected,
                    "semantic_errors": errors,
                    "realized_coverage": selected / total if total else 0.0,
                    "matched_coverage_semantic_error": (
                        errors / selected if selected else np.nan
                    ),
                    "matched_coverage_passed": matched_coverage_passed,
                    "high_confidence_semantic_rows": high_confidence_rows,
                    "high_confidence_semantic_errors": high_confidence_errors,
                    "high_confidence_semantic_error_rate": (
                        high_confidence_errors / high_confidence_rows
                        if high_confidence_rows else np.nan
                    ),
                    "high_confidence_semantic_passed": high_confidence_passed,
                    "semantic_acceptance_threshold": float(
                        semantic_acceptance_threshold
                    ),
                    "passed": bool(matched_coverage_passed and high_confidence_passed),
                    "selection_rule": (
                        "top_calibrated_semantic_score_within_visual_group_trace"
                    ),
                })
        cells = pd.DataFrame(rows).sort_values(
            ["trace_id", "visual_group_id", "policy"], kind="mergesort",
        ).reset_index(drop=True)
        audit_rows: list[dict[str, Any]] = []
        for policy, group in cells.groupby("policy", sort=True):
            audit_rows.append({
                "trace_id": trace_id,
                "policy": policy,
                "target_coverage": float(target_coverage),
                "visual_group_trace_cells": int(len(group)),
                "available_semantic_rows": int(group["available_semantic_rows"].sum()),
                "selected_rows": int(group["selected_rows"].sum()),
                "realized_group_trace_equal_coverage": float(group["realized_coverage"].mean()),
                "group_trace_equal_selective_risk": float(
                    group["matched_coverage_semantic_error"].mean()
                ),
                # Compatibility alias used by the existing paper report.
                "group_equal_selective_risk": float(
                    group["matched_coverage_semantic_error"].mean()
                ),
                "high_confidence_semantic_rows": int(
                    group["high_confidence_semantic_rows"].sum()
                ),
                "high_confidence_semantic_errors": int(
                    group["high_confidence_semantic_errors"].sum()
                ),
                "group_trace_equal_high_confidence_semantic_error_rate": float(
                    group["high_confidence_semantic_error_rate"].mean()
                ),
                "zero_high_confidence_visual_group_trace_cells": int(
                    (~group["high_confidence_semantic_passed"]).sum()
                ),
                "matched_coverage_passed": bool(
                    group["matched_coverage_passed"].all() and len(group)
                ),
                "high_confidence_semantic_passed": bool(
                    group["high_confidence_semantic_passed"].all() and len(group)
                ),
                "semantic_acceptance_threshold": float(
                    semantic_acceptance_threshold
                ),
                "insufficient_visual_group_trace_cells": int((~group["passed"]).sum()),
                "passed": bool(group["passed"].all() and len(group)),
                "selection_rule": (
                    "top_calibrated_semantic_score_within_visual_group_trace"
                ),
            })
        return cells, pd.DataFrame(audit_rows)
    finally:
        connection.close()


def _resource_and_pair_frames(
    connection: sqlite3.Connection,
    policies: Sequence[str],
    semantic_acceptance_threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_cells = pd.read_sql_query(
        """
        SELECT visual_group_id, trace_id, policy,
               COUNT(*) AS trials,
               SUM(correct_timely_action) AS correct_timely_actions,
               AVG(correct_timely_action) AS correct_timely_rate,
               SUM(CASE WHEN critical=1 THEN critical_miss ELSE 0 END) AS critical_misses,
               SUM(CASE WHEN critical=1 THEN 1 ELSE 0 END) AS critical_trials,
               SUM(CASE WHEN critical=0 THEN false_alert ELSE 0 END) AS false_alerts,
               SUM(CASE WHEN critical=0 THEN 1 ELSE 0 END) AS noncritical_trials,
               SUM(CASE WHEN selected_mode IN ('L1_ALERT','L2_SEMANTICS')
                              AND calibrated_correctness_probability>=?
                        THEN 1 ELSE 0 END)
                   AS high_confidence_semantic_rows,
               SUM(CASE WHEN selected_mode IN ('L1_ALERT','L2_SEMANTICS')
                              AND calibrated_correctness_probability>=?
                              AND correct_action=0
                        THEN 1 ELSE 0 END)
                   AS high_confidence_semantic_errors,
               SUM(total_wire_bytes) AS total_wire_bytes,
               SUM(energy_j) AS total_energy_j
        FROM trials
        GROUP BY visual_group_id, trace_id, policy
        ORDER BY visual_group_id, trace_id, policy
        """,
        connection,
        params=(
            float(semantic_acceptance_threshold),
            float(semantic_acceptance_threshold),
        ),
    )
    pair_cells["high_confidence_semantic_error_rate"] = np.where(
        pair_cells["high_confidence_semantic_rows"].gt(0),
        pair_cells["high_confidence_semantic_errors"]
        / pair_cells["high_confidence_semantic_rows"],
        np.nan,
    )
    h3_episode = pd.read_sql_query(
        """
        SELECT visual_group_id, mission_profile, trace_id, replicate_seed,
               episode_id, policy,
               COUNT(*) AS negative_decision_exposures,
               AVG(false_alert) AS false_alert_rate
        FROM trials
        WHERE critical=0
        GROUP BY visual_group_id, mission_profile, trace_id, replicate_seed,
                 episode_id, policy
        ORDER BY visual_group_id, mission_profile, trace_id, replicate_seed,
                 episode_id, policy
        """,
        connection,
    )

    rows: list[dict[str, Any]] = []
    for policy in sorted(map(str, policies)):
        base = connection.execute(
            """
            SELECT COUNT(*), AVG(correct_timely_action), AVG(total_wire_bytes),
                   AVG(energy_j), AVG(delivered_by_deadline),
                   AVG(CASE WHEN gold_action!='RESPOND' THEN false_alert END),
                   AVG(CASE WHEN gold_action='RESPOND' THEN critical_miss END)
            FROM trials WHERE policy=?
            """,
            (policy,),
        ).fetchone()
        row = {
            "policy": policy,
            "trials": int(base[0]),
            "correct_timely_rate": float(base[1]),
            "mean_wire_bytes": float(base[2]),
            "mean_energy_j": float(base[3]),
            "delivery_rate": float(base[4]),
            "false_alert_rate": float(base[5]) if base[5] is not None else np.nan,
            "critical_miss_rate": float(base[6]) if base[6] is not None else np.nan,
        }
        latency_expression = "(action_ready_us - capture_us) / 1000.0"
        for quantile, suffix in ((0.50, "p50"), (0.95, "p95"), (0.99, "p99")):
            row[f"latency_{suffix}_ms"] = _exact_sql_quantile(
                connection,
                table="trials",
                expression=latency_expression,
                where="policy=? AND action_ready_us IS NOT NULL",
                parameters=(policy,),
                quantile=quantile,
            )
        rows.append(row)
    return pair_cells, h3_episode, pd.DataFrame(rows)


def _episode_resource_frames(
    connection: sqlite3.Connection,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Build compact episode/resource cells with exact capture exposure.

    The ordered cursor retains capture timestamps for one registered episode
    only.  The returned tables have one row per episode and one row per
    visual-group/trace/policy cell respectively; neither scales with the raw
    decisions inside an episode.
    """

    query = connection.execute(
        """
        SELECT DISTINCT policy, trace_id, replicate_seed, episode_id,
                        visual_group_id, capture_us
        FROM trials
        ORDER BY policy, trace_id, replicate_seed, episode_id,
                 visual_group_id, capture_us
        """
    )
    durations: dict[tuple[str, str, int, str], tuple[str, float]] = {}
    current: tuple[Any, ...] | None = None
    captures: list[int] = []
    max_captures = 0

    def finish(key: tuple[Any, ...] | None, values: list[int]) -> None:
        nonlocal max_captures
        if key is None:
            return
        max_captures = max(max_captures, len(values))
        if len(values) >= 2:
            interval = max(1, int(np.median(np.diff(np.asarray(values, dtype=np.int64)))))
            duration_us = int(values[-1] - values[0] + interval)
        else:
            duration_us = 1_000_000
        episode_key = (str(key[0]), str(key[1]), int(key[2]), str(key[3]))
        visual_group_id = str(key[4])
        if episode_key in durations:
            raise StudyGateError(
                "A resource episode maps to more than one independent visual group"
            )
        durations[episode_key] = (visual_group_id, duration_us / 1e6)

    for policy, trace_id, seed, episode, visual_group_id, capture in query:
        key = (policy, trace_id, seed, episode, visual_group_id)
        if current is not None and key != current:
            finish(current, captures)
            captures = []
        current = key
        captures.append(int(capture))
    finish(current, captures)
    component_sql = ", ".join(
        f"SUM({column}) AS {column}" for column in OPTIONAL_ENERGY_COLUMNS
    )
    totals = connection.execute(
        f"""
        SELECT policy, trace_id, replicate_seed, episode_id,
               SUM(total_wire_bytes) AS wire_bytes,
               SUM(energy_j) AS total_energy_j,
               {component_sql}
        FROM trials
        GROUP BY policy, trace_id, replicate_seed, episode_id
        ORDER BY policy, trace_id, replicate_seed, episode_id
        """
    )
    names = [item[0] for item in totals.description]
    episode_rows: list[dict[str, Any]] = []
    for raw in totals:
        row = dict(zip(names, raw))
        key = (
            str(row["policy"]), str(row["trace_id"]),
            int(row["replicate_seed"]), str(row["episode_id"]),
        )
        if key not in durations:
            raise StudyGateError(f"Resource episode lacks capture exposure: {key}")
        visual_group_id, duration_s = durations.pop(key)
        episode_rows.append({
            "policy": key[0], "trace_id": key[1], "replicate_seed": key[2],
            "episode_id": key[3], "visual_group_id": visual_group_id,
            "duration_s": float(duration_s),
            "wire_bytes": float(row["wire_bytes"]),
            "total_energy_j": float(row["total_energy_j"]),
            **{
                column: float(row[column])
                for column in OPTIONAL_ENERGY_COLUMNS
            },
        })
    if durations:
        raise StudyGateError("Some capture-exposure episodes lack resource totals")
    episodes = pd.DataFrame(episode_rows).sort_values(
        ["policy", "visual_group_id", "trace_id", "replicate_seed", "episode_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    clusters = episodes.groupby(
        ["policy", "visual_group_id", "trace_id"], as_index=False, sort=True,
    ).agg(
        episodes=("episode_id", "size"),
        exposure_s=("duration_s", "sum"),
        wire_bytes=("wire_bytes", "sum"),
        total_energy_j=("total_energy_j", "sum"),
        **{
            column: (column, "sum")
            for column in OPTIONAL_ENERGY_COLUMNS
        },
    )
    return episodes, clusters, max_captures


def _resource_rate_frame(
    episode_cells: pd.DataFrame,
    policies: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for policy in sorted(map(str, policies)):
        group = episode_cells[episode_cells["policy"].astype(str).eq(policy)]
        exposure = float(group["duration_s"].sum())
        if exposure <= 0:
            raise StudyGateError(f"Policy {policy!r} has no positive episode exposure")
        wire = float(group["wire_bytes"].sum())
        energy = float(group["total_energy_j"].sum())
        row: dict[str, Any] = {
            "policy": policy,
            "episodes": int(len(group)),
            "exposure_s": exposure,
            "total_wire_bytes": wire,
            "energy_j": energy,
            "wire_bytes_per_s": wire / exposure,
            "energy_j_per_s": energy / exposure,
        }
        for column in OPTIONAL_ENERGY_COLUMNS:
            value = float(group[column].sum())
            row[column] = value
            row[f"{column}_per_s"] = value / exposure
        rows.append(row)
    return pd.DataFrame(rows)


def _exp5_frame(
    connection: sqlite3.Connection,
    energy_scope: str,
    semantic_acceptance_threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    subsets = {
        "natural_prevalence": "1=1",
        "mission_priority_positive": "critical=1",
        "mission_priority_negative": "critical=0",
    }
    for subset, predicate in subsets.items():
        cells = pd.read_sql_query(
            f"""
            SELECT policy, visual_group_id, COUNT(*) AS trials,
                   AVG(correct_timely_action) AS correct_timely_rate,
                   SUM(CASE WHEN critical=0 THEN 1 ELSE 0 END)
                       AS noncritical_rows,
                   SUM(CASE WHEN critical=0 AND false_alert=1 THEN 1 ELSE 0 END)
                       AS false_alerts,
                   CASE WHEN SUM(CASE WHEN critical=0 THEN 1 ELSE 0 END)>0
                        THEN 1.0 * SUM(CASE WHEN critical=0 AND false_alert=1 THEN 1 ELSE 0 END)
                                   / SUM(CASE WHEN critical=0 THEN 1 ELSE 0 END)
                        ELSE NULL END AS false_alert_rate,
                   SUM(CASE WHEN critical=1 THEN 1 ELSE 0 END)
                       AS critical_rows,
                   SUM(CASE WHEN critical=1 AND critical_miss=1 THEN 1 ELSE 0 END)
                       AS critical_misses,
                   CASE WHEN SUM(CASE WHEN critical=1 THEN 1 ELSE 0 END)>0
                        THEN 1.0 * SUM(CASE WHEN critical=1 AND critical_miss=1 THEN 1 ELSE 0 END)
                                   / SUM(CASE WHEN critical=1 THEN 1 ELSE 0 END)
                        ELSE NULL END AS critical_miss_rate,
                   AVG(total_wire_bytes) AS mean_wire_bytes,
                   AVG(energy_j) AS mean_gpu_board_energy_j,
                   SUM(CASE WHEN selected_mode IN ('L1_ALERT','L2_SEMANTICS')
                                  AND calibrated_correctness_probability>=?
                            THEN 1 ELSE 0 END)
                       AS high_confidence_semantic_rows,
                   SUM(CASE WHEN selected_mode IN ('L1_ALERT','L2_SEMANTICS')
                                  AND calibrated_correctness_probability>=?
                                  AND correct_action=0
                            THEN 1 ELSE 0 END)
                       AS high_confidence_semantic_errors,
                   AVG(evidence_escalated) AS evidence_escalation_rate
            FROM trials WHERE {predicate}
            GROUP BY policy, visual_group_id
            ORDER BY policy, visual_group_id
            """,
            connection,
            params=(
                float(semantic_acceptance_threshold),
                float(semantic_acceptance_threshold),
            ),
        )
        cells["high_confidence_semantic_error_rate"] = np.where(
            cells["high_confidence_semantic_rows"].gt(0),
            cells["high_confidence_semantic_errors"]
            / cells["high_confidence_semantic_rows"],
            np.nan,
        )
        for policy, group in cells.groupby("policy", sort=True):
            rows.append({
                "subset": subset,
                "policy": str(policy),
                "trials": int(group["trials"].sum()),
                "independent_visual_groups": int(len(group)),
                "correct_timely_rate": float(group["correct_timely_rate"].mean()),
                "noncritical_rows": int(group["noncritical_rows"].sum()),
                "false_alerts": int(group["false_alerts"].sum()),
                "false_alert_rate": float(group["false_alert_rate"].mean()),
                "critical_rows": int(group["critical_rows"].sum()),
                "critical_misses": int(group["critical_misses"].sum()),
                "critical_miss_rate": float(group["critical_miss_rate"].mean()),
                "mean_wire_bytes": float(group["mean_wire_bytes"].mean()),
                "mean_gpu_board_energy_j": float(group["mean_gpu_board_energy_j"].mean()),
                "high_confidence_semantic_rows": int(
                    group["high_confidence_semantic_rows"].sum()
                ),
                "high_confidence_semantic_errors": int(
                    group["high_confidence_semantic_errors"].sum()
                ),
                "high_confidence_semantic_visual_groups": int(
                    group["high_confidence_semantic_rows"].gt(0).sum()
                ),
                "high_confidence_semantic_error_rate": float(
                    group["high_confidence_semantic_error_rate"].mean()
                ),
                "semantic_acceptance_threshold": float(
                    semantic_acceptance_threshold
                ),
                "evidence_escalation_rate": float(
                    group["evidence_escalation_rate"].mean()
                ),
                "ood_claim_applicable": False,
                "energy_scope": str(energy_scope),
            })
    return pd.DataFrame(rows).sort_values(
        ["subset", "policy"], kind="mergesort",
    ).reset_index(drop=True)


def _controller_frame(
    connection: sqlite3.Connection,
    policies: Sequence[str],
    *,
    reference_policy: str,
    semantic_acceptance_threshold: float,
) -> pd.DataFrame:
    """Stream ordered controller rows and retain only policy accumulators."""

    if reference_policy not in set(map(str, policies)):
        raise StudyGateError(
            f"Controller reduction lacks reference policy {reference_policy!r}"
        )
    connection.execute(
        "CREATE TEMP TABLE derived_quantiles "
        "(policy TEXT NOT NULL, metric TEXT NOT NULL, value REAL NOT NULL)"
    )
    accum: dict[str, defaultdict[str, float]] = {
        str(policy): defaultdict(float) for policy in policies
    }
    mode_counts: dict[str, defaultdict[str, int]] = {
        str(policy): defaultdict(int) for policy in policies
    }
    # A reference self-join avoids holding one outcome per raw trial in RAM.
    columns = [column for column in REQUIRED_COLUMNS if column not in PAIR_COLUMNS]
    select = ", ".join(f"t.{column}" for column in PAIR_COLUMNS)
    select += ", " + ", ".join(f"t.{column}" for column in columns)
    select += ", r.correct_timely_action AS reference_correct_timely_action"
    join = " AND ".join(
        [f"r.{column}=t.{column}" for column in PAIR_COLUMNS]
        + ["r.episode_id=t.episode_id", "r.policy=?"]
    )
    cursor = connection.execute(
        f"""
        SELECT {select}
        FROM trials AS t
        LEFT JOIN trials AS r ON {join}
        ORDER BY t.policy, t.trace_id, t.replicate_seed, t.episode_id,
                 t.decision_us, t.sample_id
        """,
        (reference_policy,),
    )
    names = [item[0] for item in cursor.description]
    current_key: tuple[Any, ...] | None = None
    previous_representation: str | None = None
    two_back_representation: str | None = None
    run_start_us = 0
    run_decisions = 0
    episode_decisions = 0
    episode_outages = 0
    episode_recovered_declared = 0
    episode_recovery_events = 0

    def finish_run(policy: str, *, next_start_us: int | None) -> None:
        nonlocal run_decisions
        if run_decisions <= 0:
            return
        target = accum[policy]
        target["dwell_runs"] += 1
        connection.execute(
            "INSERT INTO derived_quantiles VALUES (?, 'dwell_decisions', ?)",
            (policy, float(run_decisions)),
        )
        if next_start_us is not None:
            target["completed_dwell_runs"] += 1
            connection.execute(
                "INSERT INTO derived_quantiles VALUES (?, 'completed_dwell_ms', ?)",
                (policy, (int(next_start_us) - int(run_start_us)) / 1000.0),
            )

    def finish_episode(policy: str) -> None:
        if current_key is None:
            return
        finish_run(policy, next_start_us=None)
        target = accum[policy]
        target["episodes"] += 1
        target["switch_opportunities"] += max(episode_decisions - 1, 0)
        target["oscillation_opportunities"] += max(episode_decisions - 2, 0)
        target["outages"] += episode_outages
        target["outages_recovered"] += episode_recovery_events
        if episode_recovery_events != episode_recovered_declared:
            raise StudyGateError(
                "Controller outage recovery annotations are internally inconsistent"
            )

    for raw in cursor:
        row = dict(zip(names, raw))
        if row["reference_correct_timely_action"] is None:
            raise StudyGateError("Controller row lacks paired lookahead-reference outcome")
        policy = str(row["policy"])
        key = (
            policy, str(row["trace_id"]), int(row["replicate_seed"]),
            str(row["episode_id"]),
        )
        representation = f"{row['selected_mode']}:{row['variant']}"
        decision_us = int(row["decision_us"])
        if current_key != key:
            if current_key is not None:
                finish_episode(str(current_key[0]))
            current_key = key
            previous_representation = None
            two_back_representation = None
            run_start_us = decision_us
            run_decisions = 0
            episode_decisions = 0
            episode_outages = int(row["episode_outage_count"])
            episode_recovered_declared = int(row["episode_outages_recovered"])
            episode_recovery_events = 0
        target = accum[policy]
        switched = previous_representation is not None and representation != previous_representation
        oscillated = (
            two_back_representation is not None
            and representation == two_back_representation
            and representation != previous_representation
        )
        if switched:
            finish_run(policy, next_start_us=decision_us)
            run_start_us = decision_us
            run_decisions = 0
            target["mode_switches"] += 1
        if oscillated:
            target["oscillations_aba"] += 1
        run_decisions += 1
        episode_decisions += 1
        two_back_representation = previous_representation
        previous_representation = representation

        own_success = bool(row["correct_timely_action"])
        reference_success = bool(row["reference_correct_timely_action"])
        correct_action = bool(row["correct_action"])
        target["decisions"] += 1
        target["correct_timely_actions"] += int(own_success)
        target["total_energy_j"] += float(row["energy_j"])
        target["lookahead_reference_signed_gap_sum"] += int(reference_success) - int(own_success)
        target["lookahead_reference_missed_opportunities"] += int(reference_success and not own_success)
        target["lookahead_reference_inversions"] += int(own_success and not reference_success)
        target["payload_bytes"] += float(row["application_payload_bytes"])
        target["uplink_wire_bytes"] += float(row["original_uplink_bytes"]) + float(row["retry_uplink_bytes"])
        complete = bool(row["full_message_received"]) and bool(row["state_valid"])
        if not complete:
            target["aborted_generated_payload_bytes"] += float(row["application_payload_bytes"])
        target["wasted_uplink_wire_bytes"] += float(row["retry_uplink_bytes"])
        if not own_success:
            target["wasted_uplink_wire_bytes"] += float(row["original_uplink_bytes"])
        target["queue_pending_sum"] += float(row["decision_queue_pending_bytes"])
        target["outstanding_unacked_sum"] += float(row["decision_queue_outstanding_unacked_bytes"])
        target["queue_messages_sum"] += float(row["decision_queue_messages"])
        target["queue_oldest_age_sum"] += float(row["decision_queue_oldest_age_ms"])
        target["backlog_sum"] += float(row["decision_backlog_bytes"])
        target["queue_pending_max"] = max(
            target["queue_pending_max"], float(row["decision_queue_pending_bytes"])
        )
        stateful = str(row["state_kind"]).lower() not in {"none", "statekind.none"}
        target["stateful_decisions"] += int(stateful)
        target["stale_state_events"] += int(str(row["apply_status"]) == "stale")
        target["sequence_gap_rejections"] += int(str(row["apply_status"]) == "base_mismatch")
        target["decision_sequence_gap_events"] += int(int(row["decision_sender_sequence_gap"]) > 0)
        recovery = bool(row["outage_recovery_event"])
        episode_recovery_events += int(recovery)
        if recovery and row["outage_recovery_latency_ms"] is not None:
            connection.execute(
                "INSERT INTO derived_quantiles VALUES (?, 'outage_recovery_ms', ?)",
                (policy, float(row["outage_recovery_latency_ms"])),
            )
        semantic = str(row["selected_mode"]) in {"L1_ALERT", "L2_SEMANTICS"}
        score = row["calibrated_correctness_probability"]
        accepted = bool(
            semantic and score is not None
            and 0.0 <= float(score) <= 1.0
            and float(score) >= float(semantic_acceptance_threshold)
        )
        target["semantic_predictions"] += int(semantic)
        target["semantic_accepted"] += int(accepted)
        target["semantic_accepted_wrong"] += int(accepted and not correct_action)
        target["semantic_accepted_mission_failure"] += int(accepted and not own_success)
        target["high_confidence_semantic_errors"] += int(
            accepted and not correct_action
        )
        mode_counts[policy][str(row["selected_mode"]).lower()] += 1
        connection.execute(
            "INSERT INTO derived_quantiles VALUES (?, 'queue_pending_bytes', ?)",
            (policy, float(row["decision_queue_pending_bytes"])),
        )
        connection.execute(
            "INSERT INTO derived_quantiles VALUES (?, 'backlog_bytes', ?)",
            (policy, float(row["decision_backlog_bytes"])),
        )
    if current_key is not None:
        finish_episode(str(current_key[0]))
    connection.commit()

    rows: list[dict[str, Any]] = []
    all_modes = (
        "l0_heartbeat", "l1_alert", "l2_semantics", "l3_semantics_roi",
        "l4_features", "l5_image_or_clip",
    )
    for policy in sorted(map(str, policies)):
        item = accum[policy]
        decisions = int(item["decisions"])
        correct = int(item["correct_timely_actions"])
        stateful = int(item["stateful_decisions"])
        accepted = int(item["semantic_accepted"])
        semantic = int(item["semantic_predictions"])
        outages = int(item["outages"])
        recovered = int(item["outages_recovered"])
        row: dict[str, Any] = {
            "policy": policy,
            "episodes": int(item["episodes"]),
            "decisions": decisions,
            "correct_timely_actions": correct,
            "total_energy_j": float(item["total_energy_j"]),
            "joules_per_correct_timely_action": (
                float(item["total_energy_j"]) / correct if correct else np.nan
            ),
            "mode_switches": int(item["mode_switches"]),
            "switch_opportunities": int(item["switch_opportunities"]),
            "mode_switch_rate": (
                float(item["mode_switches"]) / item["switch_opportunities"]
                if item["switch_opportunities"] else 0.0
            ),
            "oscillations_aba": int(item["oscillations_aba"]),
            "oscillation_opportunities": int(item["oscillation_opportunities"]),
            "oscillation_rate": (
                float(item["oscillations_aba"]) / item["oscillation_opportunities"]
                if item["oscillation_opportunities"] else 0.0
            ),
            "dwell_runs": int(item["dwell_runs"]),
            "completed_dwell_runs": int(item["completed_dwell_runs"]),
            "mean_dwell_decisions": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='dwell_decisions'", parameters=(policy,),
                quantile=0.5,
            ),
            "median_dwell_decisions": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='dwell_decisions'", parameters=(policy,),
                quantile=0.5,
            ),
            "mean_completed_dwell_ms": float(connection.execute(
                "SELECT AVG(value) FROM derived_quantiles "
                "WHERE policy=? AND metric='completed_dwell_ms'", (policy,),
            ).fetchone()[0] or np.nan),
            "median_completed_dwell_ms": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='completed_dwell_ms'", parameters=(policy,),
                quantile=0.5,
            ),
            "p95_completed_dwell_ms": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='completed_dwell_ms'", parameters=(policy,),
                quantile=0.95,
            ),
            "aborted_generated_payload_bytes": float(item["aborted_generated_payload_bytes"]),
            "aborted_generated_payload_fraction": (
                float(item["aborted_generated_payload_bytes"]) / item["payload_bytes"]
                if item["payload_bytes"] else np.nan
            ),
            "wasted_uplink_wire_bytes": float(item["wasted_uplink_wire_bytes"]),
            "wasted_uplink_wire_fraction": (
                float(item["wasted_uplink_wire_bytes"]) / item["uplink_wire_bytes"]
                if item["uplink_wire_bytes"] else np.nan
            ),
            "mean_queue_pending_bytes_at_decision": float(item["queue_pending_sum"]) / decisions,
            "p95_queue_pending_bytes_at_decision": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='queue_pending_bytes'", parameters=(policy,),
                quantile=0.95,
            ),
            "max_queue_pending_bytes_at_decision": float(item["queue_pending_max"]),
            "mean_outstanding_unacked_bytes_at_decision": float(item["outstanding_unacked_sum"]) / decisions,
            "mean_queue_messages_at_decision": float(item["queue_messages_sum"]) / decisions,
            "mean_queue_oldest_age_ms_at_decision": float(item["queue_oldest_age_sum"]) / decisions,
            "mean_backlog_bytes_at_decision": float(item["backlog_sum"]) / decisions,
            "p95_backlog_bytes_at_decision": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='backlog_bytes'", parameters=(policy,),
                quantile=0.95,
            ),
            "stateful_decisions": stateful,
            "stale_state_events": int(item["stale_state_events"]),
            "stale_state_rate": float(item["stale_state_events"]) / stateful if stateful else np.nan,
            "sequence_gap_rejections": int(item["sequence_gap_rejections"]),
            "sequence_gap_rejection_rate": float(item["sequence_gap_rejections"]) / stateful if stateful else np.nan,
            "decision_sequence_gap_rate": float(item["decision_sequence_gap_events"]) / decisions,
            "outages": outages,
            "outages_recovered": recovered,
            "outages_unrecovered": outages - recovered,
            "outage_recovery_rate": recovered / outages if outages else np.nan,
            "median_outage_recovery_ms": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='outage_recovery_ms'", parameters=(policy,),
                quantile=0.5,
            ),
            "p95_outage_recovery_ms": _exact_sql_quantile(
                connection, table="derived_quantiles", expression="value",
                where="policy=? AND metric='outage_recovery_ms'", parameters=(policy,),
                quantile=0.95,
            ),
            "mean_lookahead_reference_signed_gap": float(item["lookahead_reference_signed_gap_sum"]) / decisions,
            "lookahead_reference_signed_gap_percentage_points": 100.0 * float(item["lookahead_reference_signed_gap_sum"]) / decisions,
            "lookahead_reference_missed_opportunity_rate": float(item["lookahead_reference_missed_opportunities"]) / decisions,
            "lookahead_reference_inversion_rate": float(item["lookahead_reference_inversions"]) / decisions,
            "semantic_predictions": semantic,
            "semantic_accepted": accepted,
            "semantic_selective_coverage": accepted / decisions,
            "semantic_acceptance_rate_within_l1_l2": accepted / semantic if semantic else np.nan,
            "semantic_selective_risk": float(item["semantic_accepted_wrong"]) / accepted if accepted else np.nan,
            "semantic_selective_mission_failure_rate": float(item["semantic_accepted_mission_failure"]) / accepted if accepted else np.nan,
            "high_confidence_semantic_rows": accepted,
            "high_confidence_semantic_errors": int(
                item["high_confidence_semantic_errors"]
            ),
            "high_confidence_semantic_error_rate": (
                float(item["high_confidence_semantic_errors"]) / accepted
                if accepted else np.nan
            ),
            "semantic_acceptance_threshold": float(semantic_acceptance_threshold),
        }
        # Mean differs from median for arbitrary run lengths; compute it
        # explicitly while keeping every raw run only in SQLite.
        mean_dwell = connection.execute(
            "SELECT AVG(value) FROM derived_quantiles "
            "WHERE policy=? AND metric='dwell_decisions'", (policy,),
        ).fetchone()[0]
        row["mean_dwell_decisions"] = float(mean_dwell) if mean_dwell is not None else np.nan
        for mode in all_modes:
            row[f"mode_frequency_{mode}"] = mode_counts[policy][mode] / decisions
        rows.append(row)
    return pd.DataFrame(rows)


def compact_two_way_clustered_paired_bootstrap(
    cells: pd.DataFrame,
    method: str,
    reference: str,
    *,
    endpoint: str,
    repetitions: int = 10_000,
    seed: int = 20260823,
    confidence: float = 0.95,
) -> BootstrapEffect:
    """Exact two-way bootstrap from compact visual-group/trace cells.

    The raw estimator first averages paired decision deltas within every
    visual-group/trace cell.  ``policy_pair_cells`` and
    ``h4_matched_coverage_cells`` store exactly those sufficient statistics,
    so resampling this table reproduces the registered cluster estimand while
    avoiding the raw decisions entirely.
    """

    required = {"visual_group_id", "trace_id", "policy", endpoint}
    if missing := required.difference(cells.columns):
        raise ValueError(f"Compact bootstrap lacks columns: {sorted(missing)}")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    subset = cells[cells["policy"].astype(str).isin([method, reference])].copy()
    if set(subset["policy"].astype(str)) != {method, reference}:
        raise StudyGateError("Compact bootstrap is missing a paired policy")
    keys = ["visual_group_id", "trace_id"]
    if subset.duplicated([*keys, "policy"]).any():
        raise StudyGateError("Compact bootstrap has duplicate cluster-policy cells")
    pivot = subset.pivot(index=keys, columns="policy", values=endpoint)
    if method not in pivot or reference not in pivot or pivot[[method, reference]].isna().any().any():
        raise StudyGateError("Compact bootstrap cluster cells are not fully paired")
    pivot = pivot.sort_index()
    pivot["delta"] = (
        pd.to_numeric(pivot[method], errors="raise")
        - pd.to_numeric(pivot[reference], errors="raise")
    )
    visuals = sorted(map(str, subset["visual_group_id"].unique()))
    traces = sorted(map(str, subset["trace_id"].unique()))
    full_index = pd.MultiIndex.from_product(
        [visuals, traces], names=keys,
    )
    values = pivot["delta"].reindex(full_index).to_numpy(float).reshape(
        len(visuals), len(traces),
    )
    presence = np.isfinite(values).astype(float)
    value_matrix = np.nan_to_num(values, nan=0.0)
    if not presence.any():
        raise StudyGateError("Compact bootstrap has no observed cluster cells")
    estimate = float(value_matrix.sum() / presence.sum())
    rng = np.random.default_rng(int(seed))
    draws = np.full(int(repetitions), np.nan, dtype=float)
    entries = max(1, len(visuals) + len(traces))
    batch_size = max(1, min(int(repetitions), 2_000_000 // entries))
    for start in range(0, int(repetitions), batch_size):
        stop = min(int(repetitions), start + batch_size)
        count = stop - start
        visual_weights = np.empty((count, len(visuals)), dtype=np.int64)
        trace_weights = np.empty((count, len(traces)), dtype=np.int64)
        # Match the frozen raw estimator's per-repetition RNG order.
        for offset in range(count):
            visual_weights[offset] = np.bincount(
                rng.choice(len(visuals), len(visuals), replace=True),
                minlength=len(visuals),
            )
            trace_weights[offset] = np.bincount(
                rng.choice(len(traces), len(traces), replace=True),
                minlength=len(traces),
            )
        numerators = np.einsum(
            "bi,ij,bj->b", visual_weights, value_matrix, trace_weights,
            optimize=True,
        )
        denominators = np.einsum(
            "bi,ij,bj->b", visual_weights, presence, trace_weights,
            optimize=True,
        )
        valid = denominators > 0
        block = np.full(count, np.nan)
        block[valid] = numerators[valid] / denominators[valid]
        draws[start:stop] = block
    finite = draws[np.isfinite(draws)]
    if len(finite) < int(repetitions) * 0.99:
        raise StudyGateError("Compact bootstrap produced too many empty cluster draws")
    alpha = 1.0 - float(confidence)
    low, high = np.quantile(finite, [alpha / 2.0, 1.0 - alpha / 2.0])
    null = finite - estimate
    p = (1 + np.sum(np.abs(null) >= abs(estimate))) / (len(null) + 1)
    return BootstrapEffect(
        str(method), str(reference), str(endpoint), estimate,
        float(low), float(high), float(min(1.0, p)), len(finite),
        len(visuals), len(traces),
    )


def reduce_policy_trial_partitions(
    index: str | Path | pd.DataFrame | Sequence[PolicyPartition | Mapping[str, Any]],
    output_dir: str | Path,
    *,
    chunk_size: int = 50_000,
    h4_policies: Sequence[str] = (
        "proposed_full", "ablate_evidence_escalation",
    ),
    h4_target_coverage: float = 0.80,
    reference_policy: str = "heuristic_lookahead_reference",
    semantic_acceptance_threshold: float = 0.80,
    energy_scope: str = (
        "combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system"
    ),
) -> PolicyReductionResult:
    """Reduce checksummed policy partitions with a bounded-memory workflow."""

    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    h4_policies = tuple(dict.fromkeys(map(str, h4_policies)))
    if len(h4_policies) != 2:
        raise ValueError("H4 reduction requires exactly two registered policies")
    if not 0.0 < float(h4_target_coverage) <= 1.0:
        raise ValueError("h4_target_coverage must lie in (0, 1]")
    if not 0.0 <= float(semantic_acceptance_threshold) <= 1.0:
        raise ValueError("semantic_acceptance_threshold must lie in [0, 1]")

    partitions = load_verified_partition_index(index)
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix=".policy-reducer-", dir=output))
    database_path = workspace / "policy_trials.sqlite3"
    h4_directory = workspace / "h4"
    h4_directory.mkdir()
    connection = sqlite3.connect(database_path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    escaped_workspace = str(workspace).replace("'", "''")
    connection.execute(f"PRAGMA temp_store_directory='{escaped_workspace}'")

    input_columns: tuple[str, ...] | None = None
    max_chunk_rows = 0
    verified_rows = 0
    h4_connections: dict[str, sqlite3.Connection] = {}
    h4_paths: dict[str, Path] = {}
    try:
        for partition_number, partition in enumerate(partitions):
            columns = _csv_columns(partition.path)
            missing = set((*REQUIRED_COLUMNS, *OPTIONAL_ENERGY_COLUMNS)).difference(columns)
            if missing:
                raise StudyGateError(
                    f"Policy partition lacks reducer columns {sorted(missing)}: {partition.path}"
                )
            selected = tuple(column for column in (*REQUIRED_COLUMNS, *OPTIONAL_ENERGY_COLUMNS) if column in columns)
            if input_columns is None:
                input_columns = selected
                definitions = ", ".join(
                    f'"{column}" {_sqlite_type(column)}' for column in input_columns
                )
                primary = ", ".join(f'"{column}"' for column in (*PAIR_COLUMNS, "episode_id", "policy"))
                connection.execute(
                    f"CREATE TABLE trials ({definitions}, PRIMARY KEY ({primary})) WITHOUT ROWID"
                )
                connection.execute(
                    "CREATE INDEX trials_policy ON trials(policy)"
                )
                connection.execute(
                    "CREATE INDEX trials_episode_order ON trials"
                    "(policy, trace_id, replicate_seed, episode_id, decision_us, sample_id)"
                )
                connection.execute(
                    "CREATE INDEX trials_hypothesis_cells ON trials"
                    "(visual_group_id, trace_id, policy)"
                )
            elif selected != input_columns:
                raise StudyGateError(
                    "Policy partition columns differ in reducer-relevant fields"
                )
            quoted = ", ".join(f'"{column}"' for column in input_columns)
            placeholders = ", ".join("?" for _ in input_columns)
            insert = f"INSERT INTO trials ({quoted}) VALUES ({placeholders})"
            partition_rows = 0
            for chunk in pd.read_csv(
                partition.path,
                compression="infer",
                usecols=list(input_columns),
                chunksize=int(chunk_size),
                low_memory=False,
            ):
                # v3r2 correction (receipted in v3r1): pandas returns
                # ``usecols`` in file order, while the INSERT below binds
                # values in ``input_columns`` order.
                chunk = chunk[list(input_columns)]
                max_chunk_rows = max(max_chunk_rows, len(chunk))
                partition_rows += len(chunk)
                verified_rows += len(chunk)
                try:
                    connection.executemany(
                        insert,
                        (
                            tuple(_python_value(value, column) for column, value in zip(input_columns, row))
                            for row in chunk.itertuples(index=False, name=None)
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise StudyGateError(
                        "Duplicate paired policy row detected while reducing partitions"
                    ) from exc
                h4_chunk = chunk[chunk["policy"].astype(str).isin(h4_policies)]
                for trace_id, trace_rows in h4_chunk.groupby("trace_id", sort=True):
                    trace_id = str(trace_id)
                    if trace_id not in h4_connections:
                        path = h4_directory / f"h4-{_h4_trace_token(trace_id)}.sqlite3"
                        h4_paths[trace_id] = path
                        h4_connections[trace_id] = _create_h4_spill(path)
                    h4_connection = h4_connections[trace_id]
                    h4_connection.executemany(
                        """
                        INSERT INTO h4_trials VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            (
                                str(row.sample_id), str(row.visual_group_id),
                                str(row.mission_profile), str(row.trace_id),
                                int(row.replicate_seed), str(row.policy),
                                str(row.selected_mode),
                                (
                                    None
                                    if pd.isna(row.calibrated_correctness_probability)
                                    else float(row.calibrated_correctness_probability)
                                ),
                                int(bool(row.correct_action)),
                            )
                            for row in trace_rows[[
                                "sample_id", "visual_group_id", "mission_profile",
                                "trace_id", "replicate_seed", "policy", "selected_mode",
                                "calibrated_correctness_probability", "correct_action",
                            ]].itertuples(index=False)
                        ),
                    )
            connection.commit()
            if partition_rows != int(partition.row_count):
                raise StudyGateError(
                    f"Policy partition row count changed: {partition.path} "
                    f"({partition_rows} != {partition.row_count})"
                )

        if input_columns is None or verified_rows == 0:
            raise StudyGateError("Policy partitions contain no trial rows")
        for h4_connection in h4_connections.values():
            h4_connection.commit()
            h4_connection.close()
        h4_connections.clear()

        policies = [
            str(row[0]) for row in connection.execute(
                "SELECT DISTINCT policy FROM trials ORDER BY policy"
            )
        ]
        if set(h4_policies).difference(policies):
            raise StudyGateError("Campaign lacks a registered H4 policy")
        if reference_policy not in policies:
            raise StudyGateError("Campaign lacks the registered lookahead reference")
        bad_pairs = int(connection.execute(
            f"""
            SELECT COUNT(*) FROM (
                SELECT {', '.join(PAIR_COLUMNS)}, episode_id,
                       COUNT(DISTINCT policy) AS policies
                FROM trials
                GROUP BY {', '.join(PAIR_COLUMNS)}, episode_id
                HAVING policies != ?
            )
            """,
            (len(policies),),
        ).fetchone()[0])
        if bad_pairs:
            raise StudyGateError(
                f"Policy campaign has {bad_pairs} incomplete paired trial keys"
            )

        pair_cells, h3_episode_cells, resource = _resource_and_pair_frames(
            connection, policies, float(semantic_acceptance_threshold),
        )
        resource_episode_cells, resource_cluster_cells, max_episode_captures = (
            _episode_resource_frames(connection)
        )
        resource_rates = _resource_rate_frame(resource_episode_cells, policies)
        exp5 = _exp5_frame(
            connection, energy_scope,
            float(semantic_acceptance_threshold),
        )
        controller = _controller_frame(
            connection, policies,
            reference_policy=reference_policy,
            semantic_acceptance_threshold=float(semantic_acceptance_threshold),
        )

        h4_cells_rows: list[dict[str, Any]] = []
        h4_audit_rows: list[dict[str, Any]] = []
        h4_index_rows: list[dict[str, Any]] = []
        trace_output = output / "h4_trace_cells"
        trace_output.mkdir(exist_ok=True)
        for trace_id in sorted(h4_paths):
            cells, audit = _reduce_h4_trace(
                h4_paths[trace_id], trace_id, h4_policies,
                float(h4_target_coverage),
                float(semantic_acceptance_threshold),
            )
            # Lists contain one row per bounded cluster cell, never raw trials.
            h4_cells_rows.extend(cells.to_dict("records"))
            h4_audit_rows.extend(audit.to_dict("records"))
            path = trace_output / f"h4_cells_{_h4_trace_token(trace_id)}.csv.gz"
            atomic_write_frame(path, cells)
            h4_index_rows.append({
                "trace_id": trace_id,
                "artifact_path": str(path),
                "artifact_sha256": hash_file(path),
                "row_count": int(len(cells)),
            })
        h4_cells = pd.DataFrame(h4_cells_rows).sort_values(
            ["trace_id", "visual_group_id", "policy"], kind="mergesort",
        ).reset_index(drop=True)
        per_trace_audit = pd.DataFrame(h4_audit_rows)
        h4_audit: list[dict[str, Any]] = []
        for policy, group in h4_cells.groupby("policy", sort=True):
            h4_audit.append({
                "policy": policy,
                "target_coverage": float(h4_target_coverage),
                "visual_group_trace_cells": int(len(group)),
                "trace_realizations": int(group["trace_id"].nunique()),
                "available_semantic_rows": int(group["available_semantic_rows"].sum()),
                "selected_rows": int(group["selected_rows"].sum()),
                "realized_group_trace_equal_coverage": float(group["realized_coverage"].mean()),
                "group_trace_equal_selective_risk": float(
                    group["matched_coverage_semantic_error"].mean()
                ),
                "group_equal_selective_risk": float(
                    group["matched_coverage_semantic_error"].mean()
                ),
                "high_confidence_semantic_rows": int(
                    group["high_confidence_semantic_rows"].sum()
                ),
                "high_confidence_semantic_errors": int(
                    group["high_confidence_semantic_errors"].sum()
                ),
                "group_trace_equal_high_confidence_semantic_error_rate": float(
                    group["high_confidence_semantic_error_rate"].mean()
                ),
                "zero_high_confidence_visual_group_trace_cells": int(
                    (~group["high_confidence_semantic_passed"]).sum()
                ),
                "matched_coverage_passed": bool(
                    group["matched_coverage_passed"].all() and len(group)
                ),
                "high_confidence_semantic_passed": bool(
                    group["high_confidence_semantic_passed"].all() and len(group)
                ),
                "semantic_acceptance_threshold": float(
                    semantic_acceptance_threshold
                ),
                "insufficient_visual_group_trace_cells": int((~group["passed"]).sum()),
                "passed": bool(group["passed"].all() and len(group)),
                "selection_rule": (
                    "top_calibrated_semantic_score_within_visual_group_trace"
                ),
            })

        pairing_audit = pd.DataFrame([{
            "partition_count": int(len(partitions)),
            "verified_trial_rows": int(verified_rows),
            "policies": int(len(policies)),
            "incomplete_pair_keys": int(bad_pairs),
            "passed": bool(bad_pairs == 0),
            "pair_key": json.dumps([*PAIR_COLUMNS, "episode_id"], separators=(",", ":")),
        }])
        io_audit = pd.DataFrame([{
            "reducer_version": REDUCER_VERSION,
            "source_partitions": int(len(partitions)),
            "verified_rows": int(verified_rows),
            "configured_chunk_size": int(chunk_size),
            "maximum_observed_chunk_rows": int(max_chunk_rows),
            "maximum_episode_capture_values_in_memory": int(max_episode_captures),
            "h4_trace_spills": int(len(h4_paths)),
            "h4_raw_selection_backend": "one_sqlite_database_per_trace",
            "h4_campaign_selected_trial_rows_materialized": False,
            "exact_quantile_backend": "sqlite_order_statistics_linear",
            "passed": bool(max_chunk_rows <= int(chunk_size)),
        }])
        frames = {
            "policy_pair_cells": pair_cells,
            "h3_negative_episode_cells": h3_episode_cells,
            "resource_summary": resource,
            "resource_rate_summary": resource_rates,
            "resource_episode_cells": resource_episode_cells,
            "resource_cluster_cells": resource_cluster_cells,
            "controller_summary": controller,
            "exp5_summary": exp5,
            "h4_matched_coverage_cells": h4_cells,
            "h4_matched_coverage_audit": pd.DataFrame(h4_audit),
            "h4_per_trace_audit": per_trace_audit,
            "h4_trace_spill_index": pd.DataFrame(h4_index_rows),
            "pairing_audit": pairing_audit,
            "io_audit": io_audit,
        }
        artifacts = _write_compact_frames(output, frames)
        receipt = {
            "version": REDUCER_VERSION,
            "source_partitions": [
                {
                    "path": str(part.path),
                    "sha256": part.sha256,
                    "row_count": int(part.row_count),
                    "trace_id": part.trace_id,
                    "replicate_seed": part.replicate_seed,
                    "stage": part.stage,
                }
                for part in partitions
            ],
            "parameters": {
                "chunk_size": int(chunk_size),
                "h4_policies": list(h4_policies),
                "h4_target_coverage": float(h4_target_coverage),
                "reference_policy": str(reference_policy),
                "semantic_acceptance_threshold": float(semantic_acceptance_threshold),
                "energy_scope": str(energy_scope),
            },
            "artifacts": {
                name: {
                    "path": str(path),
                    "sha256": hash_file(path),
                    "row_count": int(len(frames[name])),
                }
                for name, path in sorted(artifacts.items())
            },
            "io_audit": io_audit.iloc[0].to_dict(),
        }
        receipt_path = atomic_write_json(
            output / "policy_reduction_receipt.json", receipt,
        )
        artifacts["receipt"] = receipt_path
        return PolicyReductionResult(artifacts, frames, receipt_path)
    finally:
        for h4_connection in h4_connections.values():
            h4_connection.close()
        connection.close()
        shutil.rmtree(workspace, ignore_errors=True)


__all__ = [
    "PAIR_COLUMNS",
    "PolicyPartition",
    "PolicyReductionResult",
    "REDUCER_VERSION",
    "REQUIRED_COLUMNS",
    "compact_two_way_clustered_paired_bootstrap",
    "load_verified_partition_index",
    "reduce_policy_trial_partitions",
]

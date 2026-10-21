"""Frozen model assets and detector preparation for the VisDrone paper study.

This module has three deliberately bounded responsibilities:

* resolve the registered Hugging Face repositories to immutable commits,
  download those exact snapshots, and freeze a byte-level ``MODEL_LOCK.json``;
* convert only the publisher's VisDrone DET train partition into an
  Ultralytics-compatible label tree, with a deterministic natural-group split
  for detector fitting and early stopping (publisher validation and test-dev
  are never accepted by this path); and
* exercise every active frozen VLM on one real image while recording every
  success or failure.  A failed model is a row in the smoke report, never an
  omission.

The helpers prepare and verify inputs.  They do not train a detector, choose a
winning model, open the untouched test partition, or create scientific claims.
"""

from __future__ import annotations

import contextlib
import csv
import gc
import io
import json
import math
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from decimal import Decimal, InvalidOperation
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
import yaml
from PIL import Image

from .perception import VISDRONE_REGISTERED_MAX_NEW_TOKENS
from .registry import (
    StudyGateError,
    atomic_write_json,
    atomic_write_text,
    hash_file,
    hash_object,
    sha256_bytes,
    source_tree_hash,
)


MODEL_LOCK_SCHEMA_VERSION = 1
MODEL_LOCK_TYPE = "semuav_huggingface_model_lock"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")

OFFICIAL_COMPACT_MODEL_REPOS: Mapping[str, str] = {
    "qwen3_vl_2b": "Qwen/Qwen3-VL-2B-Instruct",
    "smolvlm2_2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
}
OFFICIAL_MODEL_BACKENDS: Mapping[str, str] = {
    "qwen3_vl_2b": "qwen3_vl",
    "smolvlm2_2b": "smolvlm2",
}
OFFICIAL_UPPER_BOUND_ID = "qwen3_vl_8b_upper_bound"
OFFICIAL_UPPER_BOUND_REPO = "Qwen/Qwen3-VL-8B-Instruct"
OFFICIAL_RUNTIME_MODEL_IDS: tuple[str, ...] = (
    *OFFICIAL_COMPACT_MODEL_REPOS,
    OFFICIAL_UPPER_BOUND_ID,
)

# Moondream was removed prospectively from the active v2r6 design after its
# v2r5 train-canary response failed the strict one-call decision contract.  The
# constants and helpers below remain only so historical v2r5 evidence can be
# authenticated; they are not members of ``OFFICIAL_RUNTIME_MODEL_IDS`` and
# must never enter a v2r6-or-later active model lock, smoke report, or
# scientific seal.
HISTORICAL_MOONDREAM_MODEL_ID = "moondream2"
HISTORICAL_MOONDREAM_MODEL_REPO = "vikhyatk/moondream2"
HISTORICAL_V2R5_COMPACT_MODEL_REPOS: Mapping[str, str] = {
    **OFFICIAL_COMPACT_MODEL_REPOS,
    HISTORICAL_MOONDREAM_MODEL_ID: HISTORICAL_MOONDREAM_MODEL_REPO,
}
HISTORICAL_V2R5_MODEL_BACKENDS: Mapping[str, str] = {
    **OFFICIAL_MODEL_BACKENDS,
    HISTORICAL_MOONDREAM_MODEL_ID: "moondream2",
}
HISTORICAL_V2R5_RUNTIME_MODEL_IDS: tuple[str, ...] = (
    *HISTORICAL_V2R5_COMPACT_MODEL_REPOS,
    OFFICIAL_UPPER_BOUND_ID,
)
V2R6_MODEL_LOCK_DERIVATION_SCHEMA = (
    "semuav.visdrone-v2r6-three-model-lock-derivation.v1"
)

VISDRONE_DET_TRAIN_COMPONENT = "VisDrone2019-DET-train"
VISDRONE_DET_PUBLISHER_VALIDATION_COMPONENT = "VisDrone2019-DET-val"
VISDRONE_DET_TEST_COMPONENTS = frozenset(
    {"VisDrone2019-DET-test-dev", "VisDrone2019-DET-test-challenge"}
)
VISDRONE_DET_EXCLUDED_COMPONENTS = frozenset(
    {VISDRONE_DET_PUBLISHER_VALIDATION_COMPONENT, *VISDRONE_DET_TEST_COMPONENTS}
)
DETECTOR_TRAIN_PARTITION = "detector_train"
DETECTOR_INTERNAL_VALIDATION_PARTITION = "internal_val"
DETECTOR_SPLIT_VERSION = "visdrone.detector-study-train-group-hash.v2"
DETECTOR_SOURCE_AUDIT_SCHEMA = "semuav.visdrone-detector-source-audit.v1"
DETECTOR_STUDY_MANIFEST = "STUDY_SOURCE_AUDIT.jsonl"
DETECTOR_LEAKAGE_AUDIT = "DETECTOR_LEAKAGE_AUDIT.json"
DETECTOR_ALLOWED_STUDY_SPLIT = "train"
DETECTOR_FORBIDDEN_STUDY_SPLITS = frozenset({"calibration", "validation", "test"})
DETECTOR_TRAINING_PLAN_SCHEMA = "semuav.visdrone-detector-training-plan.v1"
DETECTOR_TRAINING_RECEIPT_SCHEMA = "semuav.visdrone-detector-training-receipt.v1"
DETECTOR_ENVIRONMENT_SCHEMA = "semuav.environment.v2"
VISDRONE_YOLO_NAMES: tuple[str, ...] = (
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning_tricycle",
    "bus",
    "motor",
)
VISDRONE_IGNORED_CATEGORY_IDS = frozenset({0, 11})
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
VISDRONE_RUNTIME_SMOKE_INPUT_CONTRACT = "semuav.visdrone-proposal-decision.v2"
RUNTIME_SMOKE_REPORT_SCHEMA_VERSION = 4
RUNTIME_SMOKE_ISOLATION_SCHEMA = "semuav.runtime-smoke-subprocess.v2"
RUNTIME_SMOKE_WORKER_INPUT_SCHEMA = "semuav.runtime-smoke-worker-input.v2"
RUNTIME_SMOKE_WORKER_RESULT_SCHEMA = "semuav.runtime-smoke-worker-result.v2"
RUNTIME_SMOKE_WORKER_TIMEOUT_SECONDS = 1800


RevisionResolver = Callable[[str, str | None], str]
SnapshotDownloader = Callable[[str, str, Path], str | Path]
OfflineSnapshotChecker = Callable[[str, str, Path], str | Path]
BackendFactory = Callable[[Mapping[str, Any], str, str], Any]
AuxiliaryArtifactDownloader = Callable[[str, str, str, Path], str | Path]
AuxiliaryArtifactChecker = Callable[[str, str, str, Path], str | Path]


# The historical pinned Moondream implementation constructs its lexical tokenizer with
# ``Tokenizer.from_pretrained("moondream/starmie-v1")``.  That repository is a
# separate runtime dependency; a same-named file in the Moondream model
# snapshot is not an identity-preserving substitute.  The revision below is
# the exact Hugging Face commit that changed the tokenizer bytes and repaired
# Moondream's reserved-token spellings.  A later audited repository head only
# updates README.md and is recorded separately; it is not the runtime pin.
MOONDREAM_TOKENIZER_REPO = "moondream/starmie-v1"
MOONDREAM_TOKENIZER_REVISION = "360559108ac4586b96328de81f817859a72fd5ca"
MOONDREAM_TOKENIZER_AUDITED_REPO_HEAD = "35192e10a54e36eabe0a7cc57a2c1aab371cafc5"
MOONDREAM_TOKENIZER_FILENAME = "tokenizer.json"
MOONDREAM_TOKENIZER_BYTES = 3_694_206
MOONDREAM_TOKENIZER_SHA256 = (
    "0512fdcac4a5f9e7746cbefce4a468dc93bf0f93f11e701b84f8b31bff199e9c"
)
MOONDREAM_TOKENIZER_LOCK_SCHEMA = "semuav.moondream-starmie-tokenizer-lock.v1"
MOONDREAM_TOKENIZER_LOCK_TYPE = "huggingface_single_file_auxiliary_dependency"


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Build one JSON object while rejecting ambiguous duplicate members."""

    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise StudyGateError(
                f"Moondream tokenizer lock contains duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _requested_revision(value: Any) -> str | None:
    """Return a usable remote revision, treating configuration sentinels as main."""

    text = str(value or "").strip()
    if not text or text.startswith("REQUIRED_") or text.startswith("${"):
        return None
    return text


def _model_entries(models: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate the active v2r13 two-compact-plus-upper-bound set."""

    compact = models.get("vlms")
    if not isinstance(compact, Sequence) or isinstance(compact, (str, bytes)):
        raise StudyGateError(
            "models.vlms must be a sequence of the two registered compact models"
        )

    observed: dict[str, Mapping[str, Any]] = {}
    for raw in compact:
        if not isinstance(raw, Mapping):
            raise StudyGateError("Every models.vlms entry must be a mapping")
        model_id = str(raw.get("id", "")).strip()
        if not model_id or model_id in observed:
            raise StudyGateError("Compact model IDs must be populated and unique")
        observed[model_id] = raw
    if set(observed) != set(OFFICIAL_COMPACT_MODEL_REPOS):
        raise StudyGateError(
            "The v2r13 VisDrone paper lock requires exactly the two registered compact "
            f"models: {sorted(OFFICIAL_COMPACT_MODEL_REPOS)}"
        )

    entries: list[dict[str, Any]] = []
    for model_id, expected_repo in OFFICIAL_COMPACT_MODEL_REPOS.items():
        raw = observed[model_id]
        repo = str(raw.get("repo", "")).strip()
        if repo != expected_repo:
            raise StudyGateError(
                f"{model_id} must use the official repository {expected_repo!r}, observed {repo!r}"
            )
        backend = str(raw.get("backend", "")).strip()
        if backend != OFFICIAL_MODEL_BACKENDS[model_id]:
            raise StudyGateError(
                f"{model_id} must use backend {OFFICIAL_MODEL_BACKENDS[model_id]!r}, "
                f"observed {backend!r}"
            )
        entries.append(
            {
                "model_id": model_id,
                "role": "compact",
                "backend": backend,
                "repo": repo,
                "requested_revision": _requested_revision(raw.get("revision")),
                "quantization": str(raw.get("quantization", "none")).strip(),
                "dtype": str(raw.get("dtype", "bfloat16")).strip(),
                "calibration_eligible": bool(raw.get("calibration_eligible", True)),
            }
        )

    upper = models.get("upper_bound")
    if not isinstance(upper, Mapping):
        raise StudyGateError("models.upper_bound must define the concrete Qwen upper bound")
    upper_id = str(upper.get("id", "")).strip()
    upper_repo = str(upper.get("repo", "")).strip()
    if upper_id != OFFICIAL_UPPER_BOUND_ID or upper_repo != OFFICIAL_UPPER_BOUND_REPO:
        raise StudyGateError(
            "The VisDrone paper upper bound must be "
            f"{OFFICIAL_UPPER_BOUND_ID!r} from {OFFICIAL_UPPER_BOUND_REPO!r}"
        )
    if str(upper.get("backend", "")).strip() != "qwen3_vl":
        raise StudyGateError("The registered Qwen upper bound must use backend 'qwen3_vl'")
    entries.append(
        {
            "model_id": upper_id,
            "role": "upper_bound",
            "backend": str(upper.get("backend", "qwen3_vl")).strip(),
            "repo": upper_repo,
            "requested_revision": _requested_revision(upper.get("revision")),
            "quantization": str(upper.get("quantization", "none")).strip(),
            "dtype": str(upper.get("dtype", "bfloat16")).strip(),
            "calibration_eligible": False,
        }
    )
    return entries


def validate_registered_model_configuration(models: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    """Public, side-effect-free validation of the registered model set."""

    return tuple(dict(entry) for entry in _model_entries(models))


def _model_set_fingerprint(models: Sequence[Mapping[str, Any]]) -> str:
    return hash_object(
        [
            {
                "model_id": item["model_id"],
                "role": item["role"],
                "backend": item["backend"],
                "repo": item["repo"],
                "resolved_revision": item["resolved_revision"],
                "quantization": item["quantization"],
                "dtype": item["dtype"],
                "calibration_eligible": item["calibration_eligible"],
                "inventory_sha256": item["inventory_sha256"],
            }
            for item in models
        ]
    )


def _default_resolver(repo: str, requested_revision: str | None) -> str:
    try:
        from huggingface_hub import HfApi  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in real setup
        raise StudyGateError("Install huggingface_hub before resolving model revisions") from exc
    info = HfApi().model_info(repo_id=repo, revision=requested_revision or "main")
    return str(info.sha)


def _default_downloader(repo: str, resolved_revision: str, cache_root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in real setup
        raise StudyGateError("Install huggingface_hub before downloading model snapshots") from exc
    return Path(
        snapshot_download(
            repo_id=repo,
            revision=resolved_revision,
            cache_dir=str(cache_root),
        )
    )


def _default_offline_checker(repo: str, resolved_revision: str, cache_root: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in real setup
        raise StudyGateError("Install huggingface_hub before checking offline snapshots") from exc
    return Path(
        snapshot_download(
            repo_id=repo,
            revision=resolved_revision,
            cache_dir=str(cache_root),
            local_files_only=True,
        )
    )


def _default_auxiliary_artifact_downloader(
    repo: str,
    resolved_revision: str,
    filename: str,
    cache_root: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in real setup
        raise StudyGateError(
            "Install huggingface_hub before downloading the Moondream tokenizer"
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=resolved_revision,
            cache_dir=str(cache_root),
        )
    )


def _default_auxiliary_artifact_checker(
    repo: str,
    resolved_revision: str,
    filename: str,
    cache_root: Path,
) -> Path:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only in real setup
        raise StudyGateError(
            "Install huggingface_hub before checking the Moondream tokenizer"
        ) from exc
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=resolved_revision,
            cache_dir=str(cache_root),
            local_files_only=True,
        )
    )


def _lexical_absolute(path: str | Path) -> Path:
    """Return an absolute path without dereferencing its final snapshot link."""

    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _reject_symlinked_path_components(path: Path, *, label: str) -> None:
    """Reject symlinks in an artifact/receipt's lexical parent chain."""

    for parent in (path.parent, *path.parent.parents):
        if parent.is_symlink():
            raise StudyGateError(f"{label} has a symlinked parent path")
        if parent.exists() and not parent.is_dir():
            raise StudyGateError(f"{label} has a non-directory parent path")


def _moondream_tokenizer_snapshot_path(cache_root: Path) -> Path:
    return (
        cache_root
        / "models--moondream--starmie-v1"
        / "snapshots"
        / MOONDREAM_TOKENIZER_REVISION
        / MOONDREAM_TOKENIZER_FILENAME
    )


def _verified_auxiliary_artifact_record(
    artifact_path: str | Path,
    *,
    cache_root: Path,
) -> dict[str, Any]:
    lexical = _lexical_absolute(artifact_path)
    expected = _moondream_tokenizer_snapshot_path(cache_root)
    if lexical != expected:
        raise StudyGateError(
            "Moondream tokenizer did not resolve to the exact pinned Starmie snapshot"
        )
    if not lexical.is_file():
        raise StudyGateError("Pinned Starmie tokenizer.json is missing")
    parent = lexical.parent
    while parent != cache_root:
        if parent.is_symlink() or not parent.is_dir():
            raise StudyGateError(
                "Pinned Starmie tokenizer snapshot has an unsafe parent path"
            )
        if cache_root not in parent.parents:
            raise StudyGateError(
                "Pinned Starmie tokenizer snapshot escapes the Hugging Face cache"
            )
        parent = parent.parent
    try:
        resolved = lexical.resolve(strict=True)
        resolved_relative = resolved.relative_to(cache_root)
    except (OSError, ValueError) as exc:
        raise StudyGateError(
            "Pinned Starmie tokenizer resolves outside the Hugging Face cache"
        ) from exc
    if lexical.is_symlink():
        expected_blob_root = (
            cache_root / "models--moondream--starmie-v1" / "blobs"
        )
        try:
            resolved.relative_to(expected_blob_root)
        except ValueError as exc:
            raise StudyGateError(
                "Pinned Starmie tokenizer link does not target its own repository blob"
            ) from exc
    record = {
        "filename": MOONDREAM_TOKENIZER_FILENAME,
        "snapshot_path": str(lexical),
        "snapshot_relative_path": lexical.relative_to(cache_root).as_posix(),
        "resolved_relative_path": resolved_relative.as_posix(),
        "bytes": lexical.stat().st_size,
        "sha256": hash_file(lexical),
    }
    if (
        record["bytes"] != MOONDREAM_TOKENIZER_BYTES
        or record["sha256"] != MOONDREAM_TOKENIZER_SHA256
    ):
        raise StudyGateError(
            "Pinned Starmie tokenizer differs from its registered byte count or SHA-256"
        )
    return record


def freeze_moondream_tokenizer_lock(
    *,
    cache_root: str | Path,
    lock_path: str | Path,
    downloader: AuxiliaryArtifactDownloader | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Acquire and byte-lock Moondream's exact, separate Starmie tokenizer.

    No mutable branch is resolved here.  The registered 40-character commit is
    passed directly to ``hf_hub_download`` and only the single file consumed by
    ``tokenizers.Tokenizer.from_pretrained`` is admitted to the lock.
    """

    cache = _lexical_absolute(cache_root)
    destination = _lexical_absolute(lock_path)
    cache.mkdir(parents=True, exist_ok=True)
    _reject_symlinked_path_components(cache, label="Moondream tokenizer cache")
    if cache.is_symlink() or cache != cache.resolve() or not cache.is_dir():
        raise StudyGateError("Moondream tokenizer cache root is unsafe")
    destination.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlinked_path_components(destination, label="Moondream tokenizer lock")
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            raise StudyGateError(
                f"Moondream tokenizer lock already exists: {destination}"
            )
        if destination.is_symlink() or not destination.is_file():
            raise StudyGateError("Existing Moondream tokenizer lock is unsafe")

    download = downloader or _default_auxiliary_artifact_downloader
    downloaded = download(
        MOONDREAM_TOKENIZER_REPO,
        MOONDREAM_TOKENIZER_REVISION,
        MOONDREAM_TOKENIZER_FILENAME,
        cache,
    )
    artifact = _verified_auxiliary_artifact_record(
        downloaded,
        cache_root=cache,
    )
    inventory = [{
        "path": MOONDREAM_TOKENIZER_FILENAME,
        "bytes": artifact["bytes"],
        "sha256": artifact["sha256"],
    }]
    payload = {
        "schema_version": MOONDREAM_TOKENIZER_LOCK_SCHEMA,
        "lock_type": MOONDREAM_TOKENIZER_LOCK_TYPE,
        "created_utc": _utc_now(),
        "repo": MOONDREAM_TOKENIZER_REPO,
        "requested_revision": MOONDREAM_TOKENIZER_REVISION,
        "resolved_revision": MOONDREAM_TOKENIZER_REVISION,
        "cache_root": str(cache),
        "file_count": 1,
        "artifacts": [artifact],
        "inventory_sha256": hash_object(inventory),
    }
    atomic_write_json(destination, payload)
    return payload


def verify_moondream_tokenizer_lock(
    lock_path: str | Path,
    *,
    offline_checker: AuxiliaryArtifactChecker | None = None,
) -> dict[str, Any]:
    """Fail closed unless the pinned Starmie tokenizer is locally exact."""

    path = _lexical_absolute(lock_path)
    _reject_symlinked_path_components(path, label="Moondream tokenizer lock")
    if path.is_symlink() or not path.is_file():
        raise StudyGateError("Moondream tokenizer lock is missing or unsafe")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError("Moondream tokenizer lock is not valid JSON") from exc
    expected_keys = {
        "schema_version", "lock_type", "created_utc", "repo",
        "requested_revision", "resolved_revision", "cache_root", "file_count",
        "artifacts", "inventory_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise StudyGateError("Moondream tokenizer lock envelope changed")
    if (
        payload["schema_version"] != MOONDREAM_TOKENIZER_LOCK_SCHEMA
        or payload["lock_type"] != MOONDREAM_TOKENIZER_LOCK_TYPE
        or payload["repo"] != MOONDREAM_TOKENIZER_REPO
        or payload["requested_revision"] != MOONDREAM_TOKENIZER_REVISION
        or payload["resolved_revision"] != MOONDREAM_TOKENIZER_REVISION
        or type(payload["file_count"]) is not int
        or payload["file_count"] != 1
    ):
        raise StudyGateError("Moondream tokenizer lock identity changed")
    try:
        time.strptime(str(payload["created_utc"]), "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as exc:
        raise StudyGateError("Moondream tokenizer lock timestamp is invalid") from exc
    if HEX64.fullmatch(str(payload["inventory_sha256"])) is None:
        raise StudyGateError("Moondream tokenizer inventory hash is invalid")
    cache = _lexical_absolute(str(payload["cache_root"]))
    if cache != cache.resolve() or cache.is_symlink() or not cache.is_dir():
        raise StudyGateError("Moondream tokenizer lock has an unsafe cache root")
    artifacts = payload["artifacts"]
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise StudyGateError("Moondream tokenizer lock must contain one artifact")
    stored = artifacts[0]
    expected_artifact_keys = {
        "filename", "snapshot_path", "snapshot_relative_path",
        "resolved_relative_path", "bytes", "sha256",
    }
    if not isinstance(stored, dict) or set(stored) != expected_artifact_keys:
        raise StudyGateError("Moondream tokenizer artifact record changed")
    current = _verified_auxiliary_artifact_record(
        str(stored["snapshot_path"]),
        cache_root=cache,
    )
    if stored != current:
        raise StudyGateError("Pinned Starmie tokenizer bytes or path changed")
    inventory = [{
        "path": MOONDREAM_TOKENIZER_FILENAME,
        "bytes": current["bytes"],
        "sha256": current["sha256"],
    }]
    if payload["inventory_sha256"] != hash_object(inventory):
        raise StudyGateError("Moondream tokenizer inventory hash changed")

    checker = offline_checker or _default_auxiliary_artifact_checker
    offline_path = _lexical_absolute(checker(
        MOONDREAM_TOKENIZER_REPO,
        MOONDREAM_TOKENIZER_REVISION,
        MOONDREAM_TOKENIZER_FILENAME,
        cache,
    ))
    if offline_path != Path(str(current["snapshot_path"])):
        raise StudyGateError(
            "Local-only Starmie lookup resolved a different tokenizer artifact"
        )
    return {
        **payload,
        "lock_path": str(path),
        "lock_sha256": hash_file(path),
        "offline_verification": "OFFLINE_VERIFIED",
    }


def _relative_to(path: Path, root: Path, *, label: str) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise StudyGateError(f"{label} escapes the configured Hugging Face cache: {path}") from exc


def _snapshot_inventory(snapshot: Path, cache_root: Path) -> list[dict[str, Any]]:
    if not snapshot.is_dir():
        raise StudyGateError(f"Downloaded model snapshot is not a directory: {snapshot}")
    _relative_to(snapshot, cache_root, label="Model snapshot")
    records: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*"), key=lambda value: value.relative_to(snapshot).as_posix()):
        if path.is_symlink():
            _relative_to(path.resolve(), cache_root, label="Model snapshot symlink target")
        if not path.is_file():
            continue
        records.append(
            {
                "path": path.relative_to(snapshot).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
            }
        )
    if not records:
        raise StudyGateError(f"Model snapshot contains no files: {snapshot}")
    return records


def freeze_huggingface_models(
    models: Mapping[str, Any],
    *,
    cache_root: str | Path,
    lock_path: str | Path,
    resolver: RevisionResolver | None = None,
    downloader: SnapshotDownloader | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Resolve, download, inventory, and atomically freeze the registered models.

    ``resolver`` and ``downloader`` are injectable so unit tests never require
    network access.  The production defaults use ``HfApi.model_info`` and
    ``snapshot_download`` respectively.
    """

    entries = _model_entries(models)
    cache = Path(cache_root).expanduser().resolve()
    destination = Path(lock_path).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise StudyGateError(
            f"Model lock already exists: {destination}; use a new preparation run or explicit overwrite"
        )
    resolve = resolver or _default_resolver
    download = downloader or _default_downloader

    locked: list[dict[str, Any]] = []
    for entry in entries:
        requested = entry["requested_revision"]
        resolved = str(resolve(str(entry["repo"]), requested)).strip().lower()
        if HEX40.fullmatch(resolved) is None:
            raise StudyGateError(
                f"Hugging Face did not resolve an immutable 40-character commit for {entry['repo']}"
            )
        if requested is not None and HEX40.fullmatch(requested.lower()) and requested.lower() != resolved:
            raise StudyGateError(
                f"Configured immutable revision for {entry['repo']} resolved to a different commit"
            )
        snapshot = Path(download(str(entry["repo"]), resolved, cache)).expanduser().resolve()
        inventory = _snapshot_inventory(snapshot, cache)
        locked.append(
            {
                **entry,
                "resolved_revision": resolved,
                "snapshot_path": str(snapshot),
                "snapshot_relative_path": _relative_to(
                    snapshot, cache, label="Model snapshot"
                ).as_posix(),
                "file_count": len(inventory),
                "total_bytes": sum(int(item["bytes"]) for item in inventory),
                "inventory_sha256": hash_object(inventory),
                "files": inventory,
            }
        )

    payload = {
        "schema_version": MODEL_LOCK_SCHEMA_VERSION,
        "lock_type": MODEL_LOCK_TYPE,
        "created_utc": _utc_now(),
        "cache_root": str(cache),
        "models": locked,
    }
    payload["model_set_sha256"] = _model_set_fingerprint(locked)
    atomic_write_json(destination, payload)
    return payload


def _read_model_lock(lock_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(lock_path).expanduser().resolve()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"Cannot read model lock: {path}") from exc
    if not isinstance(payload, dict):
        raise StudyGateError("MODEL_LOCK.json must contain one JSON object")
    if payload.get("schema_version") != MODEL_LOCK_SCHEMA_VERSION or payload.get("lock_type") != MODEL_LOCK_TYPE:
        raise StudyGateError("MODEL_LOCK.json has an unsupported schema or lock type")
    return path, payload


def _verify_huggingface_model_lock_for_set(
    lock_path: str | Path,
    *,
    compact_repos: Mapping[str, str],
    model_backends: Mapping[str, str],
    models: Mapping[str, Any] | None = None,
    offline_checker: OfflineSnapshotChecker | None = None,
) -> dict[str, Any]:
    """Fail closed unless every frozen byte is available through local-only loading.

    This is the mandatory validation/test boundary.  The production checker
    invokes ``snapshot_download(..., local_files_only=True)`` and therefore
    cannot fetch missing model bytes from the network.
    """

    _, payload = _read_model_lock(lock_path)
    cache = Path(str(payload.get("cache_root", ""))).expanduser().resolve()
    if not cache.is_dir():
        raise StudyGateError(f"Frozen Hugging Face cache is missing: {cache}")
    raw_models = payload.get("models")
    if not isinstance(raw_models, list):
        raise StudyGateError("MODEL_LOCK.json models must be a list")

    if models is not None:
        if compact_repos != OFFICIAL_COMPACT_MODEL_REPOS:
            raise StudyGateError(
                "Configured-model validation is unavailable for a historical lock"
            )
        configured = _model_entries(models)
        expected = {
            (item["model_id"], item["repo"], item["role"])
            for item in configured
        }
        requested_by_id = {
            str(item["model_id"]): item["requested_revision"] for item in configured
        }
    else:
        expected = {
            (model_id, repo, "compact")
            for model_id, repo in compact_repos.items()
        }
        expected.add((OFFICIAL_UPPER_BOUND_ID, OFFICIAL_UPPER_BOUND_REPO, "upper_bound"))
        requested_by_id = {}
    observed = {
        (str(item.get("model_id", "")), str(item.get("repo", "")), str(item.get("role", "")))
        for item in raw_models
        if isinstance(item, Mapping)
    }
    expected_order = (*compact_repos, OFFICIAL_UPPER_BOUND_ID)
    observed_order = tuple(
        str(item.get("model_id", "")) if isinstance(item, Mapping) else ""
        for item in raw_models
    )
    if (
        observed != expected
        or len(raw_models) != len(expected)
        or observed_order != expected_order
    ):
        raise StudyGateError("MODEL_LOCK.json does not contain the exact registered model set")

    checker = offline_checker or _default_offline_checker
    verified: list[dict[str, Any]] = []
    for item in raw_models:
        if not isinstance(item, Mapping):
            raise StudyGateError("MODEL_LOCK.json contains a non-object model entry")
        revision = str(item.get("resolved_revision", "")).lower()
        if HEX40.fullmatch(revision) is None:
            raise StudyGateError(f"Frozen revision is invalid for {item.get('model_id')}")
        model_id = str(item.get("model_id", ""))
        expected_backend = (
            "qwen3_vl" if model_id == OFFICIAL_UPPER_BOUND_ID
            else model_backends.get(model_id)
        )
        if str(item.get("backend", "")) != expected_backend:
            raise StudyGateError(f"Frozen backend is invalid for {model_id}")
        configured_revision = requested_by_id.get(str(item.get("model_id", "")))
        if (
            configured_revision is not None
            and HEX40.fullmatch(str(configured_revision).lower())
            and str(configured_revision).lower() != revision
        ):
            raise StudyGateError(
                f"Configured revision differs from MODEL_LOCK.json for {item.get('model_id')}"
            )
        snapshot = Path(str(item.get("snapshot_path", ""))).expanduser().resolve()
        _relative_to(snapshot, cache, label="Frozen model snapshot")
        inventory = _snapshot_inventory(snapshot, cache)
        expected_inventory = item.get("files")
        if inventory != expected_inventory:
            raise StudyGateError(f"Frozen model bytes changed for {item.get('model_id')}")
        if hash_object(inventory) != str(item.get("inventory_sha256", "")):
            raise StudyGateError(f"Frozen model inventory hash changed for {item.get('model_id')}")
        if len(inventory) != int(item.get("file_count", -1)):
            raise StudyGateError(f"Frozen model file count changed for {item.get('model_id')}")
        if sum(int(record["bytes"]) for record in inventory) != int(item.get("total_bytes", -1)):
            raise StudyGateError(f"Frozen model byte count changed for {item.get('model_id')}")
        local_snapshot = Path(checker(str(item["repo"]), revision, cache)).expanduser().resolve()
        if local_snapshot != snapshot:
            raise StudyGateError(
                f"Local-only Hugging Face lookup resolved a different snapshot for {item.get('model_id')}"
            )
        verified.append(
            {
                "model_id": str(item["model_id"]),
                "repo": str(item["repo"]),
                "revision": revision,
                "status": "OFFLINE_VERIFIED",
            }
        )

    expected_set_hash = _model_set_fingerprint(raw_models)
    if expected_set_hash != str(payload.get("model_set_sha256", "")):
        raise StudyGateError("MODEL_LOCK.json model-set hash is inconsistent")
    return {**payload, "offline_verification": verified}


def verify_huggingface_model_lock(
    lock_path: str | Path,
    *,
    models: Mapping[str, Any] | None = None,
    offline_checker: OfflineSnapshotChecker | None = None,
) -> dict[str, Any]:
    """Verify the exact active v2r13 two-compact-plus-upper-bound lock."""

    return _verify_huggingface_model_lock_for_set(
        lock_path,
        compact_repos=OFFICIAL_COMPACT_MODEL_REPOS,
        model_backends=OFFICIAL_MODEL_BACKENDS,
        models=models,
        offline_checker=offline_checker,
    )


def verify_historical_v2r5_huggingface_model_lock(
    lock_path: str | Path,
    *,
    offline_checker: OfflineSnapshotChecker | None = None,
) -> dict[str, Any]:
    """Verify the quarantined historical v2r5 four-model source lock only."""

    return _verify_huggingface_model_lock_for_set(
        lock_path,
        compact_repos=HISTORICAL_V2R5_COMPACT_MODEL_REPOS,
        model_backends=HISTORICAL_V2R5_MODEL_BACKENDS,
        offline_checker=offline_checker,
    )


def _create_or_exact_reverify_json(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    label: str,
    allow_create: bool = True,
) -> str:
    """Create one canonical JSON file without replacement, or reverify bytes."""

    destination = _lexical_absolute(path)
    _reject_symlinked_path_components(destination, label=label)
    if allow_create:
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Recheck after creation so an existing symlink can never be followed
        # as a side effect of preparing the destination hierarchy.
        _reject_symlinked_path_components(destination, label=label)
    expected = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )

    def reverify_existing() -> str:
        if destination.is_symlink() or not destination.is_file():
            raise StudyGateError(f"Existing {label} is unsafe or not a regular file")
        try:
            observed = destination.read_bytes()
        except OSError as exc:
            raise StudyGateError(f"Existing {label} cannot be read") from exc
        if observed != expected:
            raise StudyGateError(f"Existing {label} differs from the exact derivation")
        return "REVERIFIED_EXISTING"

    if destination.exists() or destination.is_symlink():
        return reverify_existing()
    if not allow_create:
        raise StudyGateError(f"{label} is absent and read-only reverify cannot create it")

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent
    )
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(expected)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A same-directory hard link publishes the complete temporary file
            # atomically and fails instead of replacing a concurrent artifact.
            os.link(temporary_name, destination)
        except FileExistsError:
            return reverify_existing()
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    return "CREATED"


def _existing_lock_timestamp(path: Path, *, label: str) -> str | None:
    """Return a canonical existing artifact timestamp without accepting drift."""

    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink() or not path.is_file():
        raise StudyGateError(f"Existing {label} is unsafe or not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        created_utc = str(payload["created_utc"])
        time.strptime(created_utc, "%Y-%m-%dT%H:%M:%SZ")
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
        raise StudyGateError(f"Existing {label} has an invalid timestamp") from exc
    return created_utc


def derive_v2r6_three_model_lock(
    source_lock_path: str | Path,
    *,
    destination: str | Path,
    derivation_destination: str | Path,
    expected_source_sha256: str,
    amendment_id: str,
    exclusion_reason: str,
    offline_checker: OfflineSnapshotChecker | None = None,
    allow_create: bool = True,
) -> dict[str, Any]:
    """Derive and seal the prospective v2r6 lock from authenticated v2r5 bytes.

    This is a model-set amendment, not a new download.  It verifies the source
    under the exact historical four-model contract, removes only Moondream,
    preserves every remaining model-row field and row order, and creates both
    a new active lock and a separate derivation receipt.  Existing targets are
    accepted only when their canonical bytes exactly match the derivation.
    ``allow_create=False`` is the strict validation/test reverify path: neither
    target may be recreated if it is absent at the instant of publication.
    """

    source = _lexical_absolute(source_lock_path)
    active_destination = _lexical_absolute(destination)
    receipt_destination = _lexical_absolute(derivation_destination)
    if len({source, active_destination, receipt_destination}) != 3:
        raise StudyGateError(
            "Source, active model lock, and derivation receipt must be distinct"
        )
    _reject_symlinked_path_components(source, label="Historical v2r5 model lock")
    if source.is_symlink() or not source.is_file():
        raise StudyGateError(
            "Historical v2r5 model lock is missing, unsafe, or not a regular file"
        )
    expected_source = str(expected_source_sha256).strip().lower()
    if HEX64.fullmatch(expected_source) is None:
        raise StudyGateError("Historical model-lock anchor must be one SHA-256")
    observed_source = hash_file(source)
    if observed_source != expected_source:
        raise StudyGateError(
            "Historical v2r5 model lock differs from its external SHA-256 anchor"
        )
    amendment = str(amendment_id).strip()
    reason = str(exclusion_reason).strip()
    if not amendment or not reason:
        raise StudyGateError(
            "The v2r6 model-set amendment ID and exclusion reason are required"
        )

    verified_source = verify_historical_v2r5_huggingface_model_lock(
        source,
        offline_checker=offline_checker,
    )
    _, source_payload = _read_model_lock(source)
    source_rows = source_payload.get("models")
    if not isinstance(source_rows, list):
        raise StudyGateError("Historical model lock has no model rows")
    active_rows = [
        dict(row)
        for row in source_rows
        if isinstance(row, Mapping)
        and str(row.get("model_id", "")) != HISTORICAL_MOONDREAM_MODEL_ID
    ]
    if tuple(str(row.get("model_id", "")) for row in active_rows) != (
        OFFICIAL_RUNTIME_MODEL_IDS
    ):
        raise StudyGateError(
            "Historical lock cannot yield the exact ordered v2r6 model set"
        )
    if len(source_rows) - len(active_rows) != 1:
        raise StudyGateError("v2r6 derivation must exclude exactly one model row")

    active_created_utc = _existing_lock_timestamp(
        active_destination, label="v2r6 active model lock"
    ) or _utc_now()
    derivation_metadata = {
        "schema_version": V2R6_MODEL_LOCK_DERIVATION_SCHEMA,
        "source_model_lock_path": str(source),
        "source_model_lock_sha256": observed_source,
        "source_model_set_sha256": str(source_payload["model_set_sha256"]),
        "source_model_ids": list(HISTORICAL_V2R5_RUNTIME_MODEL_IDS),
        "active_model_ids": list(OFFICIAL_RUNTIME_MODEL_IDS),
        "excluded_model_ids": [HISTORICAL_MOONDREAM_MODEL_ID],
        "amendment_id": amendment,
        "exclusion_reason": reason,
        "derivation_operation": "ordered_filter_excluding_only_moondream2",
        "prior_runtime_smoke_rows_reused": False,
        "validation_or_test_evidence_used": False,
    }
    active_payload = {
        "schema_version": MODEL_LOCK_SCHEMA_VERSION,
        "lock_type": MODEL_LOCK_TYPE,
        "created_utc": active_created_utc,
        "cache_root": str(source_payload["cache_root"]),
        "models": active_rows,
        "derivation": derivation_metadata,
    }
    active_payload["model_set_sha256"] = _model_set_fingerprint(active_rows)
    lock_mode = _create_or_exact_reverify_json(
        active_destination,
        active_payload,
        label="v2r6 active model lock",
        allow_create=allow_create,
    )
    verified_active = verify_huggingface_model_lock(
        active_destination,
        offline_checker=offline_checker,
    )
    active_lock_sha256 = hash_file(active_destination)

    receipt_created_utc = _existing_lock_timestamp(
        receipt_destination, label="v2r6 model-lock derivation receipt"
    ) or active_created_utc
    derivation_receipt = {
        **derivation_metadata,
        "created_utc": receipt_created_utc,
        "receipt_type": "prospective_pre_validation_model_set_amendment",
        "derived_model_lock_path": str(active_destination),
        "derived_model_lock_sha256": active_lock_sha256,
        "derived_model_set_sha256": str(active_payload["model_set_sha256"]),
    }
    receipt_mode = _create_or_exact_reverify_json(
        receipt_destination,
        derivation_receipt,
        label="v2r6 model-lock derivation receipt",
        allow_create=allow_create,
    )
    return {
        **verified_active,
        "lock_path": str(active_destination),
        "lock_sha256": active_lock_sha256,
        "derivation_receipt_path": str(receipt_destination),
        "derivation_receipt_sha256": hash_file(receipt_destination),
        "derivation_mode": {
            "model_lock": lock_mode,
            "receipt": receipt_mode,
        },
        "historical_source_offline_verification": verified_source[
            "offline_verification"
        ],
    }


def model_loader_environment(lock_path: str | Path) -> dict[str, str]:
    """Return the environment that makes later Hugging Face loads fail offline.

    Apply these values before importing Transformers in each validation/test
    process.  Every Hugging Face/Transformers cache alias points at the exact
    cache layout recorded by the lock, while the three offline flags prevent
    mutable remote fallback.  Keeping the aliases identical is required for
    historical locks whose snapshots live directly below ``HF_HOME`` rather
    than below a newer ``hub`` subdirectory.
    """

    _, payload = _read_model_lock(lock_path)
    cache = Path(str(payload.get("cache_root", ""))).expanduser().resolve()
    if not cache.is_dir():
        raise StudyGateError(f"Frozen Hugging Face cache is missing: {cache}")
    revisions = {
        str(item["model_id"]): str(item["resolved_revision"])
        for item in payload.get("models", [])
        if isinstance(item, Mapping)
    }
    required = set(OFFICIAL_COMPACT_MODEL_REPOS) | {OFFICIAL_UPPER_BOUND_ID}
    if set(revisions) != required:
        raise StudyGateError("MODEL_LOCK.json does not contain the exact registered model set")
    return {
        "HF_HUB_CACHE": str(cache),
        "HUGGINGFACE_HUB_CACHE": str(cache),
        "TRANSFORMERS_CACHE": str(cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "SEMUAV_MODELS_LOCAL_ONLY": "1",
        "SEMUAV_QWEN_REVISION": revisions["qwen3_vl_2b"],
        "SEMUAV_SMOLVLM_REVISION": revisions["smolvlm2_2b"],
        "SEMUAV_UPPER_BOUND_REVISION": revisions[OFFICIAL_UPPER_BOUND_ID],
    }


def _component_root(visdrone_root: Path, directory_name: str) -> Path:
    if directory_name in VISDRONE_DET_TEST_COMPONENTS or "test" in directory_name.casefold():
        raise StudyGateError("Detector preparation may not access VisDrone test-dev/challenge")
    direct = visdrone_root / directory_name
    if direct.is_dir():
        return direct.resolve()
    matches = [
        child / directory_name
        for child in visdrone_root.iterdir()
        if child.is_dir() and (child / directory_name).is_dir()
    ] if visdrone_root.is_dir() else []
    if len(matches) != 1:
        raise StudyGateError(f"Missing or ambiguous official VisDrone component: {directory_name}")
    return matches[0].resolve()


def _parse_visdrone_det_annotation(
    annotation: Path,
    *,
    image_width: int,
    image_height: int,
) -> tuple[list[str], int, int]:
    """Return YOLO rows plus ignored/clipped publisher-row counts."""

    output: list[str] = []
    ignored = 0
    clipped_count = 0
    for line_number, raw in enumerate(annotation.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.rstrip(",").split(",")]
        if len(fields) != 8 or any(value == "" for value in fields):
            raise StudyGateError(f"Invalid 8-field VisDrone DET row: {annotation}:{line_number}")
        try:
            left, top, width, height, score = map(float, fields[:5])
            category_id, truncation, occlusion = map(int, fields[5:])
        except ValueError as exc:
            raise StudyGateError(f"Nonnumeric VisDrone DET row: {annotation}:{line_number}") from exc
        if not all(math.isfinite(value) for value in (left, top, width, height, score)):
            raise StudyGateError(f"Nonfinite VisDrone DET row: {annotation}:{line_number}")
        if category_id < 0 or category_id > len(VISDRONE_YOLO_NAMES) + 1:
            raise StudyGateError(
                f"Unsupported VisDrone category {category_id}: {annotation}:{line_number}"
            )
        if width < 0 or height < 0 or not 0.0 <= score <= 1.0:
            raise StudyGateError(f"Invalid VisDrone box/score: {annotation}:{line_number}")
        if truncation not in {0, 1, 2} or occlusion not in {0, 1, 2}:
            raise StudyGateError(f"Invalid VisDrone visibility code: {annotation}:{line_number}")
        if score == 0.0 or category_id in VISDRONE_IGNORED_CATEGORY_IDS:
            ignored += 1
            continue
        if width == 0 or height == 0:
            raise StudyGateError(f"Invalid VisDrone box/score: {annotation}:{line_number}")
        x1, y1, x2, y2 = left, top, left + width, top + height
        clipped = (
            max(0.0, min(float(image_width), x1)),
            max(0.0, min(float(image_height), y1)),
            max(0.0, min(float(image_width), x2)),
            max(0.0, min(float(image_height), y2)),
        )
        if clipped != (x1, y1, x2, y2):
            clipped_count += 1
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise StudyGateError(f"VisDrone box has no visible pixels: {annotation}:{line_number}")
        center_x = ((clipped[0] + clipped[2]) / 2.0) / image_width
        center_y = ((clipped[1] + clipped[3]) / 2.0) / image_height
        normalized_width = (clipped[2] - clipped[0]) / image_width
        normalized_height = (clipped[3] - clipped[1]) / image_height
        output.append(
            f"{category_id - 1} {center_x:.9f} {center_y:.9f} "
            f"{normalized_width:.9f} {normalized_height:.9f}"
        )
    return output, ignored, clipped_count


def _detector_natural_group(image_stem: str) -> str:
    """Return the publisher filename's capture/sequence prefix.

    Official VisDrone DET filenames place the natural capture identity before
    the first underscore.  Keeping every such image together prevents near-
    duplicate frames from crossing the detector fit/early-stopping boundary.
    """

    stem = str(image_stem).strip()
    if not stem:
        raise StudyGateError("VisDrone DET image stem is empty")
    group = stem.split("_", 1)[0].strip()
    if not group:
        raise StudyGateError(f"Cannot derive a natural group from VisDrone image stem {stem!r}")
    return group


def _detector_group_assignments(
    groups: Iterable[str],
    *,
    seed: int,
    internal_validation_fraction: float,
) -> tuple[dict[str, str], dict[str, str]]:
    """Hash-rank natural groups into detector fit and internal validation.

    Rank-based allocation, rather than an independent threshold per group,
    guarantees that both partitions are populated whenever at least two
    natural groups exist.  The hash never consumes pixels or annotations.
    """

    fraction = float(internal_validation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("internal_validation_fraction must lie strictly within (0, 1)")
    normalized = sorted({str(group).strip() for group in groups})
    if any(not group for group in normalized):
        raise StudyGateError("Detector natural groups must be populated")
    if len(normalized) < 2:
        raise StudyGateError(
            "Detector preparation requires at least two publisher-train natural groups "
            "so fit and internal-validation data remain disjoint"
        )
    digests = {
        group: hash_object(
            {
                "version": DETECTOR_SPLIT_VERSION,
                "seed": int(seed),
                "natural_group": group,
            }
        )
        for group in normalized
    }
    ordered = sorted(normalized, key=lambda group: (digests[group], group))
    validation_groups = max(
        1,
        min(len(ordered) - 1, int(round(len(ordered) * fraction))),
    )
    internal = set(ordered[:validation_groups])
    assignments = {
        group: (
            DETECTOR_INTERNAL_VALIDATION_PARTITION
            if group in internal
            else DETECTOR_TRAIN_PARTITION
        )
        for group in normalized
    }
    return assignments, digests


_DETECTOR_STUDY_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "source_uid",
    "component",
    "task",
    "publisher_partition",
    "study_partition",
    "split",
    "split_unit",
    "media_path",
    "annotation_path",
)

_VISDRONE_COMPONENT_CONTRACT: Mapping[str, tuple[str, str, str, frozenset[str]]] = {
    "det_train": ("DET", "train", "development_train", frozenset({"train", "calibration"})),
    "det_val": ("DET", "val", "development_validation", frozenset({"validation"})),
    "det_test_dev": ("DET", "test-dev", "untouched_test", frozenset({"test"})),
    "mot_train": ("MOT", "train", "development_train", frozenset({"train", "calibration"})),
    "mot_val": ("MOT", "val", "development_validation", frozenset({"validation"})),
    "mot_test_dev": ("MOT", "test-dev", "untouched_test", frozenset({"test"})),
}


def _source_id_inventory(values: Iterable[str]) -> dict[str, Any]:
    normalized = sorted({str(value).strip() for value in values})
    if any(not value for value in normalized):
        raise StudyGateError("Detector source-ID inventories may not contain blank values")
    return {"count": len(normalized), "sha256": hash_object(normalized)}


def _read_detector_study_manifest(
    manifest: str | Path | pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load and normalize the explicit study-split source contract.

    A path is preferred for real preparation because its original bytes can be
    bound into the conversion receipt.  A DataFrame is accepted as an explicit
    in-memory allow/deny manifest, but an immutable normalized copy is still
    written into the prepared detector tree for later verification.
    """

    if isinstance(manifest, pd.DataFrame):
        frame = manifest.copy()
        origin = {"kind": "explicit_dataframe", "path": "", "sha256": ""}
    else:
        path = Path(manifest).expanduser().resolve()
        if not path.is_file():
            raise StudyGateError(f"Frozen detector study manifest is missing: {path}")
        suffixes = "".join(path.suffixes).lower()
        try:
            if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
                frame = pd.read_csv(path)
            elif suffixes.endswith(".jsonl") or suffixes.endswith(".ndjson"):
                frame = pd.read_json(path, lines=True)
            else:
                raise StudyGateError(
                    "Frozen detector study manifest must be CSV, CSV.GZ, JSONL, or NDJSON"
                )
        except (OSError, UnicodeError, ValueError) as exc:
            raise StudyGateError(f"Cannot read frozen detector study manifest: {path}") from exc
        origin = {"kind": "frozen_file", "path": str(path), "sha256": hash_file(path)}

    missing = sorted(set(_DETECTOR_STUDY_COLUMNS).difference(frame.columns))
    if missing:
        raise StudyGateError(
            f"Detector study manifest is missing required columns: {missing}"
        )
    normalized = frame.loc[:, list(_DETECTOR_STUDY_COLUMNS)].copy()
    for column in _DETECTOR_STUDY_COLUMNS:
        normalized[column] = normalized[column].fillna("").astype(str).str.strip()
    if normalized.empty:
        raise StudyGateError("Detector study manifest is empty")
    blank_columns = [
        column for column in _DETECTOR_STUDY_COLUMNS
        if normalized[column].eq("").any()
    ]
    if blank_columns:
        raise StudyGateError(
            f"Detector study manifest has blank required values: {blank_columns}"
        )
    for column in ("media_path", "annotation_path"):
        normalized[column] = normalized[column].map(
            lambda value: str(Path(value).expanduser().resolve())
        )
    if normalized["sample_id"].duplicated().any():
        raise StudyGateError("Detector study manifest has duplicate source sample IDs")
    if normalized["source_uid"].duplicated().any():
        raise StudyGateError("Detector study manifest has duplicate source_uid values")

    unknown_components = sorted(
        set(normalized["component"]) - set(_VISDRONE_COMPONENT_CONTRACT)
    )
    if unknown_components:
        raise StudyGateError(
            f"Detector study manifest contains unknown VisDrone components: {unknown_components}"
        )
    for row in normalized.itertuples(index=False):
        expected_task, expected_publisher, expected_study, allowed_splits = (
            _VISDRONE_COMPONENT_CONTRACT[str(row.component)]
        )
        if (
            str(row.task) != expected_task
            or str(row.publisher_partition) != expected_publisher
            or str(row.study_partition) != expected_study
            or str(row.split) not in allowed_splits
        ):
            raise StudyGateError(
                "Detector study manifest component/partition metadata is inconsistent: "
                f"{row.sample_id}"
            )

    group_splits = normalized.groupby("split_unit", sort=False)["split"].nunique()
    if (group_splits != 1).any():
        raise StudyGateError(
            "Detector study manifest has a natural group crossing study splits"
        )
    resolved_media = normalized["media_path"].map(
        lambda value: str(Path(value).expanduser().resolve())
    )
    media_splits = pd.DataFrame(
        {"media_path": resolved_media, "split": normalized["split"]}
    ).groupby("media_path", sort=False)["split"].nunique()
    if (media_splits != 1).any():
        raise StudyGateError(
            "Detector study manifest aliases the same media across study splits"
        )

    detector_train = normalized.loc[
        normalized["component"].eq("det_train")
        & normalized["task"].eq("DET")
        & normalized["split"].eq(DETECTOR_ALLOWED_STUDY_SPLIT)
    ].copy()
    detector_calibration = normalized.loc[
        normalized["component"].eq("det_train")
        & normalized["task"].eq("DET")
        & normalized["split"].eq("calibration")
    ]
    if detector_train.empty:
        raise StudyGateError(
            "Detector study manifest contains no DET rows with study split=='train'"
        )
    if detector_calibration.empty:
        raise StudyGateError(
            "Detector study manifest must include DET calibration rows for a zero-overlap audit"
        )
    return (
        normalized.sort_values("sample_id", kind="stable", ignore_index=True),
        origin,
    )


def _detector_leakage_audit(
    study_manifest: pd.DataFrame,
    conversion_source_ids: Iterable[str],
) -> dict[str, Any]:
    converted = {str(value).strip() for value in conversion_source_ids}
    allowed_rows = study_manifest.loc[
        study_manifest["component"].eq("det_train")
        & study_manifest["task"].eq("DET")
        & study_manifest["split"].eq(DETECTOR_ALLOWED_STUDY_SPLIT)
    ]
    allowed = set(allowed_rows["sample_id"].astype(str))
    split_ids = {
        split: set(
            study_manifest.loc[
                study_manifest["split"].eq(split), "sample_id"
            ].astype(str)
        )
        for split in sorted(DETECTOR_FORBIDDEN_STUDY_SPLITS)
    }
    split_units = {
        split: set(
            study_manifest.loc[
                study_manifest["split"].eq(split), "split_unit"
            ].astype(str)
        )
        for split in sorted(DETECTOR_FORBIDDEN_STUDY_SPLITS)
    }
    converted_units = set(
        allowed_rows.loc[
            allowed_rows["sample_id"].astype(str).isin(converted), "split_unit"
        ].astype(str)
    )
    overlap = {
        split: {
            **_source_id_inventory(converted.intersection(split_ids[split])),
            "passed": not converted.intersection(split_ids[split]),
        }
        for split in sorted(DETECTOR_FORBIDDEN_STUDY_SPLITS)
    }
    group_overlap = {
        split: {
            **_source_id_inventory(converted_units.intersection(split_units[split])),
            "passed": not converted_units.intersection(split_units[split]),
        }
        for split in sorted(DETECTOR_FORBIDDEN_STUDY_SPLITS)
    }
    exact_allowed_set = converted == allowed
    return {
        "schema": DETECTOR_SOURCE_AUDIT_SCHEMA,
        "source_id_column": "sample_id",
        "allowed_study_split": DETECTOR_ALLOWED_STUDY_SPLIT,
        "allowed_source_ids": _source_id_inventory(allowed),
        "conversion_source_ids": _source_id_inventory(converted),
        "study_split_source_ids": {
            split: _source_id_inventory(split_ids[split])
            for split in sorted(DETECTOR_FORBIDDEN_STUDY_SPLITS)
        },
        "source_id_overlap": overlap,
        "split_unit_overlap": group_overlap,
        "conversion_equals_allowed_source_set": bool(exact_allowed_set),
        "zero_forbidden_source_overlap": bool(
            all(item["passed"] for item in overlap.values())
            and all(item["passed"] for item in group_overlap.values())
        ),
        "passed": bool(
            exact_allowed_set
            and all(item["passed"] for item in overlap.values())
            and all(item["passed"] for item in group_overlap.values())
        ),
    }


def prepare_ultralytics_visdrone_det(
    visdrone_root: str | Path,
    output_root: str | Path,
    *,
    study_source_manifest: str | Path | pd.DataFrame,
    hash_images: bool = True,
    seed: int = 20260823,
    internal_validation_fraction: float = 0.20,
) -> dict[str, Any]:
    """Build a leakage-safe Ultralytics detector-training tree.

    Image files are represented by absolute symlinks and therefore are not
    duplicated.  Labels are converted from the official eight-field format.
    ``study_source_manifest`` is mandatory and is the only source allowlist.
    Only its publisher DET-train rows already frozen to study ``split=='train'``
    are converted.  Those allowed study groups are seed/hash-ranked into
    ``detector_train`` and ``internal_val``.  Study calibration, validation,
    and test IDs are retained only as hashed deny-set audit inputs and can
    never be selected through this API.
    """

    source_root = Path(visdrone_root).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    study_manifest, manifest_origin = _read_detector_study_manifest(
        study_source_manifest
    )
    if destination.exists():
        raise StudyGateError(
            f"Ultralytics preparation output already exists: {destination}; use a new frozen run"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    records: list[dict[str, Any]] = []
    try:
        directory_name = VISDRONE_DET_TRAIN_COMPONENT
        component = _component_root(source_root, directory_name)
        image_source = component / "images"
        annotation_source = component / "annotations"
        if not image_source.is_dir() or not annotation_source.is_dir():
            raise StudyGateError(f"Incomplete official VisDrone DET component: {component}")
        image_source = image_source.resolve()
        annotation_source = annotation_source.resolve()
        allowed = study_manifest.loc[
            study_manifest["component"].eq("det_train")
            & study_manifest["task"].eq("DET")
            & study_manifest["split"].eq(DETECTOR_ALLOWED_STUDY_SPLIT)
        ].sort_values("sample_id", kind="stable")
        source_rows: list[tuple[Any, Path, Path, str]] = []
        canonical_ids: set[str] = set()
        canonical_uids: set[str] = set()
        for row in allowed.itertuples(index=False):
            image_path = Path(str(row.media_path)).expanduser().resolve()
            annotation_path = Path(str(row.annotation_path)).expanduser().resolve()
            if (
                image_path.parent != image_source
                or annotation_path.parent != annotation_source
                or image_path.suffix.lower() not in IMAGE_SUFFIXES
                or annotation_path.suffix.lower() != ".txt"
                or annotation_path.stem != image_path.stem
            ):
                raise StudyGateError(
                    "An allowed detector row is not an exact publisher DET-train image/annotation pair: "
                    f"{row.sample_id}"
                )
            if not image_path.is_file() or not annotation_path.is_file():
                raise StudyGateError(
                    f"An allowed detector source is missing: {row.sample_id}"
                )
            expected_id = f"visdrone:det_train:{image_path.stem}"
            expected_uid = f"det_train/images/{image_path.name}"
            if str(row.sample_id) != expected_id or str(row.source_uid) != expected_uid:
                raise StudyGateError(
                    "An allowed detector source ID is not canonically bound to its publisher path: "
                    f"{row.sample_id}"
                )
            if expected_id in canonical_ids or expected_uid in canonical_uids:
                raise StudyGateError("Allowed detector source IDs are not unique")
            canonical_ids.add(expected_id)
            canonical_uids.add(expected_uid)
            source_rows.append(
                (row, image_path, annotation_path, _detector_natural_group(image_path.stem))
            )

        natural_to_study_unit: dict[str, str] = {}
        study_unit_to_natural: dict[str, str] = {}
        for row, _, _, natural_group in source_rows:
            study_unit = str(row.split_unit)
            previous_study = natural_to_study_unit.setdefault(natural_group, study_unit)
            previous_natural = study_unit_to_natural.setdefault(study_unit, natural_group)
            if previous_study != study_unit or previous_natural != natural_group:
                raise StudyGateError(
                    "Frozen study split units disagree with VisDrone DET natural groups"
                )
        group_assignments, group_digests = _detector_group_assignments(
            study_unit_to_natural,
            seed=int(seed),
            internal_validation_fraction=float(internal_validation_fraction),
        )

        for row, image_path, annotation_path, natural_group in source_rows:
            study_unit = str(row.split_unit)
            partition = group_assignments[study_unit]
            image_output = staging / "images" / partition
            label_output = staging / "labels" / partition
            image_output.mkdir(parents=True, exist_ok=True)
            label_output.mkdir(parents=True, exist_ok=True)
            with Image.open(image_path) as image:
                image_width, image_height = image.size
            if image_width <= 0 or image_height <= 0:
                raise StudyGateError(f"Image has invalid dimensions: {image_path}")
            yolo_rows, ignored, clipped = _parse_visdrone_det_annotation(
                annotation_path,
                image_width=image_width,
                image_height=image_height,
            )
            generated_image = image_output / image_path.name
            generated_image.symlink_to(image_path)
            generated_label = label_output / f"{image_path.stem}.txt"
            generated_label.write_text(
                "".join(f"{row}\n" for row in yolo_rows), encoding="utf-8"
            )
            records.append(
                {
                    "partition": partition,
                    "publisher_component": directory_name,
                    "study_source_id": str(row.sample_id),
                    "study_source_uid": str(row.source_uid),
                    "study_split": str(row.split),
                    "study_split_unit": study_unit,
                    "natural_group": natural_group,
                    "natural_group_split_sha256": group_digests[study_unit],
                    "split_assignment_version": DETECTOR_SPLIT_VERSION,
                    "split_assignment_seed": int(seed),
                    "internal_validation_fraction": float(internal_validation_fraction),
                    "source_image": str(image_path),
                    "source_image_bytes": image_path.stat().st_size,
                    "source_image_sha256": hash_file(image_path) if hash_images else None,
                    "source_annotation": str(annotation_path.resolve()),
                    "source_annotation_sha256": hash_file(annotation_path),
                    "image_width": image_width,
                    "image_height": image_height,
                    "publisher_rows_retained": len(yolo_rows),
                    "publisher_rows_ignored": ignored,
                    "publisher_rows_clipped_to_image": clipped,
                    "generated_image": (
                        Path("images") / partition / image_path.name
                    ).as_posix(),
                    "generated_label": (
                        Path("labels") / partition / generated_label.name
                    ).as_posix(),
                    "generated_label_sha256": hash_file(generated_label),
                }
            )

        frozen_study_manifest = staging / DETECTOR_STUDY_MANIFEST
        frozen_study_manifest.write_text(
            "".join(
                json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
                for record in study_manifest.to_dict(orient="records")
            ),
            encoding="utf-8",
        )
        leakage_audit = _detector_leakage_audit(
            study_manifest,
            (record["study_source_id"] for record in records),
        )
        leakage_audit.update(
            {
                "input_manifest": manifest_origin,
                "frozen_study_manifest": DETECTOR_STUDY_MANIFEST,
                "frozen_study_manifest_sha256": hash_file(frozen_study_manifest),
            }
        )
        if not leakage_audit["passed"]:
            raise StudyGateError(
                "Detector source audit found calibration/validation/test leakage"
            )
        leakage_audit_path = staging / DETECTOR_LEAKAGE_AUDIT
        leakage_audit_path.write_text(
            json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        manifest_text = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
            for record in records
        )
        (staging / "CONVERSION_MANIFEST.jsonl").write_text(manifest_text, encoding="utf-8")
        dataset_yaml = {
            "path": str(destination),
            "train": f"images/{DETECTOR_TRAIN_PARTITION}",
            "val": f"images/{DETECTOR_INTERNAL_VALIDATION_PARTITION}",
            "names": {index: name for index, name in enumerate(VISDRONE_YOLO_NAMES)},
        }
        (staging / "visdrone_det.yaml").write_text(
            yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8"
        )
        summary = {
            "schema_version": 1,
            "conversion": "official_visdrone_det_train_to_ultralytics_yolo_group_split",
            "source_partitions": [VISDRONE_DET_TRAIN_COMPONENT],
            "excluded_partitions": sorted(VISDRONE_DET_EXCLUDED_COMPONENTS),
            "publisher_validation_used": False,
            "publisher_test_used": False,
            "split_assignment_version": DETECTOR_SPLIT_VERSION,
            "split_assignment_seed": int(seed),
            "internal_validation_fraction": float(internal_validation_fraction),
            "category_id_mapping": {
                str(source_id): {
                    "ultralytics_id": source_id - 1,
                    "name": VISDRONE_YOLO_NAMES[source_id - 1],
                }
                for source_id in range(1, 11)
            },
            "images": len(records),
            "natural_groups": len(group_assignments),
            "detector_train_groups": sum(
                partition == DETECTOR_TRAIN_PARTITION
                for partition in group_assignments.values()
            ),
            "internal_validation_groups": sum(
                partition == DETECTOR_INTERNAL_VALIDATION_PARTITION
                for partition in group_assignments.values()
            ),
            "detector_train_images": sum(
                record["partition"] == DETECTOR_TRAIN_PARTITION for record in records
            ),
            "internal_validation_images": sum(
                record["partition"] == DETECTOR_INTERNAL_VALIDATION_PARTITION
                for record in records
            ),
            "manifest_sha256": hash_file(staging / "CONVERSION_MANIFEST.jsonl"),
            "image_hashes_included": bool(hash_images),
            "study_source_manifest": DETECTOR_STUDY_MANIFEST,
            "study_source_manifest_sha256": hash_file(frozen_study_manifest),
            "detector_leakage_audit": DETECTOR_LEAKAGE_AUDIT,
            "detector_leakage_audit_sha256": hash_file(leakage_audit_path),
            "allowed_source_id_count": leakage_audit["allowed_source_ids"]["count"],
            "allowed_source_ids_sha256": leakage_audit["allowed_source_ids"]["sha256"],
            "zero_forbidden_source_overlap": True,
        }
        (staging / "CONVERSION_SUMMARY.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, destination)
        return {
            **summary,
            "output_root": str(destination),
            "dataset_yaml": str(destination / "visdrone_det.yaml"),
            "conversion_manifest": str(destination / "CONVERSION_MANIFEST.jsonl"),
            "study_source_manifest": str(destination / DETECTOR_STUDY_MANIFEST),
            "detector_leakage_audit": str(destination / DETECTOR_LEAKAGE_AUDIT),
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_ultralytics_visdrone_det(
    output_root: str | Path,
    *,
    require_image_hashes: bool = True,
) -> dict[str, Any]:
    """Verify the frozen conversion tree and its publisher-source bindings."""

    root = Path(output_root).expanduser().resolve()
    summary_path = root / "CONVERSION_SUMMARY.json"
    manifest_path = root / "CONVERSION_MANIFEST.jsonl"
    dataset_path = root / "visdrone_det.yaml"
    study_manifest_path = root / DETECTOR_STUDY_MANIFEST
    leakage_audit_path = root / DETECTOR_LEAKAGE_AUDIT
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_lines = manifest_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in manifest_lines if line.strip()]
        dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
        study_lines = study_manifest_path.read_text(encoding="utf-8").splitlines()
        study_records = [json.loads(line) for line in study_lines if line.strip()]
        leakage_audit = json.loads(leakage_audit_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise StudyGateError(f"Cannot verify Ultralytics VisDrone tree: {root}") from exc
    if (
        not isinstance(summary, Mapping)
        or not isinstance(dataset, Mapping)
        or not isinstance(leakage_audit, Mapping)
        or not study_records
    ):
        raise StudyGateError("Ultralytics conversion summary/YAML/source audit is malformed")
    if str(summary.get("manifest_sha256", "")) != hash_file(manifest_path):
        raise StudyGateError("Ultralytics conversion manifest hash changed")
    if summary.get("study_source_manifest") != DETECTOR_STUDY_MANIFEST:
        raise StudyGateError("Detector conversion does not name its frozen study manifest")
    if summary.get("detector_leakage_audit") != DETECTOR_LEAKAGE_AUDIT:
        raise StudyGateError("Detector conversion does not name its leakage audit")
    if str(summary.get("study_source_manifest_sha256", "")) != hash_file(
        study_manifest_path
    ):
        raise StudyGateError("Frozen detector study manifest hash changed")
    if str(summary.get("detector_leakage_audit_sha256", "")) != hash_file(
        leakage_audit_path
    ):
        raise StudyGateError("Detector leakage-audit hash changed")
    if leakage_audit.get("schema") != DETECTOR_SOURCE_AUDIT_SCHEMA:
        raise StudyGateError("Detector leakage-audit schema changed")
    if leakage_audit.get("frozen_study_manifest") != DETECTOR_STUDY_MANIFEST:
        raise StudyGateError("Detector leakage audit names the wrong study manifest")
    if str(leakage_audit.get("frozen_study_manifest_sha256", "")) != hash_file(
        study_manifest_path
    ):
        raise StudyGateError("Detector leakage audit is not bound to the study manifest")
    input_manifest = leakage_audit.get("input_manifest")
    if not isinstance(input_manifest, Mapping):
        raise StudyGateError("Detector leakage audit omitted its input-manifest origin")
    origin_kind = str(input_manifest.get("kind", ""))
    if origin_kind == "frozen_file":
        origin_path = Path(str(input_manifest.get("path", ""))).expanduser().resolve()
        origin_sha256 = str(input_manifest.get("sha256", ""))
        if (
            not origin_path.is_file()
            or HEX64.fullmatch(origin_sha256) is None
            or hash_file(origin_path) != origin_sha256
        ):
            raise StudyGateError("Original frozen detector study manifest changed or is missing")
    elif origin_kind == "explicit_dataframe":
        if input_manifest.get("path") or input_manifest.get("sha256"):
            raise StudyGateError("In-memory detector manifest origin metadata is malformed")
    else:
        raise StudyGateError("Detector leakage audit has an unknown input-manifest origin")

    try:
        study_manifest, _ = _read_detector_study_manifest(
            pd.DataFrame.from_records(study_records)
        )
    except (TypeError, ValueError) as exc:
        raise StudyGateError("Frozen detector study manifest is malformed") from exc
    if int(summary.get("images", -1)) != len(records) or not records:
        raise StudyGateError("Ultralytics conversion record count changed or is empty")
    if summary.get("source_partitions") != [VISDRONE_DET_TRAIN_COMPONENT]:
        raise StudyGateError("Detector conversion must consume publisher DET train only")
    if set(summary.get("excluded_partitions", [])) != set(VISDRONE_DET_EXCLUDED_COMPONENTS):
        raise StudyGateError("Detector conversion does not explicitly exclude publisher val/test")
    if summary.get("publisher_validation_used") is not False:
        raise StudyGateError("Publisher DET validation may not drive detector optimization")
    if summary.get("publisher_test_used") is not False:
        raise StudyGateError("Publisher DET test may not drive detector optimization")
    if summary.get("split_assignment_version") != DETECTOR_SPLIT_VERSION:
        raise StudyGateError("Detector natural-group split version changed")
    try:
        split_seed = int(summary["split_assignment_seed"])
        validation_fraction = float(summary["internal_validation_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise StudyGateError("Detector split seed/fraction metadata is invalid") from exc
    if not 0.0 < validation_fraction < 1.0:
        raise StudyGateError("Detector internal-validation fraction is invalid")

    try:
        configured_root = Path(str(dataset.get("path", ""))).expanduser().resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        raise StudyGateError("Ultralytics dataset YAML has an invalid root") from exc
    if configured_root != root:
        raise StudyGateError("Ultralytics dataset YAML points at a different conversion root")
    if (
        dataset.get("train") != f"images/{DETECTOR_TRAIN_PARTITION}"
        or dataset.get("val") != f"images/{DETECTOR_INTERNAL_VALIDATION_PARTITION}"
    ):
        raise StudyGateError(
            "Ultralytics dataset YAML must use detector_train and internal_val from publisher train"
        )
    expected_names = {index: name for index, name in enumerate(VISDRONE_YOLO_NAMES)}
    try:
        names = {
            int(key): str(value)
            for key, value in dict(dataset.get("names", {})).items()
        }
    except (TypeError, ValueError) as exc:
        raise StudyGateError("Ultralytics category mapping is malformed") from exc
    if names != expected_names:
        raise StudyGateError("Ultralytics category mapping changed")
    if require_image_hashes and not bool(summary.get("image_hashes_included", False)):
        raise StudyGateError("Scientific detector preparation requires source-image hashes")

    if any(not isinstance(record, Mapping) for record in records):
        raise StudyGateError("Ultralytics conversion manifest contains a malformed row")
    conversion_source_ids = [
        str(record.get("study_source_id", "")).strip() for record in records
    ]
    if any(not source_id for source_id in conversion_source_ids):
        raise StudyGateError("Detector conversion row omitted its frozen study source ID")
    if len(set(conversion_source_ids)) != len(conversion_source_ids):
        raise StudyGateError("Detector conversion repeats a frozen study source ID")
    recomputed_audit = _detector_leakage_audit(
        study_manifest, conversion_source_ids
    )
    audit_fields = (
        "schema",
        "source_id_column",
        "allowed_study_split",
        "allowed_source_ids",
        "conversion_source_ids",
        "study_split_source_ids",
        "source_id_overlap",
        "split_unit_overlap",
        "conversion_equals_allowed_source_set",
        "zero_forbidden_source_overlap",
        "passed",
    )
    for field in audit_fields:
        if leakage_audit.get(field) != recomputed_audit[field]:
            raise StudyGateError(
                f"Detector leakage audit disagrees with frozen source rows: {field}"
            )
    if (
        recomputed_audit["passed"] is not True
        or recomputed_audit["zero_forbidden_source_overlap"] is not True
        or summary.get("zero_forbidden_source_overlap") is not True
    ):
        raise StudyGateError(
            "Detector source IDs overlap calibration/validation/test or omit allowed train rows"
        )
    allowed_inventory = recomputed_audit["allowed_source_ids"]
    if (
        int(summary.get("allowed_source_id_count", -1))
        != int(allowed_inventory["count"])
        or str(summary.get("allowed_source_ids_sha256", ""))
        != str(allowed_inventory["sha256"])
    ):
        raise StudyGateError("Detector allowed-source ID inventory changed")
    allowed_study_rows = study_manifest.loc[
        study_manifest["component"].eq("det_train")
        & study_manifest["task"].eq("DET")
        & study_manifest["split"].eq(DETECTOR_ALLOWED_STUDY_SPLIT)
    ]
    allowed_study_by_id = {
        str(row.sample_id): row
        for row in allowed_study_rows.itertuples(index=False)
    }

    allowed_partitions = {
        DETECTOR_TRAIN_PARTITION,
        DETECTOR_INTERNAL_VALIDATION_PARTITION,
    }
    observed_study_unit_partitions: dict[str, str] = {}
    observed_study_unit_digests: dict[str, str] = {}
    observed_natural_group_partitions: dict[str, str] = {}
    natural_to_study_unit: dict[str, str] = {}
    study_unit_to_natural: dict[str, str] = {}
    for record in records:
        if record.get("partition") not in allowed_partitions:
            raise StudyGateError("Ultralytics conversion manifest has an invalid partition row")
        if record.get("publisher_component") != VISDRONE_DET_TRAIN_COMPONENT:
            raise StudyGateError("Detector conversion manifest consumes publisher data outside DET train")
        source_id = str(record.get("study_source_id", "")).strip()
        study_row = allowed_study_by_id.get(source_id)
        if study_row is None:
            raise StudyGateError(
                "Detector conversion includes a source outside frozen study split=='train'"
            )
        if (
            record.get("study_split") != DETECTOR_ALLOWED_STUDY_SPLIT
            or str(record.get("study_source_uid", "")) != str(study_row.source_uid)
            or str(record.get("study_split_unit", "")) != str(study_row.split_unit)
        ):
            raise StudyGateError(
                "Detector conversion row disagrees with its frozen study source identity"
            )
        try:
            row_seed = int(record.get("split_assignment_seed", -1))
            row_fraction = float(record.get("internal_validation_fraction", -1.0))
        except (TypeError, ValueError) as exc:
            raise StudyGateError("Detector conversion row has invalid split metadata") from exc
        if (
            record.get("split_assignment_version") != DETECTOR_SPLIT_VERSION
            or row_seed != split_seed
            or row_fraction != validation_fraction
        ):
            raise StudyGateError("Detector conversion row has inconsistent split metadata")
        source_image = Path(str(record.get("source_image", ""))).expanduser().resolve()
        source_annotation = Path(str(record.get("source_annotation", ""))).expanduser().resolve()
        if (
            source_image != Path(str(study_row.media_path)).expanduser().resolve()
            or source_annotation
            != Path(str(study_row.annotation_path)).expanduser().resolve()
        ):
            raise StudyGateError(
                "Detector conversion source paths disagree with the frozen study manifest"
            )
        if any(component in source_image.parts for component in VISDRONE_DET_EXCLUDED_COMPONENTS):
            raise StudyGateError("Detector conversion references publisher validation/test media")
        if any(
            component in source_annotation.parts
            for component in VISDRONE_DET_EXCLUDED_COMPONENTS
        ):
            raise StudyGateError("Detector conversion references publisher validation/test labels")
        if (
            source_image.parent.name != "images"
            or source_annotation.parent.name != "annotations"
            or source_image.parent.parent != source_annotation.parent.parent
            or source_image.parent.parent.name != VISDRONE_DET_TRAIN_COMPONENT
            or source_annotation.stem != source_image.stem
            or source_annotation.suffix.lower() != ".txt"
        ):
            raise StudyGateError(
                "Detector conversion source image/annotation pair is not from publisher DET train"
            )
        natural_group = str(record.get("natural_group", "")).strip()
        if natural_group != _detector_natural_group(source_image.stem):
            raise StudyGateError("Detector conversion natural-group identity changed")
        study_unit = str(record.get("study_split_unit", "")).strip()
        if not study_unit:
            raise StudyGateError("Detector conversion study split-unit identity is empty")
        previous_study_unit = natural_to_study_unit.setdefault(
            natural_group, study_unit
        )
        previous_natural_group = study_unit_to_natural.setdefault(
            study_unit, natural_group
        )
        if (
            previous_study_unit != study_unit
            or previous_natural_group != natural_group
        ):
            raise StudyGateError(
                "Frozen study split units disagree with VisDrone DET natural groups"
            )
        expected_group_digest = hash_object(
            {
                "version": DETECTOR_SPLIT_VERSION,
                "seed": split_seed,
                "natural_group": study_unit,
            }
        )
        if str(record.get("natural_group_split_sha256", "")) != expected_group_digest:
            raise StudyGateError("Detector natural-group split digest changed")
        observed_partition = str(record["partition"])
        previous_partition = observed_study_unit_partitions.setdefault(
            study_unit, observed_partition,
        )
        if previous_partition != observed_partition:
            raise StudyGateError("A detector study group crosses fit/internal-validation partitions")
        natural_partition = observed_natural_group_partitions.setdefault(
            natural_group, observed_partition
        )
        if natural_partition != observed_partition:
            raise StudyGateError("A detector natural group crosses fit/internal-validation partitions")
        observed_study_unit_digests[study_unit] = expected_group_digest
        image_relative = Path(str(record.get("generated_image", "")))
        label_relative = Path(str(record.get("generated_label", "")))
        if (
            image_relative.is_absolute()
            or label_relative.is_absolute()
            or ".." in image_relative.parts
            or ".." in label_relative.parts
        ):
            raise StudyGateError("Ultralytics conversion manifest path escapes its root")
        if (
            image_relative.parts[:2] != ("images", observed_partition)
            or label_relative.parts[:2] != ("labels", observed_partition)
        ):
            raise StudyGateError("Generated detector paths disagree with the natural-group partition")
        if (
            image_relative.name != source_image.name
            or label_relative.stem != source_image.stem
            or label_relative.suffix.lower() != ".txt"
        ):
            raise StudyGateError("Generated detector filenames no longer match their publisher source")
        generated_image_link = root / image_relative
        generated_image = generated_image_link.resolve()
        generated_label = (root / label_relative).resolve()
        try:
            generated_label.relative_to(root)
        except ValueError as exc:
            raise StudyGateError("Generated label path escapes its conversion root") from exc
        if not source_image.is_file() or not source_annotation.is_file():
            raise StudyGateError("A publisher source bound to the detector conversion is missing")
        if not generated_image_link.is_symlink() or generated_image != source_image:
            raise StudyGateError("A generated image link no longer targets its frozen publisher image")
        if not generated_label.is_file():
            raise StudyGateError("A generated Ultralytics label is missing")
        if int(record.get("source_image_bytes", -1)) != source_image.stat().st_size:
            raise StudyGateError("A publisher source image byte count changed")
        expected_image_hash = record.get("source_image_sha256")
        if require_image_hashes and (
            not isinstance(expected_image_hash, str)
            or HEX64.fullmatch(expected_image_hash) is None
            or hash_file(source_image) != expected_image_hash
        ):
            raise StudyGateError("A publisher source image hash changed or is missing")
        if hash_file(source_annotation) != str(record.get("source_annotation_sha256", "")):
            raise StudyGateError("A publisher source annotation hash changed")
        if hash_file(generated_label) != str(record.get("generated_label_sha256", "")):
            raise StudyGateError("A generated Ultralytics label hash changed")

    expected_group_partitions, expected_group_digests = _detector_group_assignments(
        observed_study_unit_partitions,
        seed=split_seed,
        internal_validation_fraction=validation_fraction,
    )
    if observed_study_unit_partitions != expected_group_partitions:
        raise StudyGateError("Detector study-group assignments do not match the frozen seeded hash")
    if observed_study_unit_digests != expected_group_digests:
        raise StudyGateError("Detector study-group digest inventory changed")
    expected_counts = {
        "natural_groups": len(expected_group_partitions),
        "detector_train_groups": sum(
            partition == DETECTOR_TRAIN_PARTITION
            for partition in expected_group_partitions.values()
        ),
        "internal_validation_groups": sum(
            partition == DETECTOR_INTERNAL_VALIDATION_PARTITION
            for partition in expected_group_partitions.values()
        ),
        "detector_train_images": sum(
            str(record["partition"]) == DETECTOR_TRAIN_PARTITION for record in records
        ),
        "internal_validation_images": sum(
            str(record["partition"]) == DETECTOR_INTERNAL_VALIDATION_PARTITION
            for record in records
        ),
    }
    for field, expected in expected_counts.items():
        if int(summary.get(field, -1)) != expected:
            raise StudyGateError(f"Detector conversion summary count changed: {field}")
    return {**dict(summary), "status": "VERIFIED", "output_root": str(root)}


def detector_training_command_plan(
    dataset_yaml: str | Path,
    run_root: str | Path,
    *,
    base_model: str | Path,
    epochs: int = 100,
    image_size: int = 768,
    batch: int = 8,
    workers: int = 8,
    device: str = "0",
    seed: int = 20260823,
    run_name: str = "visdrone_det_registered",
) -> dict[str, Any]:
    """Return (but never execute) the registered Ultralytics training command."""

    dataset = Path(dataset_yaml).expanduser().resolve()
    project = Path(run_root).expanduser().resolve()
    base = Path(base_model).expanduser().resolve()
    if not dataset.is_file():
        raise StudyGateError(f"Ultralytics dataset YAML is missing: {dataset}")
    if dataset.name != "visdrone_det.yaml":
        raise StudyGateError("Detector training requires the frozen VisDrone conversion YAML")
    if not base.is_file():
        raise StudyGateError(
            "Detector training requires a prefetched local base checkpoint"
        )
    prepared = verify_ultralytics_visdrone_det(dataset.parent)
    if int(prepared["split_assignment_seed"]) != int(seed):
        raise StudyGateError(
            "Detector training seed differs from the seed used for the natural-group split"
        )
    if min(epochs, image_size, batch, workers) <= 0 or not run_name.strip():
        raise ValueError("Training epochs, image size, batch, workers, and run name must be positive/populated")
    argv = [
        "yolo",
        "detect",
        "train",
        f"model={base}",
        f"data={dataset}",
        f"project={project}",
        f"name={run_name}",
        f"epochs={epochs}",
        f"imgsz={image_size}",
        f"batch={batch}",
        f"workers={workers}",
        f"device={device}",
        f"seed={seed}",
        "patience=0",
        "deterministic=True",
        "exist_ok=False",
    ]
    return {
        "schema_version": DETECTOR_TRAINING_PLAN_SCHEMA,
        "argv": argv,
        "shell_command": shlex.join(argv),
        "dataset_yaml": str(dataset),
        "dataset_yaml_sha256": hash_file(dataset),
        "base_model_path": str(base),
        "base_model_sha256": hash_file(base),
        "base_model_bytes": base.stat().st_size,
        "conversion_manifest": str(dataset.parent / "CONVERSION_MANIFEST.jsonl"),
        "conversion_manifest_sha256": str(prepared["manifest_sha256"]),
        "conversion_summary": str(dataset.parent / "CONVERSION_SUMMARY.json"),
        "conversion_summary_sha256": hash_file(dataset.parent / "CONVERSION_SUMMARY.json"),
        "study_source_manifest": str(dataset.parent / DETECTOR_STUDY_MANIFEST),
        "study_source_manifest_sha256": str(
            prepared["study_source_manifest_sha256"]
        ),
        "detector_leakage_audit": str(dataset.parent / DETECTOR_LEAKAGE_AUDIT),
        "detector_leakage_audit_sha256": str(
            prepared["detector_leakage_audit_sha256"]
        ),
        "allowed_source_id_count": int(prepared["allowed_source_id_count"]),
        "allowed_source_ids_sha256": str(prepared["allowed_source_ids_sha256"]),
        "zero_forbidden_source_overlap": bool(
            prepared["zero_forbidden_source_overlap"]
        ),
        "publisher_partitions": [VISDRONE_DET_TRAIN_COMPONENT],
        "optimization_partitions": [
            DETECTOR_TRAIN_PARTITION,
            DETECTOR_INTERNAL_VALIDATION_PARTITION,
        ],
        "publisher_validation_used": False,
        "untouched_test_used": False,
        "expected_checkpoint": str(project / run_name / "weights" / "best.pt"),
        "seed": seed,
        "split_assignment_version": str(prepared["split_assignment_version"]),
        "internal_validation_fraction": float(prepared["internal_validation_fraction"]),
    }


def _read_json_mapping(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"Cannot read {label}: {resolved}") from exc
    if not isinstance(payload, dict):
        raise StudyGateError(f"{label} must contain one JSON object")
    return resolved, payload


def _verify_detector_training_plan(
    training_plan_path: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    """Rehash every immutable input consumed by the detector training command."""

    plan_path, plan = _read_json_mapping(
        training_plan_path, label="detector training plan",
    )
    if plan.get("schema_version") != DETECTOR_TRAINING_PLAN_SCHEMA:
        raise StudyGateError("Detector training plan schema is not registered")
    argv = plan.get("argv")
    if (
        not isinstance(argv, list)
        or list(map(str, argv[:3])) != ["yolo", "detect", "train"]
    ):
        raise StudyGateError("Detector training plan does not contain the registered YOLO command")
    if (
        plan.get("zero_forbidden_source_overlap") is not True
        or plan.get("publisher_validation_used") is not False
        or plan.get("untouched_test_used") is not False
        or plan.get("publisher_partitions") != [VISDRONE_DET_TRAIN_COMPONENT]
        or plan.get("optimization_partitions")
        != [DETECTOR_TRAIN_PARTITION, DETECTOR_INTERNAL_VALIDATION_PARTITION]
    ):
        raise StudyGateError("Detector training plan does not preserve the frozen leakage boundary")

    artifacts: dict[str, dict[str, Any]] = {}
    artifact_fields = (
        ("dataset_yaml", "dataset_yaml_sha256"),
        ("conversion_manifest", "conversion_manifest_sha256"),
        ("conversion_summary", "conversion_summary_sha256"),
        ("study_source_manifest", "study_source_manifest_sha256"),
        ("detector_leakage_audit", "detector_leakage_audit_sha256"),
        ("base_model_path", "base_model_sha256"),
    )
    for path_field, hash_field in artifact_fields:
        artifact = Path(str(plan.get(path_field, ""))).expanduser().resolve()
        expected_hash = str(plan.get(hash_field, "")).strip().lower()
        if not artifact.is_file() or HEX64.fullmatch(expected_hash) is None:
            raise StudyGateError(
                f"Detector training plan has an invalid frozen artifact: {path_field}"
            )
        observed_hash = hash_file(artifact)
        if observed_hash != expected_hash:
            raise StudyGateError(f"Detector training input changed after planning: {path_field}")
        artifacts[path_field] = {
            "path": str(artifact),
            "sha256": observed_hash,
            "bytes": artifact.stat().st_size,
        }
    if artifacts["dataset_yaml"]["path"] != str(
        Path(str(plan["dataset_yaml"])).expanduser().resolve()
    ) or Path(artifacts["dataset_yaml"]["path"]).name != "visdrone_det.yaml":
        raise StudyGateError("Detector training plan dataset YAML is not the frozen conversion")
    if int(plan.get("base_model_bytes", -1)) != int(
        artifacts["base_model_path"]["bytes"]
    ):
        raise StudyGateError("Detector base checkpoint byte count changed")
    expected_checkpoint = Path(str(plan.get("expected_checkpoint", ""))).expanduser().resolve()
    if expected_checkpoint.name != "best.pt" or expected_checkpoint.parent.name != "weights":
        raise StudyGateError("Detector training plan does not target the registered best.pt artifact")
    return plan_path, plan, artifacts


def _training_argument_map(argv: Sequence[Any]) -> dict[str, str]:
    arguments: dict[str, str] = {}
    for token in map(str, argv[3:]):
        if "=" not in token:
            raise StudyGateError(f"Detector training argument is not key=value: {token!r}")
        key, value = token.split("=", 1)
        if not key or key in arguments:
            raise StudyGateError("Detector training arguments contain an empty or duplicate key")
        arguments[key] = value
    expected = {
        "model", "data", "project", "name", "epochs", "imgsz", "batch",
        "workers", "device", "seed", "patience", "deterministic", "exist_ok",
    }
    if set(arguments) != expected:
        raise StudyGateError("Detector training arguments differ from the registered command surface")
    return arguments


def _normalized_training_argument(key: str, value: Any) -> Any:
    if key in {"model", "data", "project"}:
        return str(Path(str(value)).expanduser().resolve())
    if key in {"epochs", "imgsz", "batch", "workers", "seed", "patience"}:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(f"Detector training argument {key!r} is not an integer") from exc
    if key in {"deterministic", "exist_ok"}:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
        raise StudyGateError(f"Detector training argument {key!r} is not boolean")
    return str(value)


def _detector_training_receipt_payload(
    training_plan_path: str | Path,
    *,
    checkpoint: str | Path,
    environment_fingerprint_path: str | Path,
) -> dict[str, Any]:
    plan_path, plan, artifacts = _verify_detector_training_plan(training_plan_path)
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    expected_checkpoint = Path(str(plan["expected_checkpoint"])).expanduser().resolve()
    if checkpoint_path != expected_checkpoint:
        raise StudyGateError("Detector checkpoint path differs from the registered training plan")
    if checkpoint_path.name != "best.pt" or not checkpoint_path.is_file():
        raise StudyGateError(f"Registered detector best.pt is missing: {checkpoint_path}")
    run_root = checkpoint_path.parent.parent
    args_path = run_root / "args.yaml"
    results_path = run_root / "results.csv"
    if not args_path.is_file() or not results_path.is_file():
        raise StudyGateError("Detector training run lacks args.yaml or results.csv")
    try:
        args_payload = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise StudyGateError("Cannot read detector training args.yaml") from exc
    if not isinstance(args_payload, Mapping):
        raise StudyGateError("Detector training args.yaml must contain one mapping")
    expected_arguments = _training_argument_map(plan["argv"])
    for key, expected in expected_arguments.items():
        if key not in args_payload:
            raise StudyGateError(f"Detector training args.yaml is missing {key!r}")
        if _normalized_training_argument(key, args_payload[key]) != _normalized_training_argument(
            key, expected,
        ):
            raise StudyGateError(f"Detector training args.yaml differs from the plan: {key}")
    if str(args_payload.get("task", "detect")) != "detect" or str(
        args_payload.get("mode", "train")
    ) != "train":
        raise StudyGateError("Detector training args.yaml task/mode is not detect/train")

    try:
        results_text = results_path.read_text(encoding="utf-8")
        raw_header = next(csv.reader(io.StringIO(results_text)), [])
        normalized_header = [str(column).strip() for column in raw_header]
        if len(normalized_header) != len(set(normalized_header)):
            raise StudyGateError("Detector training results.csv has duplicate columns")
        results = pd.read_csv(
            io.StringIO(results_text), dtype=str, keep_default_na=False,
        )
    except StudyGateError:
        raise
    except (
        OSError, UnicodeError, csv.Error,
        pd.errors.ParserError, pd.errors.EmptyDataError,
    ) as exc:
        raise StudyGateError("Cannot read detector training results.csv") from exc
    results.columns = [str(column).strip() for column in results.columns]
    if results.empty or "epoch" not in results.columns:
        raise StudyGateError("Detector training results.csv is empty or lacks epoch")
    expected_epochs = int(expected_arguments["epochs"])
    epoch_tokens = results["epoch"].tolist()
    if len(epoch_tokens) != expected_epochs:
        raise StudyGateError("Detector training did not complete every registered epoch")
    for expected_epoch, token in enumerate(epoch_tokens, start=1):
        try:
            observed_epoch = Decimal(str(token).strip())
        except InvalidOperation as exc:
            raise StudyGateError(
                "Detector training did not complete every registered epoch"
            ) from exc
        if not observed_epoch.is_finite() or observed_epoch != Decimal(expected_epoch):
            raise StudyGateError("Detector training did not complete every registered epoch")

    environment_path, environment = _read_json_mapping(
        environment_fingerprint_path, label="training environment fingerprint",
    )
    if environment.get("schema_version") != DETECTOR_ENVIRONMENT_SCHEMA:
        raise StudyGateError("Detector training requires a v2 environment fingerprint")
    return {
        "schema_version": DETECTOR_TRAINING_RECEIPT_SCHEMA,
        "training_plan": {
            "path": str(plan_path),
            "sha256": hash_file(plan_path),
            "object_sha256": hash_object(plan),
        },
        "training_environment": {
            "path": str(environment_path),
            "sha256": hash_file(environment_path),
            "object_sha256": hash_object(environment),
            "schema_version": DETECTOR_ENVIRONMENT_SCHEMA,
        },
        "frozen_training_inputs": artifacts,
        "run_artifacts": {
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": hash_file(checkpoint_path),
                "bytes": checkpoint_path.stat().st_size,
            },
            "args_yaml": {
                "path": str(args_path.resolve()),
                "sha256": hash_file(args_path),
                "bytes": args_path.stat().st_size,
            },
            "results_csv": {
                "path": str(results_path.resolve()),
                "sha256": hash_file(results_path),
                "bytes": results_path.stat().st_size,
                "completed_epochs": expected_epochs,
            },
        },
        "leakage_boundary": {
            "publisher_partitions": [VISDRONE_DET_TRAIN_COMPONENT],
            "optimization_partitions": [
                DETECTOR_TRAIN_PARTITION,
                DETECTOR_INTERNAL_VALIDATION_PARTITION,
            ],
            "publisher_validation_used": False,
            "untouched_test_used": False,
            "zero_forbidden_source_overlap": True,
            "allowed_source_id_count": int(plan["allowed_source_id_count"]),
            "allowed_source_ids_sha256": str(plan["allowed_source_ids_sha256"]),
        },
    }


def write_detector_training_receipt(
    training_plan_path: str | Path,
    *,
    checkpoint: str | Path,
    environment_fingerprint_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Freeze a completed Ultralytics run and every input that produced it."""

    destination = Path(output_path).expanduser().resolve()
    if destination.exists():
        raise StudyGateError(f"Detector training receipt already exists: {destination}")
    payload = _detector_training_receipt_payload(
        training_plan_path,
        checkpoint=checkpoint,
        environment_fingerprint_path=environment_fingerprint_path,
    )
    atomic_write_json(destination, payload)
    return payload


def verify_detector_training_receipt(
    receipt_path: str | Path,
    *,
    checkpoint: str | Path | None = None,
    expected_checkpoint_sha256: str | None = None,
    training_plan_path: str | Path | None = None,
    environment_fingerprint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute and verify a detector-training provenance receipt."""

    resolved_receipt, receipt = _read_json_mapping(
        receipt_path, label="detector training receipt",
    )
    if receipt.get("schema_version") != DETECTOR_TRAINING_RECEIPT_SCHEMA:
        raise StudyGateError("Detector training receipt schema is not registered")
    try:
        stored_plan = receipt["training_plan"]["path"]
        stored_environment = receipt["training_environment"]["path"]
        stored_checkpoint = receipt["run_artifacts"]["checkpoint"]["path"]
    except (KeyError, TypeError) as exc:
        raise StudyGateError("Detector training receipt structure is incomplete") from exc
    selected_plan = Path(training_plan_path or stored_plan).expanduser().resolve()
    selected_environment = Path(
        environment_fingerprint_path or stored_environment
    ).expanduser().resolve()
    selected_checkpoint = Path(checkpoint or stored_checkpoint).expanduser().resolve()
    if selected_plan != Path(str(stored_plan)).expanduser().resolve():
        raise StudyGateError("Configured detector training plan differs from its receipt")
    if selected_environment != Path(str(stored_environment)).expanduser().resolve():
        raise StudyGateError("Configured training environment differs from its receipt")
    if selected_checkpoint != Path(str(stored_checkpoint)).expanduser().resolve():
        raise StudyGateError("Configured detector checkpoint differs from its training receipt")
    recomputed = _detector_training_receipt_payload(
        selected_plan,
        checkpoint=selected_checkpoint,
        environment_fingerprint_path=selected_environment,
    )
    if receipt != recomputed:
        raise StudyGateError("Detector training receipt no longer matches its frozen inputs/run")
    observed_checkpoint_hash = str(
        recomputed["run_artifacts"]["checkpoint"]["sha256"]
    )
    if expected_checkpoint_sha256 is not None:
        expected_hash = str(expected_checkpoint_sha256).strip().lower()
        if HEX64.fullmatch(expected_hash) is None or expected_hash != observed_checkpoint_hash:
            raise StudyGateError("Detector training receipt checkpoint SHA-256 mismatch")
    return {
        **recomputed,
        "receipt_path": str(resolved_receipt),
        "receipt_sha256": hash_file(resolved_receipt),
        "status": "VERIFIED",
    }


def verify_detector_checkpoint(
    checkpoint: str | Path,
    expected_sha256: str,
    *,
    training_receipt: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a frozen detector checkpoint without loading or modifying it."""

    path = Path(checkpoint).expanduser().resolve()
    expected = str(expected_sha256).strip().lower()
    if HEX64.fullmatch(expected) is None:
        raise StudyGateError("Expected detector SHA-256 must be 64 lowercase hex characters")
    if not path.is_file():
        raise StudyGateError(f"Detector checkpoint is missing: {path}")
    observed = hash_file(path)
    if observed != expected:
        raise StudyGateError(
            f"Detector checkpoint hash mismatch (expected {expected}, observed {observed})"
        )
    record = {
        "checkpoint": str(path),
        "sha256": observed,
        "bytes": path.stat().st_size,
        "status": "VERIFIED",
    }
    if training_receipt is not None:
        receipt = verify_detector_training_receipt(
            training_receipt,
            checkpoint=path,
            expected_checkpoint_sha256=expected,
        )
        record.update({
            "training_receipt": str(receipt["receipt_path"]),
            "training_receipt_sha256": str(receipt["receipt_sha256"]),
            "training_plan_sha256": str(receipt["training_plan"]["sha256"]),
            "training_environment_sha256": str(
                receipt["training_environment"]["sha256"]
            ),
        })
    return record


@contextlib.contextmanager
def _local_only_models() -> Iterable[None]:
    previous = os.environ.get("SEMUAV_MODELS_LOCAL_ONLY")
    os.environ["SEMUAV_MODELS_LOCAL_ONLY"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SEMUAV_MODELS_LOCAL_ONLY", None)
        else:
            os.environ["SEMUAV_MODELS_LOCAL_ONLY"] = previous


def registered_model_load_reference(entry: Mapping[str, Any]) -> str:
    """Return the immutable Transformers load reference for a verified lock row.

    Every active v2r13 model loads directly from its already verified snapshot
    directory.  The historical Moondream branch is retained solely to verify
    or reproduce quarantined v2r5 development evidence; an active v2r13 model
    lock cannot contain that row.
    """

    backend = str(entry.get("backend", ""))
    if backend != "moondream2":
        return str(entry["snapshot_path"])

    model_id = str(entry.get("model_id", ""))
    repo = str(entry.get("repo", ""))
    revision = str(entry.get("resolved_revision", "")).lower()
    if (
        model_id != HISTORICAL_MOONDREAM_MODEL_ID
        or repo != HISTORICAL_MOONDREAM_MODEL_REPO
        or HEX40.fullmatch(revision) is None
    ):
        raise StudyGateError(
            "Moondream loader identity is inconsistent with the registered lock"
        )
    return repo


def bind_moondream_tokenizer_lock(
    model_lock_payload: Mapping[str, Any],
    tokenizer_lock_path: str | Path,
    *,
    offline_checker: AuxiliaryArtifactChecker | None = None,
) -> dict[str, Any]:
    """Bind a verified auxiliary tokenizer to an in-memory model-lock copy.

    The historical MODEL_LOCK.json remains byte-for-byte unchanged.  Runtime
    callers receive a derived mapping whose one Moondream row carries the
    separately authenticated Starmie identity.
    """

    verified = verify_moondream_tokenizer_lock(
        tokenizer_lock_path,
        offline_checker=offline_checker,
    )
    models = model_lock_payload.get("models")
    if not isinstance(models, list):
        raise StudyGateError("MODEL_LOCK.json models must be a list")
    tokenizer = verified["artifacts"][0]
    bound_record = {
        "repo": verified["repo"],
        "revision": verified["resolved_revision"],
        "filename": tokenizer["filename"],
        "path": tokenizer["snapshot_path"],
        "cache_root": verified["cache_root"],
        "bytes": tokenizer["bytes"],
        "sha256": tokenizer["sha256"],
        "lock_path": verified["lock_path"],
        "lock_sha256": verified["lock_sha256"],
    }
    bound_models: list[dict[str, Any]] = []
    matches = 0
    for raw in models:
        if not isinstance(raw, Mapping):
            raise StudyGateError("MODEL_LOCK.json contains a non-object model entry")
        entry = dict(raw)
        if str(entry.get("model_id", "")) == "moondream2":
            matches += 1
            if "moondream_tokenizer" in entry:
                raise StudyGateError("MODEL_LOCK row already contains a tokenizer binding")
            entry["moondream_tokenizer"] = dict(bound_record)
        bound_models.append(entry)
    if matches != 1:
        raise StudyGateError("MODEL_LOCK.json must contain one Moondream row")
    return {
        **dict(model_lock_payload),
        "models": bound_models,
        "moondream_tokenizer_lock": {
            "path": verified["lock_path"],
            "sha256": verified["lock_sha256"],
            "repo": verified["repo"],
            "revision": verified["resolved_revision"],
            "artifact_sha256": tokenizer["sha256"],
        },
    }


def verify_bound_moondream_tokenizer_artifact(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-authenticate one derived runtime binding against its cited lock."""

    expected_keys = {
        "repo", "revision", "filename", "path", "cache_root", "bytes",
        "sha256", "lock_path", "lock_sha256",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise StudyGateError(
            "Moondream row lacks its exact auxiliary Starmie tokenizer binding"
        )
    if (
        record["repo"] != MOONDREAM_TOKENIZER_REPO
        or record["revision"] != MOONDREAM_TOKENIZER_REVISION
        or record["filename"] != MOONDREAM_TOKENIZER_FILENAME
        or type(record["bytes"]) is not int
        or record["bytes"] != MOONDREAM_TOKENIZER_BYTES
        or record["sha256"] != MOONDREAM_TOKENIZER_SHA256
        or HEX64.fullmatch(str(record["lock_sha256"])) is None
    ):
        raise StudyGateError("Moondream auxiliary tokenizer binding is invalid")

    def bound_checker(
        repo: str,
        revision: str,
        filename: str,
        cache_root: Path,
    ) -> str:
        if (
            repo != record["repo"]
            or revision != record["revision"]
            or filename != record["filename"]
            or cache_root != _lexical_absolute(str(record["cache_root"]))
        ):
            raise StudyGateError("Moondream auxiliary tokenizer binding changed")
        return str(record["path"])

    verified = verify_moondream_tokenizer_lock(
        str(record["lock_path"]),
        offline_checker=bound_checker,
    )
    tokenizer = verified["artifacts"][0]
    expected = {
        "repo": verified["repo"],
        "revision": verified["resolved_revision"],
        "filename": tokenizer["filename"],
        "path": tokenizer["snapshot_path"],
        "cache_root": verified["cache_root"],
        "bytes": tokenizer["bytes"],
        "sha256": tokenizer["sha256"],
        "lock_path": verified["lock_path"],
        "lock_sha256": verified["lock_sha256"],
    }
    if dict(record) != expected:
        raise StudyGateError(
            "Moondream auxiliary tokenizer binding differs from its cited lock"
        )
    return expected


def registered_moondream_tokenizer_artifact(
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return and recheck the separately locked Starmie tokenizer record."""

    if (
        str(entry.get("backend", "")) != "moondream2"
        or registered_model_load_reference(entry)
        != HISTORICAL_MOONDREAM_MODEL_REPO
    ):
        raise StudyGateError("Tokenizer artifact requested for a non-Moondream lock row")
    record = entry.get("moondream_tokenizer")
    if not isinstance(record, Mapping):
        raise StudyGateError(
            "Moondream row lacks its exact auxiliary Starmie tokenizer binding"
        )
    return verify_bound_moondream_tokenizer_artifact(record)


def _moondream_tokenizer_runtime_identity(
    bound_model_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact auxiliary-tokenizer identity bound to runtime rows."""

    tokenizer_lock = bound_model_lock.get("moondream_tokenizer_lock")
    models = bound_model_lock.get("models")
    if not isinstance(tokenizer_lock, Mapping) or not isinstance(models, list):
        raise StudyGateError("Runtime model lock lacks its tokenizer-lock binding")
    moon_rows = [
        entry for entry in models
        if isinstance(entry, Mapping)
        and str(entry.get("model_id", "")) == "moondream2"
    ]
    if len(moon_rows) != 1:
        raise StudyGateError("Bound runtime model set lacks one Moondream row")
    tokenizer_artifact = registered_moondream_tokenizer_artifact(moon_rows[0])
    identity = {
        "lock_path": str(tokenizer_lock.get("path", "")),
        "lock_sha256": str(tokenizer_lock.get("sha256", "")),
        "repo": str(tokenizer_lock.get("repo", "")),
        "revision": str(tokenizer_lock.get("revision", "")),
        "artifact_filename": str(tokenizer_artifact["filename"]),
        "artifact_path": str(tokenizer_artifact["path"]),
        "artifact_bytes": int(tokenizer_artifact["bytes"]),
        "artifact_sha256": str(tokenizer_artifact["sha256"]),
    }
    if (
        Path(identity["lock_path"]).expanduser().resolve()
        != Path(str(tokenizer_artifact["lock_path"])).expanduser().resolve()
        or identity["lock_sha256"] != tokenizer_artifact["lock_sha256"]
        or identity["repo"] != tokenizer_artifact["repo"]
        or identity["revision"] != tokenizer_artifact["revision"]
        or HEX64.fullmatch(identity["artifact_sha256"]) is None
        or identity["artifact_bytes"] <= 0
    ):
        raise StudyGateError("Runtime Moondream tokenizer identity is inconsistent")
    return identity


def moondream_tokenizer_runtime_identity(
    bound_model_lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Public exact eight-field tokenizer identity used by every phase."""

    return _moondream_tokenizer_runtime_identity(bound_model_lock)


def _default_backend_factory(entry: Mapping[str, Any], device: str, dtype: str) -> Any:
    from .perception import HFMultimodalBackend, ModelSpec

    backend = str(entry.get("backend", ""))
    if backend not in {"qwen3_vl", "smolvlm2"}:
        raise StudyGateError(
            f"Inactive or unsupported v2r13 model backend: {backend!r}"
        )
    spec = ModelSpec(
        model_id=str(entry["model_id"]),
        repo=registered_model_load_reference(entry),
        revision=str(entry["resolved_revision"]),
        role=str(entry["role"]),
        quantization=str(entry.get("quantization", "none")),
        trust_remote_code=False,
    )
    return HFMultimodalBackend(spec, device=device, dtype=str(entry.get("dtype", dtype)))


def _runtime_smoke_input_receipt(
    request: Any,
    *,
    image_sha256: str,
) -> dict[str, Any]:
    """Bind one path-free VisDrone smoke request to its exact model inputs."""

    observation_id = str(getattr(request, "observation_id", "")).strip()
    mission = str(getattr(request, "mission", "")).strip()
    capture_time_us = getattr(request, "capture_time_us", None)
    telemetry = getattr(request, "telemetry", None)
    detector_proposals = getattr(request, "detector_proposals", None)
    if not observation_id:
        raise StudyGateError("Runtime smoke request lacks observation_id")
    if mission != "visdrone_urban":
        raise StudyGateError("Runtime smoke request must use mission 'visdrone_urban'")
    if isinstance(capture_time_us, bool) or not isinstance(capture_time_us, int):
        raise StudyGateError("Runtime smoke capture_time_us must be an integer")
    if not isinstance(telemetry, Mapping):
        raise StudyGateError("Runtime smoke telemetry must be a mapping")
    if (
        not isinstance(detector_proposals, Sequence)
        or isinstance(detector_proposals, (str, bytes))
        or any(not isinstance(item, Mapping) for item in detector_proposals)
    ):
        raise StudyGateError("Runtime smoke detector_proposals must be a sequence of mappings")
    proposals = list(detector_proposals)
    request_payload = {
        "observation_id": observation_id,
        "mission": mission,
        "capture_time_us": int(capture_time_us),
        "telemetry": dict(telemetry),
        "image_sha256": str(image_sha256),
        "detector_proposals": proposals,
    }
    return {
        "input_contract": VISDRONE_RUNTIME_SMOKE_INPUT_CONTRACT,
        "request_sha256": hash_object(request_payload),
        "detector_proposals_sha256": hash_object(proposals),
        "detector_proposal_count": len(proposals),
    }


_MOONDREAM_RUNTIME_BUDGET_FIELDS = (
    "moondream_image_positions",
    "moondream_native_template_tokens",
    "moondream_question_tokens",
    "moondream_prefill_tokens",
    "moondream_generation_reserve_tokens",
    "moondream_required_tokens",
    "moondream_effective_context_tokens",
    "moondream_context_margin_tokens",
)
_MOONDREAM_RUNTIME_PROVENANCE_FIELDS = (
    *_MOONDREAM_RUNTIME_BUDGET_FIELDS,
    "moondream_tokenizer_lock_sha256",
    "moondream_tokenizer_artifact_sha256",
)


def _moondream_runtime_row_provenance_valid(
    row: Mapping[str, Any],
    *,
    tokenizer_identity: Mapping[str, Any],
    max_new_tokens: int,
) -> bool:
    values = [row.get(name) for name in _MOONDREAM_RUNTIME_BUDGET_FIELDS]
    if any(type(value) is not int for value in values):
        return False
    (
        image_positions,
        template_tokens,
        question_tokens,
        prefill_tokens,
        reserve_tokens,
        required_tokens,
        context_tokens,
        margin_tokens,
    ) = values
    return bool(
        image_positions > 0
        and template_tokens > 0
        and question_tokens > 0
        and prefill_tokens == image_positions + template_tokens + question_tokens
        and reserve_tokens == max_new_tokens
        and required_tokens == prefill_tokens + reserve_tokens
        and context_tokens == 2048
        and margin_tokens == context_tokens - required_tokens
        and margin_tokens >= 0
        and row.get("input_tokens") == prefill_tokens
        and row.get("moondream_tokenizer_lock_sha256")
        == tokenizer_identity.get("lock_sha256")
        and row.get("moondream_tokenizer_artifact_sha256")
        == tokenizer_identity.get("artifact_sha256")
    )


def _empty_runtime_smoke_row(
    entry: Mapping[str, Any],
    *,
    image_path: Path,
    image_sha256: str,
    input_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete fail-closed row envelope for one locked model."""

    return {
        "model_id": str(entry["model_id"]),
        "repo": str(entry["repo"]),
        "revision": str(entry["resolved_revision"]),
        "role": str(entry["role"]),
        "status": "ERROR",
        "schema_valid": False,
        "error": None,
        "latency_ms": None,
        "record_model_id": None,
        "record_revision": None,
        "record_observation_id": None,
        "record_prompt_hash": None,
        "raw_text": None,
        "raw_text_sha256": None,
        "parsed_packet_sha256": None,
        "input_tokens": None,
        "output_tokens": None,
        "decision_contract_version": None,
        "decision_core_valid": None,
        "decision_core_json": None,
        "decision_context_sha256": None,
        "decision_proposal_count": None,
        "decision_composer_policy_sha256": None,
        "terminated_by_eos": None,
        "reached_token_cap": None,
        "generation_policy_version": None,
        "generation_policy_sha256": None,
        "generation_strategy": None,
        "generation_max_new_tokens": None,
        "assistant_prefill_sha256": None,
        "assistant_prefill_characters": None,
        "generated_continuation_sha256": None,
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        **dict(input_receipt),
    }


def _json_safe_runtime_smoke_row(raw_row: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize one row without losing nullable integer token counts."""

    clean: dict[str, Any] = {}
    optional_token_count_fields = {"input_tokens", "output_tokens"}
    for key, value in raw_row.items():
        if key in optional_token_count_fields:
            if value is None:
                clean[key] = None
                continue
            if isinstance(value, bool) or not isinstance(value, Real):
                raise StudyGateError(
                    f"Runtime smoke {key} must be a nonnegative integer or null"
                )
            numeric = float(value)
            if (
                not math.isfinite(numeric)
                or numeric < 0
                or not numeric.is_integer()
            ):
                raise StudyGateError(
                    f"Runtime smoke {key} must be a nonnegative integer or null"
                )
            clean[key] = int(value)
        elif pd.isna(value):
            clean[key] = None
        elif hasattr(value, "item"):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


def _execute_runtime_smoke_entry(
    entry: Mapping[str, Any],
    request: Any,
    *,
    image_path: Path,
    image_sha256: str,
    input_receipt: Mapping[str, Any],
    prompt_text: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    expected_decision_context_sha256: str,
    expected_generation_policy_sha256: str,
    backend_factory: BackendFactory,
    cleanup_cuda: bool,
) -> tuple[dict[str, Any], str | None]:
    """Execute exactly one backend and return its row plus any cleanup error.

    The production worker passes ``cleanup_cuda=False`` because the process is
    disposable: process exit, rather than another CUDA API call, destroys its
    context.  The in-process dependency-injection path keeps the historical
    cleanup behavior for unit tests.
    """

    from .perception import (
        VISDRONE_GENERATION_POLICY_VERSION,
        VISDRONE_GENERATION_STRATEGY_BY_MODEL,
        VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
        visdrone_assistant_prefill_for_model,
    )

    started = time.perf_counter()
    row = _empty_runtime_smoke_row(
        entry,
        image_path=image_path,
        image_sha256=image_sha256,
        input_receipt=input_receipt,
    )
    backend = None
    cleanup_error: str | None = None
    try:
        with _local_only_models():
            backend = backend_factory(entry, device, dtype)
            record = backend.infer(
                request,
                prompt=prompt_text,
                max_new_tokens=max_new_tokens,
            )
        record_model = str(getattr(record, "model_id", ""))
        record_revision = str(getattr(record, "model_revision", ""))
        record_observation_id = str(getattr(record, "observation_id", ""))
        record_prompt_hash = str(getattr(record, "prompt_hash", ""))
        schema_valid = bool(getattr(record, "schema_valid", False))
        raw_text = getattr(record, "raw_text", None)
        parsed_packet = getattr(record, "parsed", None)
        row.update(
            {
                "record_model_id": record_model,
                "record_revision": record_revision,
                "record_observation_id": record_observation_id,
                "record_prompt_hash": record_prompt_hash,
                "schema_valid": schema_valid,
                "latency_ms": getattr(record, "latency_ms", None),
                "error": getattr(record, "error", None),
                "raw_text": raw_text if isinstance(raw_text, str) else None,
                "raw_text_sha256": (
                    sha256_bytes(raw_text.encode("utf-8"))
                    if isinstance(raw_text, str) else None
                ),
                "parsed_packet_sha256": (
                    hash_object(parsed_packet)
                    if isinstance(parsed_packet, Mapping) else None
                ),
                "input_tokens": getattr(record, "input_tokens", None),
                "output_tokens": getattr(record, "output_tokens", None),
                "decision_contract_version": getattr(
                    record, "decision_contract_version", None
                ),
                "decision_core_valid": getattr(
                    record, "decision_core_valid", None
                ),
                "decision_core_json": getattr(
                    record, "decision_core_json", None
                ),
                "decision_context_sha256": getattr(
                    record, "decision_context_sha256", None
                ),
                "decision_proposal_count": getattr(
                    record, "decision_proposal_count", None
                ),
                "decision_composer_policy_sha256": getattr(
                    record, "decision_composer_policy_sha256", None
                ),
                "terminated_by_eos": getattr(record, "terminated_by_eos", None),
                "reached_token_cap": getattr(record, "reached_token_cap", None),
                "generation_policy_version": getattr(
                    record, "generation_policy_version", None
                ),
                "generation_policy_sha256": getattr(
                    record, "generation_policy_sha256", None
                ),
                "generation_strategy": getattr(
                    record, "generation_strategy", None
                ),
                "generation_max_new_tokens": getattr(
                    record, "generation_max_new_tokens", None
                ),
                "assistant_prefill_sha256": getattr(
                    record, "assistant_prefill_sha256", None
                ),
                "assistant_prefill_characters": getattr(
                    record, "assistant_prefill_characters", None
                ),
                "generated_continuation_sha256": getattr(
                    record, "generated_continuation_sha256", None
                ),
            }
        )
        expected_strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL.get(
            row["model_id"]
        )
        expected_prefill = visdrone_assistant_prefill_for_model(row["model_id"])
        generated_continuation = None
        if isinstance(raw_text, str) and raw_text.startswith(expected_prefill):
            generated_continuation = raw_text[len(expected_prefill):]
        provenance_valid = (
            record_model == row["model_id"]
            and record_revision == row["revision"]
            and record_observation_id == str(getattr(request, "observation_id"))
            and record_prompt_hash == hash_object(prompt_text)
            and isinstance(raw_text, str)
            and bool(raw_text)
            and row["decision_contract_version"]
            == VISDRONE_RUNTIME_SMOKE_INPUT_CONTRACT
            and row["decision_proposal_count"]
            == input_receipt["detector_proposal_count"]
            and row["decision_context_sha256"]
            == expected_decision_context_sha256
            and row["decision_composer_policy_sha256"]
            == VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
            and row["generation_policy_version"]
            == VISDRONE_GENERATION_POLICY_VERSION
            and row["generation_policy_sha256"]
            == expected_generation_policy_sha256
            and row["generation_strategy"] == expected_strategy
            and row["generation_max_new_tokens"] == max_new_tokens
            and row["assistant_prefill_sha256"]
            == sha256_bytes(expected_prefill.encode("utf-8"))
            and row["assistant_prefill_characters"] == len(expected_prefill)
            and isinstance(generated_continuation, str)
            and row["generated_continuation_sha256"]
            == sha256_bytes(generated_continuation.encode("utf-8"))
        )
        if not provenance_valid:
            row["status"] = "FAIL_PROVENANCE"
            row["error"] = (
                "Inference record identity, prompt, decision contract, or composer "
                "lineage differs from the bound runtime-smoke input"
            )
        elif (
            not schema_valid
            or row["decision_core_valid"] is not True
            or row["parsed_packet_sha256"] is None
        ):
            row["status"] = "FAIL_SCHEMA"
            row["error"] = row["error"] or "Inference output failed the frozen schema"
        else:
            row["status"] = "PASS"
    except Exception as exc:  # every failure must be represented in the report
        row["status"] = "ERROR"
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        row["smoke_wall_ms"] = (time.perf_counter() - started) * 1000.0
        if cleanup_cuda:
            try:
                del backend
                gc.collect()
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception as exc:
                cleanup_error = f"{type(exc).__name__}: {exc}"
                primary_error = row.get("error")
                row["status"] = "ERROR"
                row["error"] = (
                    f"{primary_error}; runtime cleanup failed: {cleanup_error}"
                    if primary_error
                    else f"Runtime cleanup failed: {cleanup_error}"
                )
    return row, cleanup_error


_WORKER_ENVIRONMENT_NAMES = (
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_CACHE",
    "HF_MODULES_CACHE",
    "TRANSFORMERS_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "SEMUAV_MODELS_LOCAL_ONLY",
    "SEMUAV_QWEN_REVISION",
    "SEMUAV_SMOLVLM_REVISION",
    "SEMUAV_UPPER_BOUND_REVISION",
    "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
    "TOKENIZERS_PARALLELISM",
)
_WORKER_FORBIDDEN_ENVIRONMENT_NAMES = (
    "BASH_ENV",
    "ENV",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONHOME",
    "PYTHONPATH",
)
_WORKER_REMOVED_LEGACY_ENVIRONMENT_NAMES = (
    "SEMUAV_MOONDREAM_REVISION",
    "SEMUAV_MOONDREAM_TOKENIZER_LOCK",
)


def _worker_environment() -> tuple[dict[str, str], dict[str, str | None]]:
    """Build the exact inherited environment and its compact attestation."""

    environment = dict(os.environ)
    for name in (
        *_WORKER_FORBIDDEN_ENVIRONMENT_NAMES,
        *_WORKER_REMOVED_LEGACY_ENVIRONMENT_NAMES,
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["TOKENIZERS_PARALLELISM"] = "false"
    environment["SEMUAV_MODELS_LOCAL_ONLY"] = "1"
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    attested: dict[str, str | None] = {
        name: environment.get(name) for name in _WORKER_ENVIRONMENT_NAMES
    }
    attested.update({
        name: None for name in _WORKER_FORBIDDEN_ENVIRONMENT_NAMES
    })
    return environment, attested


def _worker_request_payload(
    entry: Mapping[str, Any],
    request: Any,
    *,
    lock_path: Path,
    model_lock_sha256: str,
    code_sha256: str,
    image_path: Path,
    image_sha256: str,
    input_receipt: Mapping[str, Any],
    prompt_text: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
    worker_script: Path,
    worker_environment: Mapping[str, str | None],
) -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_SMOKE_WORKER_INPUT_SCHEMA,
        "model_id": str(entry["model_id"]),
        "model_entry_sha256": hash_object(entry),
        "model_lock_path": str(lock_path),
        "model_lock_sha256": model_lock_sha256,
        "code_sha256": code_sha256,
        "worker_script_path": str(worker_script),
        "worker_script_sha256": hash_file(worker_script),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": hash_file(Path(sys.executable).resolve()),
        "device": str(device),
        "dtype": str(entry.get("dtype", dtype)),
        "max_new_tokens": int(max_new_tokens),
        "prompt": prompt_text,
        "prompt_sha256": hash_object(prompt_text),
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "input_receipt": dict(input_receipt),
        "request": {
            "observation_id": str(getattr(request, "observation_id")),
            "image_path": str(image_path),
            "mission": str(getattr(request, "mission")),
            "capture_time_us": int(getattr(request, "capture_time_us")),
            "telemetry": dict(getattr(request, "telemetry")),
            "detector_proposals": list(getattr(request, "detector_proposals")),
        },
        "worker_environment": dict(worker_environment),
        "worker_environment_sha256": hash_object(worker_environment),
    }


def _read_regular_worker_json(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise StudyGateError(f"{label} is missing, unsafe, or not a regular file")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise StudyGateError(f"{label} must contain one JSON object")
    return payload


def execute_runtime_smoke_worker(
    input_path: str | Path,
    result_path: str | Path,
) -> int:
    """Execute one hash-bound model request inside a disposable interpreter."""

    from .perception import (
        InferenceRequest,
        aspect_preserving_letterbox,
        visdrone_generation_policy_sha256,
        visdrone_proposal_decision_context,
    )

    input_file = Path(input_path).expanduser().resolve()
    result_file = Path(result_path).expanduser()
    if result_file.is_symlink() or result_file.exists():
        raise StudyGateError("Runtime-smoke worker result path must be new")
    worker_input = _read_regular_worker_json(
        input_file, label="runtime-smoke worker input"
    )
    if worker_input.get("schema_version") != RUNTIME_SMOKE_WORKER_INPUT_SCHEMA:
        raise StudyGateError("Runtime-smoke worker input schema changed")

    project_root = Path(__file__).resolve().parents[1]
    worker_script = project_root / "semuav" / "runtime_smoke_worker.py"
    python_executable = Path(sys.executable).resolve()
    lock_path = Path(str(worker_input.get("model_lock_path", ""))).expanduser().resolve()
    if (
        str(project_root) not in sys.path
        or source_tree_hash(project_root) != worker_input.get("code_sha256")
        or str(worker_script.resolve()) != worker_input.get("worker_script_path")
        or hash_file(worker_script) != worker_input.get("worker_script_sha256")
        or str(python_executable) != worker_input.get("python_executable")
        or hash_file(python_executable)
        != worker_input.get("python_executable_sha256")
        or hash_file(lock_path) != worker_input.get("model_lock_sha256")
        or hash_object({
            name: os.environ.get(name)
            for name in (*_WORKER_ENVIRONMENT_NAMES, *_WORKER_FORBIDDEN_ENVIRONMENT_NAMES)
        }) != worker_input.get("worker_environment_sha256")
        or {
            name: os.environ.get(name)
            for name in (*_WORKER_ENVIRONMENT_NAMES, *_WORKER_FORBIDDEN_ENVIRONMENT_NAMES)
        } != worker_input.get("worker_environment")
        or any(
            os.environ.get(name) is not None
            for name in _WORKER_REMOVED_LEGACY_ENVIRONMENT_NAMES
        )
    ):
        raise StudyGateError("Runtime-smoke worker provenance or environment changed")

    _, lock = _read_model_lock(lock_path)
    matches = [
        entry for entry in lock.get("models", [])
        if isinstance(entry, Mapping)
        and str(entry.get("model_id", "")) == str(worker_input.get("model_id", ""))
    ]
    if len(matches) != 1 or hash_object(matches[0]) != worker_input.get(
        "model_entry_sha256"
    ):
        raise StudyGateError("Runtime-smoke worker model entry changed")
    entry = matches[0]
    for variable, expected in model_loader_environment(lock_path).items():
        if os.environ.get(variable) != expected:
            raise StudyGateError(
                f"Runtime-smoke worker environment differs for {variable}"
            )

    request_payload = worker_input.get("request")
    if not isinstance(request_payload, Mapping):
        raise StudyGateError("Runtime-smoke worker request is missing")
    image_path = Path(str(request_payload.get("image_path", ""))).expanduser().resolve()
    if image_path.is_symlink() or not image_path.is_file():
        raise StudyGateError("Runtime-smoke worker image is missing or unsafe")
    image_sha256 = hash_file(image_path)
    if (
        str(image_path) != worker_input.get("image_path")
        or image_sha256 != worker_input.get("image_sha256")
    ):
        raise StudyGateError("Runtime-smoke worker image provenance changed")
    request = InferenceRequest(
        observation_id=str(request_payload.get("observation_id", "")),
        image_path=str(image_path),
        mission=str(request_payload.get("mission", "")),
        capture_time_us=int(request_payload.get("capture_time_us")),
        telemetry=dict(request_payload.get("telemetry", {})),
        detector_proposals=tuple(request_payload.get("detector_proposals", ())),
    )
    input_receipt = _runtime_smoke_input_receipt(
        request, image_sha256=image_sha256
    )
    if input_receipt != worker_input.get("input_receipt"):
        raise StudyGateError("Runtime-smoke worker request receipt changed")
    prompt_text = str(worker_input.get("prompt", "")).strip()
    if not prompt_text or hash_object(prompt_text) != worker_input.get("prompt_sha256"):
        raise StudyGateError("Runtime-smoke worker prompt changed")

    with Image.open(image_path) as opened_image:
        source_image = opened_image.convert("RGB")
    _, preprocess = aspect_preserving_letterbox(source_image, (768, 768))
    expected_context_sha256 = hash_object(
        visdrone_proposal_decision_context(
            request,
            source_image.size,
            (768, 768),
            preprocess=preprocess,
        )
    )
    max_new_tokens = int(worker_input.get("max_new_tokens"))
    row, _ = _execute_runtime_smoke_entry(
        entry,
        request,
        image_path=image_path,
        image_sha256=image_sha256,
        input_receipt=input_receipt,
        prompt_text=prompt_text,
        device=str(worker_input.get("device")),
        dtype=str(worker_input.get("dtype")),
        max_new_tokens=max_new_tokens,
        expected_decision_context_sha256=expected_context_sha256,
        expected_generation_policy_sha256=visdrone_generation_policy_sha256(
            max_new_tokens
        ),
        backend_factory=_default_backend_factory,
        cleanup_cuda=False,
    )
    result = {
        "schema_version": RUNTIME_SMOKE_WORKER_RESULT_SCHEMA,
        "worker_input_sha256": hash_file(input_file),
        "model_id": str(entry["model_id"]),
        "model_entry_sha256": hash_object(entry),
        "model_lock_sha256": hash_file(lock_path),
        "code_sha256": source_tree_hash(project_root),
        "worker_script_sha256": hash_file(worker_script),
        "python_executable": str(python_executable),
        "python_executable_sha256": hash_file(python_executable),
        "worker_environment_sha256": hash_object(worker_input["worker_environment"]),
        "row": _json_safe_runtime_smoke_row(row),
    }
    atomic_write_json(result_file, result)
    return 0


def _terminate_worker_process(process: subprocess.Popen[Any]) -> None:
    """Terminate a timed-out worker's whole process group."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait(timeout=5)


def _launch_runtime_smoke_worker(
    worker_input_path: Path,
    worker_result_path: Path,
    stdout_path: Path,
    stderr_path: Path,
    *,
    worker_script: Path,
    environment: Mapping[str, str],
    timeout_seconds: int,
) -> dict[str, Any]:
    """Launch one worker without a shell and return process provenance."""

    command = [
        str(Path(sys.executable).resolve()),
        "-I",
        "-B",
        str(worker_script),
        str(worker_input_path),
        str(worker_result_path),
    ]
    started = time.perf_counter()
    timed_out = False
    with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=str(Path(__file__).resolve().parents[1]),
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            close_fds=True,
            shell=False,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            _terminate_worker_process(process)
            returncode = process.returncode
    return {
        "pid": process.pid,
        "returncode": returncode,
        "signal": -returncode if isinstance(returncode, int) and returncode < 0 else None,
        "timed_out": timed_out,
        "wall_ms": (time.perf_counter() - started) * 1000.0,
        "argv_sha256": hash_object(command),
        "stdout_bytes": stdout_path.stat().st_size,
        "stdout_sha256": hash_file(stdout_path),
        "stderr_bytes": stderr_path.stat().st_size,
        "stderr_sha256": hash_file(stderr_path),
    }


def _validate_runtime_smoke_worker_result(
    payload: Mapping[str, Any],
    *,
    worker_input: Mapping[str, Any],
    worker_input_path: Path,
) -> dict[str, Any]:
    expected = {
        "schema_version": RUNTIME_SMOKE_WORKER_RESULT_SCHEMA,
        "worker_input_sha256": hash_file(worker_input_path),
        "model_id": worker_input["model_id"],
        "model_entry_sha256": worker_input["model_entry_sha256"],
        "model_lock_sha256": worker_input["model_lock_sha256"],
        "code_sha256": worker_input["code_sha256"],
        "worker_script_sha256": worker_input["worker_script_sha256"],
        "python_executable": worker_input["python_executable"],
        "python_executable_sha256": worker_input["python_executable_sha256"],
        "worker_environment_sha256": worker_input["worker_environment_sha256"],
    }
    if any(payload.get(key) != value for key, value in expected.items()):
        raise StudyGateError("Runtime-smoke worker result provenance changed")
    row = payload.get("row")
    if not isinstance(row, dict) or row.get("model_id") != worker_input["model_id"]:
        raise StudyGateError("Runtime-smoke worker result row changed")
    return dict(row)


def _run_runtime_smoke_subprocesses(
    entries: Sequence[Mapping[str, Any]],
    request: Any,
    *,
    output_path: Path,
    lock_path: Path,
    model_lock_sha256: str,
    code_sha256: str,
    image_path: Path,
    image_sha256: str,
    input_receipt: Mapping[str, Any],
    prompt_text: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Execute every locked model in its own fresh CPython/CUDA process."""

    worker_script = Path(__file__).resolve().parent / "runtime_smoke_worker.py"
    if worker_script.is_symlink() or not worker_script.is_file():
        raise StudyGateError("Runtime-smoke worker source is missing or unsafe")
    artifact_root = output_path.parent / (output_path.stem + "_WORKERS")
    if artifact_root.is_symlink() or artifact_root.exists():
        raise StudyGateError("Runtime-smoke worker artifact directory is not pristine")
    artifact_root.mkdir(mode=0o700)
    environment, environment_attestation = _worker_environment()
    rows: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []

    for index, entry in enumerate(entries):
        model_id = str(entry["model_id"])
        stem = f"{index:02d}-{model_id}"
        input_path = artifact_root / f"{stem}.input.json"
        result_path = artifact_root / f"{stem}.result.json"
        stdout_path = artifact_root / f"{stem}.stdout.log"
        stderr_path = artifact_root / f"{stem}.stderr.log"
        worker_input = _worker_request_payload(
            entry,
            request,
            lock_path=lock_path,
            model_lock_sha256=model_lock_sha256,
            code_sha256=code_sha256,
            image_path=image_path,
            image_sha256=image_sha256,
            input_receipt=input_receipt,
            prompt_text=prompt_text,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
            worker_script=worker_script,
            worker_environment=environment_attestation,
        )
        atomic_write_json(input_path, worker_input)
        launch: dict[str, Any]
        try:
            launch = _launch_runtime_smoke_worker(
                input_path,
                result_path,
                stdout_path,
                stderr_path,
                worker_script=worker_script,
                environment=environment,
                timeout_seconds=RUNTIME_SMOKE_WORKER_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            launch = {
                "pid": None,
                "returncode": None,
                "signal": None,
                "timed_out": False,
                "wall_ms": None,
                "argv_sha256": None,
                "stdout_bytes": stdout_path.stat().st_size if stdout_path.is_file() else None,
                "stdout_sha256": hash_file(stdout_path) if stdout_path.is_file() else None,
                "stderr_bytes": stderr_path.stat().st_size if stderr_path.is_file() else None,
                "stderr_sha256": hash_file(stderr_path) if stderr_path.is_file() else None,
                "launch_error": f"{type(exc).__name__}: {exc}",
            }

        row = _empty_runtime_smoke_row(
            entry,
            image_path=image_path,
            image_sha256=image_sha256,
            input_receipt=input_receipt,
        )
        result_valid = False
        result_error: str | None = None
        if result_path.is_file() and not result_path.is_symlink():
            try:
                worker_result = _read_regular_worker_json(
                    result_path, label=f"runtime-smoke result for {model_id}"
                )
                row = _validate_runtime_smoke_worker_result(
                    worker_result,
                    worker_input=worker_input,
                    worker_input_path=input_path,
                )
                result_valid = True
            except Exception as exc:
                result_error = f"{type(exc).__name__}: {exc}"
        else:
            result_error = "Worker did not produce a safe result artifact"
        if launch.get("returncode") != 0 or launch.get("timed_out") or not result_valid:
            row = _empty_runtime_smoke_row(
                entry,
                image_path=image_path,
                image_sha256=image_sha256,
                input_receipt=input_receipt,
            )
            row["error"] = (
                "Isolated runtime-smoke worker failed: "
                f"returncode={launch.get('returncode')}, "
                f"signal={launch.get('signal')}, "
                f"timed_out={launch.get('timed_out')}, "
                f"result_error={result_error or launch.get('launch_error')}"
            )
        row.update({
            "worker_isolation_schema": RUNTIME_SMOKE_ISOLATION_SCHEMA,
            "worker_input_sha256": hash_file(input_path),
            "worker_result_sha256": (
                hash_file(result_path)
                if result_path.is_file() and not result_path.is_symlink()
                else None
            ),
            "worker_result_valid": result_valid,
            "worker_pid": launch.get("pid"),
            "worker_returncode": launch.get("returncode"),
            "worker_signal": launch.get("signal"),
            "worker_timed_out": launch.get("timed_out"),
            "worker_wall_ms": launch.get("wall_ms"),
            "worker_argv_sha256": launch.get("argv_sha256"),
            "worker_stdout_bytes": launch.get("stdout_bytes"),
            "worker_stdout_sha256": launch.get("stdout_sha256"),
            "worker_stderr_bytes": launch.get("stderr_bytes"),
            "worker_stderr_sha256": launch.get("stderr_sha256"),
        })
        rows.append(row)
        artifact_records.append({
            "model_id": model_id,
            "input": {
                "path": str(input_path),
                "bytes": input_path.stat().st_size,
                "sha256": hash_file(input_path),
            },
            "result": (
                {
                    "path": str(result_path),
                    "bytes": result_path.stat().st_size,
                    "sha256": hash_file(result_path),
                }
                if result_path.is_file() and not result_path.is_symlink()
                else None
            ),
            "stdout": (
                {
                    "path": str(stdout_path),
                    "bytes": stdout_path.stat().st_size,
                    "sha256": hash_file(stdout_path),
                }
                if stdout_path.is_file() else None
            ),
            "stderr": (
                {
                    "path": str(stderr_path),
                    "bytes": stderr_path.stat().st_size,
                    "sha256": hash_file(stderr_path),
                }
                if stderr_path.is_file() else None
            ),
        })

    isolation = {
        "schema_version": RUNTIME_SMOKE_ISOLATION_SCHEMA,
        "strategy": "fresh_exec_per_locked_model",
        "worker_script": str(worker_script),
        "worker_script_sha256": hash_file(worker_script),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_executable_sha256": hash_file(Path(sys.executable).resolve()),
        "timeout_seconds_per_model": RUNTIME_SMOKE_WORKER_TIMEOUT_SECONDS,
        "worker_environment": environment_attestation,
        "worker_environment_sha256": hash_object(environment_attestation),
        "artifact_root": str(artifact_root),
        "artifacts": artifact_records,
    }
    isolation["artifacts_sha256"] = hash_object(artifact_records)
    return rows, isolation




def run_one_image_runtime_smoke(
    lock_path: str | Path,
    request: Any,
    *,
    output_path: str | Path,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_new_tokens: int = VISDRONE_REGISTERED_MAX_NEW_TOKENS,
    prompt: str | None = None,
    backend_factory: BackendFactory | None = None,
    offline_checker: OfflineSnapshotChecker | None = None,
    moondream_tokenizer_lock_path: str | Path | None = None,
    moondream_tokenizer_offline_checker: AuxiliaryArtifactChecker | None = None,
) -> pd.DataFrame:
    """Run every frozen model once and atomically record all outcomes.

    The returned/report table always has exactly one row per locked model.  A
    construction or inference exception becomes ``status='ERROR'`` with its
    exception text; schema-invalid output becomes ``FAIL_SCHEMA``.  Call
    :func:`require_runtime_smoke_pass` before validation or test.
    """

    payload = verify_huggingface_model_lock(
        lock_path, offline_checker=offline_checker
    )
    production_subprocess_path = backend_factory is None and offline_checker is None
    if (
        moondream_tokenizer_lock_path is not None
        or moondream_tokenizer_offline_checker is not None
    ):
        raise StudyGateError(
            "The active v2r13 runtime excludes Moondream and rejects its "
            "historical tokenizer binding"
        )
    for variable, expected in model_loader_environment(lock_path).items():
        observed = os.environ.get(variable)
        if observed is not None and observed != expected:
            raise StudyGateError(
                f"{variable} conflicts with the verified model lock"
            )
        os.environ[variable] = expected
    if prompt is None:
        from .perception import prompt_for_mission

        prompt_text = prompt_for_mission(str(getattr(request, "mission", "")))
    else:
        prompt_text = str(prompt).strip()
    if not prompt_text:
        raise StudyGateError("Runtime smoke prompt must be nonempty")
    image_path = Path(str(getattr(request, "image_path", ""))).expanduser().resolve()
    if not image_path.is_file():
        raise StudyGateError(f"Runtime smoke image is missing: {image_path}")
    image_sha256 = hash_file(image_path)
    input_receipt = _runtime_smoke_input_receipt(
        request,
        image_sha256=image_sha256,
    )
    project_root = Path(__file__).resolve().parents[1]
    code_sha256 = source_tree_hash(project_root)
    model_lock_sha256 = hash_file(Path(lock_path).expanduser().resolve())
    from .perception import (
        VISDRONE_GENERATION_POLICY_VERSION,
        VISDRONE_GENERATION_STRATEGY_BY_MODEL,
        VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
        aspect_preserving_letterbox,
        visdrone_assistant_prefill_for_model,
        visdrone_generation_policy_sha256,
        visdrone_proposal_decision_context,
    )

    with Image.open(image_path) as opened_image:
        source_image = opened_image.convert("RGB")
    source_size = source_image.size
    _, expected_preprocess = aspect_preserving_letterbox(
        source_image, (768, 768)
    )
    expected_decision_context_sha256 = hash_object(
        visdrone_proposal_decision_context(
            request,
            source_size,
            (768, 768),
            preprocess=expected_preprocess,
        )
    )
    expected_generation_policy_sha256 = visdrone_generation_policy_sha256(
        max_new_tokens
    )

    factory = backend_factory or _default_backend_factory
    execution_isolation: dict[str, Any] | None = None
    if production_subprocess_path:
        rows, execution_isolation = _run_runtime_smoke_subprocesses(
            payload["models"],
            request,
            output_path=Path(output_path).expanduser().resolve(),
            lock_path=Path(lock_path).expanduser().resolve(),
            model_lock_sha256=model_lock_sha256,
            code_sha256=code_sha256,
            image_path=image_path,
            image_sha256=image_sha256,
            input_receipt=input_receipt,
            prompt_text=prompt_text,
            device=device,
            dtype=dtype,
            max_new_tokens=max_new_tokens,
        )
    else:
        # Dependency-injected tests cannot be serialized into an isolated
        # interpreter. This path is unreachable from the locked notebook.
        rows = []
    unrecoverable_cleanup_error: str | None = None
    for entry in (() if production_subprocess_path else payload["models"]):
        started = time.perf_counter()
        row: dict[str, Any] = {
            "model_id": str(entry["model_id"]),
            "repo": str(entry["repo"]),
            "revision": str(entry["resolved_revision"]),
            "role": str(entry["role"]),
            "status": "ERROR",
            "schema_valid": False,
            "error": None,
            "latency_ms": None,
            "record_model_id": None,
            "record_revision": None,
            "record_observation_id": None,
            "record_prompt_hash": None,
            "raw_text": None,
            "raw_text_sha256": None,
            "parsed_packet_sha256": None,
            "input_tokens": None,
            "output_tokens": None,
            "decision_contract_version": None,
            "decision_core_valid": None,
            "decision_core_json": None,
            "decision_context_sha256": None,
            "decision_proposal_count": None,
            "decision_composer_policy_sha256": None,
            "terminated_by_eos": None,
            "reached_token_cap": None,
            "generation_policy_version": None,
            "generation_policy_sha256": None,
            "generation_strategy": None,
            "generation_max_new_tokens": None,
            "assistant_prefill_sha256": None,
            "assistant_prefill_characters": None,
            "generated_continuation_sha256": None,
            "image_path": str(image_path),
            "image_sha256": image_sha256,
            **input_receipt,
        }
        if unrecoverable_cleanup_error is not None:
            row["error"] = (
                "Not run after unrecoverable runtime cleanup failure: "
                + unrecoverable_cleanup_error
            )
            row["smoke_wall_ms"] = (time.perf_counter() - started) * 1000.0
            rows.append(row)
            continue
        backend = None
        try:
            with _local_only_models():
                backend = factory(entry, device, dtype)
                record = backend.infer(
                    request,
                    prompt=prompt_text,
                    max_new_tokens=max_new_tokens,
                )
            record_model = str(getattr(record, "model_id", ""))
            record_revision = str(getattr(record, "model_revision", ""))
            record_observation_id = str(getattr(record, "observation_id", ""))
            record_prompt_hash = str(getattr(record, "prompt_hash", ""))
            schema_valid = bool(getattr(record, "schema_valid", False))
            raw_text = getattr(record, "raw_text", None)
            parsed_packet = getattr(record, "parsed", None)
            row.update(
                {
                    "record_model_id": record_model,
                    "record_revision": record_revision,
                    "record_observation_id": record_observation_id,
                    "record_prompt_hash": record_prompt_hash,
                    "schema_valid": schema_valid,
                    "latency_ms": getattr(record, "latency_ms", None),
                    "error": getattr(record, "error", None),
                    "raw_text": raw_text if isinstance(raw_text, str) else None,
                    "raw_text_sha256": (
                        sha256_bytes(raw_text.encode("utf-8"))
                        if isinstance(raw_text, str) else None
                    ),
                    "parsed_packet_sha256": (
                        hash_object(parsed_packet)
                        if isinstance(parsed_packet, Mapping) else None
                    ),
                    "input_tokens": getattr(record, "input_tokens", None),
                    "output_tokens": getattr(record, "output_tokens", None),
                    "decision_contract_version": getattr(
                        record, "decision_contract_version", None
                    ),
                    "decision_core_valid": getattr(record, "decision_core_valid", None),
                    "decision_core_json": getattr(record, "decision_core_json", None),
                    "decision_context_sha256": getattr(
                        record, "decision_context_sha256", None
                    ),
                    "decision_proposal_count": getattr(
                        record, "decision_proposal_count", None
                    ),
                    "decision_composer_policy_sha256": getattr(
                        record, "decision_composer_policy_sha256", None
                    ),
                    "terminated_by_eos": getattr(record, "terminated_by_eos", None),
                    "reached_token_cap": getattr(record, "reached_token_cap", None),
                    "generation_policy_version": getattr(
                        record, "generation_policy_version", None
                    ),
                    "generation_policy_sha256": getattr(
                        record, "generation_policy_sha256", None
                    ),
                    "generation_strategy": getattr(
                        record, "generation_strategy", None
                    ),
                    "generation_max_new_tokens": getattr(
                        record, "generation_max_new_tokens", None
                    ),
                    "assistant_prefill_sha256": getattr(
                        record, "assistant_prefill_sha256", None
                    ),
                    "assistant_prefill_characters": getattr(
                        record, "assistant_prefill_characters", None
                    ),
                    "generated_continuation_sha256": getattr(
                        record, "generated_continuation_sha256", None
                    ),
                }
            )
            expected_strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL.get(
                row["model_id"]
            )
            expected_prefill = visdrone_assistant_prefill_for_model(
                row["model_id"]
            )
            generated_continuation = None
            if isinstance(raw_text, str) and raw_text.startswith(expected_prefill):
                generated_continuation = raw_text[len(expected_prefill):]
            provenance_valid = (
                record_model == row["model_id"]
                and record_revision == row["revision"]
                and record_observation_id == str(getattr(request, "observation_id"))
                and record_prompt_hash == hash_object(prompt_text)
                and isinstance(raw_text, str)
                and bool(raw_text)
                and row["decision_contract_version"]
                == VISDRONE_RUNTIME_SMOKE_INPUT_CONTRACT
                and row["decision_proposal_count"]
                == input_receipt["detector_proposal_count"]
                and row["decision_context_sha256"]
                == expected_decision_context_sha256
                and row["decision_composer_policy_sha256"]
                == VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                and row["generation_policy_version"]
                == VISDRONE_GENERATION_POLICY_VERSION
                and row["generation_policy_sha256"]
                == expected_generation_policy_sha256
                and row["generation_strategy"] == expected_strategy
                and row["generation_max_new_tokens"] == max_new_tokens
                and row["assistant_prefill_sha256"]
                == sha256_bytes(expected_prefill.encode("utf-8"))
                and row["assistant_prefill_characters"] == len(expected_prefill)
                and isinstance(generated_continuation, str)
                and row["generated_continuation_sha256"]
                == sha256_bytes(generated_continuation.encode("utf-8"))
            )
            if not provenance_valid:
                row["status"] = "FAIL_PROVENANCE"
                row["error"] = (
                    "Inference record identity, prompt, decision contract, or composer "
                    "lineage differs from the bound runtime-smoke input"
                )
            elif (
                not schema_valid
                or row["decision_core_valid"] is not True
                or row["parsed_packet_sha256"] is None
            ):
                row["status"] = "FAIL_SCHEMA"
                row["error"] = row["error"] or "Inference output failed the frozen schema"
            else:
                row["status"] = "PASS"
        except Exception as exc:  # every failure must be represented in the report
            row["status"] = "ERROR"
            row["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            row["smoke_wall_ms"] = (time.perf_counter() - started) * 1000.0
            try:  # release one model before loading the next; absence of torch is harmless here
                del backend
                gc.collect()
                import torch  # type: ignore

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            except Exception as exc:
                # A failed CUDA cleanup can leave the process unsafe for another
                # model.  Preserve any primary inference error, record this
                # cleanup failure, and emit explicit not-run rows for the rest
                # of the locked set so the fail-closed report remains complete.
                cleanup_error = f"{type(exc).__name__}: {exc}"
                primary_error = row.get("error")
                row["status"] = "ERROR"
                row["error"] = (
                    f"{primary_error}; runtime cleanup failed: {cleanup_error}"
                    if primary_error
                    else f"Runtime cleanup failed: {cleanup_error}"
                )
                unrecoverable_cleanup_error = cleanup_error
            rows.append(row)

    frame = pd.DataFrame.from_records(rows)
    if len(frame) != len(payload["models"]) or frame["model_id"].duplicated().any():
        raise StudyGateError("Runtime smoke report omitted or duplicated a frozen model")
    json_rows: list[dict[str, Any]] = []
    # Serialize from the original row mappings, not ``frame.to_dict``.  The
    # latter may coerce nullable integer token columns to floating point (for
    # example ``3560.0``), which loses the registered JSON integer type even
    # though the originating inference record was exact.
    for raw_row in rows:
        json_rows.append(_json_safe_runtime_smoke_row(raw_row))
    report = {
        "schema_version": RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
        "report_type": "one_real_image_model_runtime_smoke",
        "created_utc": _utc_now(),
        "model_lock": str(Path(lock_path).expanduser().resolve()),
        "model_lock_sha256": model_lock_sha256,
        "model_set_sha256": payload["model_set_sha256"],
        "code_sha256": code_sha256,
        "composer_policy_sha256": VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
        "generation_policy_version": VISDRONE_GENERATION_POLICY_VERSION,
        "generation_policy_sha256": expected_generation_policy_sha256,
        "expected_decision_context_sha256": expected_decision_context_sha256,
        "image_path": str(image_path),
        "image_sha256": image_sha256,
        "prompt_sha256": hash_object(prompt_text),
        **input_receipt,
        "execution_isolation": execution_isolation or {
            "schema_version": "semuav.runtime-smoke-inprocess-test.v1",
            "strategy": "dependency_injected_inprocess_test_only",
        },
        "all_passed": bool(frame["status"].eq("PASS").all()),
        "models": json_rows,
    }
    atomic_write_json(output_path, report)
    return frame


def require_runtime_smoke_pass(report: pd.DataFrame | str | Path) -> None:
    """Gate later phases on an exact fresh v2r13 3/3 PASS report."""

    payload: Mapping[str, Any] | None = None
    if isinstance(report, (str, Path)):
        try:
            payload = json.loads(Path(report).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StudyGateError(f"Cannot read runtime smoke report: {report}") from exc
        rows = payload.get("models", []) if isinstance(payload, Mapping) else []
        frame = pd.DataFrame.from_records(rows)
    else:
        frame = report.copy()
    required = set(OFFICIAL_COMPACT_MODEL_REPOS) | {OFFICIAL_UPPER_BOUND_ID}
    if "model_id" not in frame or "status" not in frame:
        raise StudyGateError("Runtime smoke report lacks model_id/status columns")
    if (
        len(frame) != len(OFFICIAL_RUNTIME_MODEL_IDS)
        or set(frame["model_id"].astype(str)) != required
        or tuple(frame["model_id"].astype(str)) != OFFICIAL_RUNTIME_MODEL_IDS
        or frame["model_id"].duplicated().any()
    ):
        raise StudyGateError(
            "Runtime smoke report is not the exact registered v2r13 3-model set"
        )
    failures = frame.loc[~frame["status"].eq("PASS"), ["model_id", "status", "error"]]
    if not failures.empty:
        raise StudyGateError(
            "Runtime smoke is not an exact 3/3 PASS; do not begin validation/test:\n"
            + failures.to_string(index=False)
        )
    if payload is not None:
        if payload.get("all_passed") is not True:
            raise StudyGateError(
                "Runtime smoke payload does not declare exact 3/3 all_passed=true"
            )
        _require_runtime_smoke_subprocess_provenance(payload)


def _contains_historical_moondream_key(value: Any) -> bool:
    """Return whether active evidence recursively names the removed model."""

    if isinstance(value, Mapping):
        return any(
            "moondream" in str(key).casefold()
            or _contains_historical_moondream_key(item)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_historical_moondream_key(item) for item in value)
    return False


def _require_runtime_smoke_subprocess_provenance(
    payload: Mapping[str, Any],
) -> None:
    """Verify the immutable fresh-process evidence of a passing smoke."""

    if _contains_historical_moondream_key(payload):
        raise StudyGateError(
            "Runtime-smoke active evidence contains historical Moondream keys"
        )
    isolation = payload.get("execution_isolation")
    rows = payload.get("models")
    if (
        payload.get("schema_version") != RUNTIME_SMOKE_REPORT_SCHEMA_VERSION
        or not isinstance(isolation, Mapping)
        or isolation.get("schema_version") != RUNTIME_SMOKE_ISOLATION_SCHEMA
        or isolation.get("strategy") != "fresh_exec_per_locked_model"
        or not isinstance(rows, list)
    ):
        raise StudyGateError(
            "Passing runtime smoke lacks the registered subprocess-isolation contract"
        )
    worker_environment = isolation.get("worker_environment")
    if not isinstance(worker_environment, Mapping):
        raise StudyGateError(
            "Runtime-smoke worker environment attestation is missing"
        )
    expected_environment_names = {
        *_WORKER_ENVIRONMENT_NAMES,
        *_WORKER_FORBIDDEN_ENVIRONMENT_NAMES,
    }
    if (
        set(worker_environment) != expected_environment_names
        or any(
            worker_environment.get(name) is not None
            for name in _WORKER_FORBIDDEN_ENVIRONMENT_NAMES
        )
    ):
        raise StudyGateError(
            "Runtime-smoke worker environment attestation has an invalid boundary"
        )
    artifacts = isolation.get("artifacts")
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != len(rows)
        or tuple(
            str(item.get("model_id", "")) if isinstance(item, Mapping) else ""
            for item in artifacts
        ) != OFFICIAL_RUNTIME_MODEL_IDS
        or hash_object(artifacts) != isolation.get("artifacts_sha256")
    ):
        raise StudyGateError("Runtime-smoke worker artifact manifest changed")
    worker_script = Path(str(isolation.get("worker_script", ""))).expanduser()
    python_executable = Path(
        str(isolation.get("python_executable", ""))
    ).expanduser()
    if (
        worker_script.is_symlink()
        or not worker_script.is_file()
        or hash_file(worker_script) != isolation.get("worker_script_sha256")
        or python_executable.is_symlink()
        or not python_executable.is_file()
        or hash_file(python_executable)
        != isolation.get("python_executable_sha256")
        or hash_object(isolation.get("worker_environment"))
        != isolation.get("worker_environment_sha256")
    ):
        raise StudyGateError("Runtime-smoke worker executable provenance changed")

    artifact_by_model = {
        str(item.get("model_id", "")): item
        for item in artifacts if isinstance(item, Mapping)
    }
    if len(artifact_by_model) != len(rows):
        raise StudyGateError("Runtime-smoke worker artifact model set changed")
    for row in rows:
        if not isinstance(row, Mapping):
            raise StudyGateError("Runtime-smoke report contains a non-object row")
        model_id = str(row.get("model_id", ""))
        record = artifact_by_model.get(model_id)
        if (
            not isinstance(record, Mapping)
            or row.get("worker_isolation_schema")
            != RUNTIME_SMOKE_ISOLATION_SCHEMA
            or row.get("worker_result_valid") is not True
            or row.get("worker_returncode") != 0
            or row.get("worker_signal") is not None
            or row.get("worker_timed_out") is not False
            or any(
                key.startswith("worker_moondream_")
                or key in _MOONDREAM_RUNTIME_PROVENANCE_FIELDS
                for key in row
            )
        ):
            raise StudyGateError(
                f"Runtime-smoke worker attestation is invalid for {model_id}"
            )
        artifact_paths: dict[str, Path] = {}
        for name in ("input", "result", "stdout", "stderr"):
            file_record = record.get(name)
            if not isinstance(file_record, Mapping):
                raise StudyGateError(
                    f"Runtime-smoke worker {name} artifact is missing for {model_id}"
                )
            path = Path(str(file_record.get("path", ""))).expanduser()
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size != file_record.get("bytes")
                or hash_file(path) != file_record.get("sha256")
            ):
                raise StudyGateError(
                    f"Runtime-smoke worker {name} artifact changed for {model_id}"
                )
            artifact_paths[name] = path
        worker_input = _read_regular_worker_json(
            artifact_paths["input"],
            label=f"runtime-smoke worker input for {model_id}",
        )
        worker_result = _read_regular_worker_json(
            artifact_paths["result"],
            label=f"runtime-smoke worker result for {model_id}",
        )
        if (
            _contains_historical_moondream_key(worker_input)
            or _contains_historical_moondream_key(worker_result)
            or worker_input.get("schema_version")
            != RUNTIME_SMOKE_WORKER_INPUT_SCHEMA
            or worker_input.get("model_id") != model_id
            or worker_input.get("worker_environment") != worker_environment
        ):
            raise StudyGateError(
                f"Runtime-smoke worker evidence boundary changed for {model_id}"
            )
        verified_worker_row = _validate_runtime_smoke_worker_result(
            worker_result,
            worker_input=worker_input,
            worker_input_path=artifact_paths["input"],
        )
        if any(row.get(key) != value for key, value in verified_worker_row.items()):
            raise StudyGateError(
                f"Runtime-smoke report row differs from worker result for {model_id}"
            )
        if (
            row.get("worker_input_sha256") != record["input"]["sha256"]
            or row.get("worker_result_sha256") != record["result"]["sha256"]
            or row.get("worker_stdout_sha256") != record["stdout"]["sha256"]
            or row.get("worker_stdout_bytes") != record["stdout"]["bytes"]
            or row.get("worker_stderr_sha256") != record["stderr"]["sha256"]
            or row.get("worker_stderr_bytes") != record["stderr"]["bytes"]
        ):
            raise StudyGateError(
                f"Runtime-smoke row/artifact binding changed for {model_id}"
            )




__all__ = [
    "MODEL_LOCK_SCHEMA_VERSION",
    "MODEL_LOCK_TYPE",
    "MOONDREAM_TOKENIZER_REPO",
    "MOONDREAM_TOKENIZER_REVISION",
    "MOONDREAM_TOKENIZER_AUDITED_REPO_HEAD",
    "MOONDREAM_TOKENIZER_FILENAME",
    "MOONDREAM_TOKENIZER_BYTES",
    "MOONDREAM_TOKENIZER_SHA256",
    "MOONDREAM_TOKENIZER_LOCK_SCHEMA",
    "MOONDREAM_TOKENIZER_LOCK_TYPE",
    "RUNTIME_SMOKE_REPORT_SCHEMA_VERSION",
    "RUNTIME_SMOKE_ISOLATION_SCHEMA",
    "RUNTIME_SMOKE_WORKER_INPUT_SCHEMA",
    "RUNTIME_SMOKE_WORKER_RESULT_SCHEMA",
    "OFFICIAL_COMPACT_MODEL_REPOS",
    "OFFICIAL_MODEL_BACKENDS",
    "OFFICIAL_RUNTIME_MODEL_IDS",
    "OFFICIAL_UPPER_BOUND_ID",
    "OFFICIAL_UPPER_BOUND_REPO",
    "HISTORICAL_MOONDREAM_MODEL_ID",
    "HISTORICAL_MOONDREAM_MODEL_REPO",
    "HISTORICAL_V2R5_RUNTIME_MODEL_IDS",
    "V2R6_MODEL_LOCK_DERIVATION_SCHEMA",
    "VISDRONE_DET_TRAIN_COMPONENT",
    "VISDRONE_DET_PUBLISHER_VALIDATION_COMPONENT",
    "VISDRONE_DET_TEST_COMPONENTS",
    "VISDRONE_DET_EXCLUDED_COMPONENTS",
    "DETECTOR_TRAIN_PARTITION",
    "DETECTOR_INTERNAL_VALIDATION_PARTITION",
    "DETECTOR_SPLIT_VERSION",
    "DETECTOR_SOURCE_AUDIT_SCHEMA",
    "DETECTOR_STUDY_MANIFEST",
    "DETECTOR_LEAKAGE_AUDIT",
    "DETECTOR_ALLOWED_STUDY_SPLIT",
    "DETECTOR_FORBIDDEN_STUDY_SPLITS",
    "DETECTOR_TRAINING_PLAN_SCHEMA",
    "DETECTOR_TRAINING_RECEIPT_SCHEMA",
    "VISDRONE_DET_TEST_COMPONENTS",
    "VISDRONE_YOLO_NAMES",
    "VISDRONE_RUNTIME_SMOKE_INPUT_CONTRACT",
    "detector_training_command_plan",
    "bind_moondream_tokenizer_lock",
    "derive_v2r6_three_model_lock",
    "execute_runtime_smoke_worker",
    "freeze_moondream_tokenizer_lock",
    "moondream_tokenizer_runtime_identity",
    "freeze_huggingface_models",
    "model_loader_environment",
    "registered_moondream_tokenizer_artifact",
    "registered_model_load_reference",
    "prepare_ultralytics_visdrone_det",
    "require_runtime_smoke_pass",
    "run_one_image_runtime_smoke",
    "validate_registered_model_configuration",
    "verify_detector_checkpoint",
    "verify_detector_training_receipt",
    "verify_huggingface_model_lock",
    "verify_historical_v2r5_huggingface_model_lock",
    "verify_moondream_tokenizer_lock",
    "verify_bound_moondream_tokenizer_artifact",
    "verify_ultralytics_visdrone_det",
    "write_detector_training_receipt",
]

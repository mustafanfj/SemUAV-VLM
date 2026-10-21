"""Publisher-grounded dataset acquisition for the SemUAV-VLM study.

Downloads are deliberately opt-in.  This module never stores credentials and
never treats a successful download as scientific readiness: licensing review,
mission annotations, group-safe splits, and provenance gates remain separate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import quote

import pandas as pd
import yaml


class AcquisitionError(RuntimeError):
    """A dataset could not be acquired or normalized safely."""


REGISTRY_PATH = Path(__file__).resolve().parents[1] / "configs" / "datasets.yaml"
SUPPORTED_DATASETS = ("visdrone", "uavid", "uavreason", "flame3")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FIRE_EVENTS = ("hanna", "hundred", "shoetank", "sycan", "willamette", "wildbill", "wild_bill")


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(*parts: Any, length: int = 32) -> str:
    return hashlib.sha256("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:length]


def _atomic_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def load_dataset_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise AcquisitionError(f"Unsupported dataset registry: {registry_path}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, Mapping) or set(datasets) != set(SUPPORTED_DATASETS):
        raise AcquisitionError("Dataset registry must contain exactly the four proposal datasets")
    return dict(payload)


def dataset_source_table(path: str | Path = REGISTRY_PATH) -> pd.DataFrame:
    rows = []
    for name, item in load_dataset_registry(path)["datasets"].items():
        rows.append({
            "dataset": name,
            "title": item["title"],
            "study_role": item["role"],
            "provider": item["provider"],
            "official_url": item["landing_page"],
            "license_or_terms": item["license"],
            "estimated_download_gb": float(item.get("estimated_download_gb", 0.0)),
            "confirmatory_note": item.get("confirmation_note", ""),
        })
    return pd.DataFrame(rows)


def _inventory(root: Path) -> tuple[int, int]:
    if not root.is_dir():
        return 0, 0
    files = 0
    size = 0
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            files += 1
            size += path.stat().st_size
    return files, size


def _dataset_checks(name: str, root: Path) -> tuple[bool, str]:
    if not root.is_dir():
        return False, "destination missing"
    if name == "visdrone":
        components = load_dataset_registry()["datasets"]["visdrone"]["components"]
        missing = [
            str(item["name"])
            for item in components
            if not _visdrone_component_tree_present(root, item)
        ]
        return (
            not missing,
            "normalized images and source annotations for all official components"
            if not missing else
            f"normalized component layout incomplete: {missing}",
        )
    if name == "uavid":
        images = any(path.is_file() and path.parent.name.lower() == "images" for path in root.rglob("*"))
        labels = any(path.is_file() and path.parent.name.lower() in {"labels", "masks", "trainid"} for path in root.rglob("*"))
        return images and labels, "labeled image sequences" if images and labels else "labeled sequences incomplete"
    if name == "uavreason":
        manifest = root / "manifest.jsonl"
        return manifest.is_file() and manifest.stat().st_size > 0, "normalized single-frame manifest" if manifest.is_file() else "run manifest preparation"
    if name == "flame3":
        manifest = root / "manifest.csv"
        try:
            paired_rows = len(pd.read_csv(manifest, nrows=1)) if manifest.is_file() else 0
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
            paired_rows = 0
        return (
            paired_rows > 0,
            "paired-source index" if paired_rows > 0 else "no eligible paired RGB/radiometric rows; inspect unpaired_rgb_manifest.csv",
        )
    raise KeyError(name)


def dataset_status(data_root: str | Path, path: str | Path = REGISTRY_PATH) -> pd.DataFrame:
    data_root = Path(data_root).expanduser().resolve()
    rows = []
    registry = load_dataset_registry(path)["datasets"]
    for name in SUPPORTED_DATASETS:
        item = registry[name]
        destination = data_root / str(item["destination"])
        files, size = _inventory(destination)
        ready, detail = _dataset_checks(name, destination)
        receipt = destination / "SOURCE_RECEIPT.json"
        hash_inventory = destination / "SOURCE_FILES_SHA256.jsonl"
        receipt_scope = ""
        resolved_source = ""
        if receipt.is_file():
            try:
                receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
                source_payload = receipt_payload.get("source", {})
                receipt_scope = str(source_payload.get("scope", "publisher_release"))
                resolved_source = str(
                    source_payload.get(
                        "annotations_revision",
                        source_payload.get("version", source_payload.get("doi", "")),
                    )
                )
            except (OSError, json.JSONDecodeError, AttributeError):
                receipt_scope = "INVALID_RECEIPT"
        rows.append({
            "dataset": name,
            "destination": str(destination),
            "present": destination.is_dir(),
            "ingest_ready": bool(ready),
            "files": files,
            "size_gb": size / 1_000_000_000,
            "source_receipt": receipt.is_file(),
            "source_hash_inventory": hash_inventory.is_file(),
            "receipt_scope": receipt_scope,
            "resolved_source": resolved_source,
            "status_detail": detail,
            "official_url": item["landing_page"],
        })
    return pd.DataFrame(rows)


def disk_preflight(data_root: str | Path, selected: Sequence[str]) -> dict[str, float]:
    data_root = Path(data_root).expanduser().resolve()
    probe = data_root if data_root.exists() else next((parent for parent in data_root.parents if parent.exists()), Path("/"))
    usage = shutil.disk_usage(probe)
    registry = load_dataset_registry()["datasets"]
    estimated = sum(float(registry[name].get("estimated_download_gb", 0.0)) for name in selected)
    # Archives and extracted files coexist until integrity receipts are written.
    recommended = max(20.0, estimated * 2.1)
    return {
        "estimated_download_gb": estimated,
        "recommended_free_gb": recommended,
        "available_free_gb": usage.free / 1_000_000_000,
    }


def _validate_selection(selected: Iterable[str]) -> tuple[str, ...]:
    names = tuple(dict.fromkeys(str(item).strip().lower() for item in selected if str(item).strip()))
    unknown = sorted(set(names) - set(SUPPORTED_DATASETS))
    if unknown:
        raise AcquisitionError(f"Unknown dataset(s): {unknown}")
    if not names:
        raise AcquisitionError("Select at least one dataset")
    return names


def _safe_zip_members(archive: zipfile.ZipFile, destination: Path) -> list[zipfile.ZipInfo]:
    destination = destination.resolve()
    accepted = []
    for member in archive.infolist():
        pure = PurePosixPath(member.filename)
        if pure.is_absolute() or ".." in pure.parts:
            raise AcquisitionError(f"Unsafe archive member: {member.filename!r}")
        mode = member.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise AcquisitionError(f"Archive symlink is not allowed: {member.filename!r}")
        target = (destination / Path(*pure.parts)).resolve()
        if target != destination and destination not in target.parents:
            raise AcquisitionError(f"Archive member escapes destination: {member.filename!r}")
        accepted.append(member)
    return accepted


def safe_extract_zip(archive_path: str | Path, destination: str | Path) -> None:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_zip_members(archive, destination)
        archive.extractall(destination, members=members)


def _download_http(
    url: str,
    destination: Path,
    *,
    timeout: int = 120,
    method: str = "GET",
    headers: Mapping[str, str] | None = None,
    json_payload: Mapping[str, Any] | None = None,
) -> Path:
    try:
        import requests
    except ImportError as exc:
        raise AcquisitionError("Install requests before HTTP dataset acquisition") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    existing = partial.stat().st_size if partial.is_file() else 0
    request_headers = dict(headers or {})
    if existing and method.upper() == "GET":
        request_headers["Range"] = f"bytes={existing}-"
    with requests.request(
        method.upper(), url, headers=request_headers, json=json_payload,
        stream=True, timeout=timeout, allow_redirects=True,
    ) as response:
        if response.status_code == 416 and partial.is_file():
            os.replace(partial, destination)
            return destination
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" in content_type or "application/json" in content_type:
            preview = response.raw.read(256)
            raise AcquisitionError(f"Download returned {content_type}, not an archive: {preview!r}")
        append = existing > 0 and method.upper() == "GET" and response.status_code == 206
        if existing and not append:
            partial.unlink(missing_ok=True)
        with partial.open("ab" if append else "wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
    os.replace(partial, destination)
    return destination


def _write_file_hash_manifest(destination: Path) -> dict[str, Any]:
    """Hash the acquired source tree without loading its file list into memory."""

    manifest = destination / "SOURCE_FILES_SHA256.jsonl"
    temporary = manifest.with_suffix(".jsonl.tmp")
    count = 0
    total = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for path in sorted(destination.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name in {
                "SOURCE_RECEIPT.json", manifest.name, temporary.name,
            }:
                continue
            size = path.stat().st_size
            record = {
                "path": path.relative_to(destination).as_posix(),
                "bytes": size,
                "sha256": _sha256(path),
            }
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
            total += size
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, manifest)
    return {
        "path": str(manifest),
        "files": count,
        "bytes": total,
        "sha256": _sha256(manifest),
    }


def _write_receipt(destination: Path, dataset: str, source: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]) -> Path:
    hash_manifest = _write_file_hash_manifest(destination)
    # The receipt envelope describes source files plus the hash inventory, but
    # never counts a previous receipt that may exist during an atomic refresh.
    files = int(hash_manifest["files"]) + 1
    size = int(hash_manifest["bytes"]) + Path(str(hash_manifest["path"])).stat().st_size
    receipt = {
        "schema_version": "semuav.dataset-source.v1",
        "dataset": dataset,
        "resolved_utc": _utc_now(),
        "source": dict(source),
        "artifacts": list(artifacts),
        "content_hash_manifest": hash_manifest,
        "inventory": {"files": files, "bytes": size},
        "scientific_readiness": False,
        "scientific_readiness_note": "Source acquisition is not a substitute for licensing, annotation, split, and provenance gates.",
    }
    return _atomic_json(destination / "SOURCE_RECEIPT.json", receipt)


def _refresh_prepared_receipt(
    destination: Path,
    dataset: str,
    registry_spec: Mapping[str, Any],
    preparation_report: Mapping[str, Any],
) -> tuple[Path, str]:
    """Create/refresh local provenance without inventing remote-source claims."""

    receipt_path = destination / "SOURCE_RECEIPT.json"
    if receipt_path.is_file():
        try:
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AcquisitionError(
                f"Existing source receipt is invalid; restore it or reacquire {dataset}: "
                f"{receipt_path} ({exc})"
            ) from exc
        if (
            not isinstance(existing, Mapping)
            or existing.get("schema_version") != "semuav.dataset-source.v1"
            or str(existing.get("dataset", "")).strip().lower() != dataset
            or not isinstance(existing.get("source"), Mapping)
        ):
            raise AcquisitionError(
                f"Existing source receipt has incompatible provenance metadata: {receipt_path}"
            )
        source = dict(existing["source"])
        artifacts = [
            dict(item)
            for item in existing.get("artifacts", [])
            if isinstance(item, Mapping) and "preparation_refresh" not in item
        ]
        receipt_status = "refreshed_existing_source_provenance"
    else:
        source = {
            "landing_page": registry_spec["landing_page"],
            "provider": registry_spec["provider"],
            "license_or_terms": registry_spec["license"],
            "confirmation_note": registry_spec.get("confirmation_note", ""),
            "scope": "user_staged_publisher_layout_source_not_remotely_verified",
            "prepared_without_network_download": True,
        }
        artifacts = []
        receipt_status = "created_user_staged_source_provenance"
    artifacts.append({
        "preparation_refresh": dict(preparation_report),
        "source_claim_note": (
            "Preparation hashed the local source tree but did not independently "
            "authenticate its remote origin."
        ),
    })
    return _write_receipt(destination, dataset, source, artifacts), receipt_status


def _selected_visdrone_components(spec: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    components = list(spec["components"])
    requested = os.environ.get("SEMUAV_VISDRONE_COMPONENTS", "").strip()
    selected = set(item.strip() for item in requested.split(",") if item.strip())
    if not selected:
        return components
    unknown = selected - {str(item["name"]) for item in components}
    if unknown:
        raise AcquisitionError(f"Unknown VisDrone component(s): {sorted(unknown)}")
    return [item for item in components if str(item["name"]) in selected]


def _visdrone_component_layout(item: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return the canonical component directory and required payload roots."""

    component = str(item.get("name", ""))
    archive_name = str(item.get("archive", ""))
    pure_archive = PurePosixPath(archive_name)
    if (
        not component
        or pure_archive.name != archive_name
        or pure_archive.suffix.lower() != ".zip"
    ):
        raise AcquisitionError(f"Invalid VisDrone component registry entry: {item!r}")
    directory = pure_archive.name[:-4]
    if component.startswith("det_"):
        media_root = "images"
    elif component.startswith("mot_"):
        media_root = "sequences"
    else:
        raise AcquisitionError(f"Unknown VisDrone component task: {component!r}")
    return directory, media_root, "annotations"


def _visdrone_tree_files(root: Path) -> dict[str, Path]:
    """Return a canonical regular-file tree, rejecting links and special files."""

    if root.is_symlink() or not root.is_dir():
        raise AcquisitionError(f"VisDrone component is not a regular directory: {root}")
    files: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise AcquisitionError(f"VisDrone component contains a symbolic link: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AcquisitionError(f"VisDrone component contains a special file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative in files:  # pragma: no cover - impossible on a normal filesystem
            raise AcquisitionError(f"Duplicate VisDrone component path: {relative}")
        files[relative] = path
    return files


def _visdrone_component_tree_present(
    destination: Path,
    item: Mapping[str, Any],
) -> bool:
    """Check the normalized publisher layout without opening image/label content."""

    directory, media_root, annotation_root = _visdrone_component_layout(item)
    try:
        files = _visdrone_tree_files(destination / directory)
    except AcquisitionError:
        return False
    media_prefix = f"{media_root}/"
    annotation_prefix = f"{annotation_root}/"
    has_media = any(
        relative.startswith(media_prefix)
        and PurePosixPath(relative).suffix.lower() in IMAGE_SUFFIXES
        for relative in files
    )
    has_annotations = any(
        relative.startswith(annotation_prefix)
        and PurePosixPath(relative).suffix.lower() == ".txt"
        for relative in files
    )
    return has_media and has_annotations


def _visdrone_archive_plan(
    archive_path: Path,
    item: Mapping[str, Any],
) -> tuple[str, dict[str, int]]:
    """Map wrapped or rootless publisher members into one component directory."""

    directory, media_root, annotation_root = _visdrone_component_layout(item)
    with zipfile.ZipFile(archive_path) as archive:
        members = _safe_zip_members(archive, Path("/") / "visdrone-layout-probe")
        file_members = [member for member in members if not member.is_dir()]
    if not file_members:
        raise AcquisitionError(f"VisDrone archive contains no files: {archive_path}")

    raw_paths = [PurePosixPath(member.filename) for member in file_members]
    wrapped_flags = {path.parts[0] == directory for path in raw_paths if path.parts}
    if len(wrapped_flags) != 1:
        raise AcquisitionError(
            f"VisDrone archive mixes wrapped and rootless layouts: {archive_path}"
        )
    wrapped = wrapped_flags.pop()
    layout = "wrapped" if wrapped else "rootless"
    planned: dict[str, int] = {}
    for member, raw in zip(file_members, raw_paths):
        parts = raw.parts[1:] if wrapped else raw.parts
        if not parts:
            raise AcquisitionError(f"Empty VisDrone component member: {member.filename!r}")
        relative = PurePosixPath(*parts).as_posix()
        if relative in planned:
            raise AcquisitionError(
                f"Duplicate normalized VisDrone archive member: {relative!r}"
            )
        planned[relative] = int(member.file_size)

    media_prefix = f"{media_root}/"
    annotation_prefix = f"{annotation_root}/"
    if not any(
        relative.startswith(media_prefix)
        and PurePosixPath(relative).suffix.lower() in IMAGE_SUFFIXES
        for relative in planned
    ):
        raise AcquisitionError(
            f"VisDrone archive lacks {media_root} media for {item['name']}"
        )
    if not any(
        relative.startswith(annotation_prefix)
        and PurePosixPath(relative).suffix.lower() == ".txt"
        for relative in planned
    ):
        raise AcquisitionError(
            f"VisDrone archive lacks source annotations for {item['name']}"
        )
    return layout, planned


def _assert_visdrone_tree_matches(
    root: Path,
    planned: Mapping[str, int],
    *,
    context: str,
) -> dict[str, Path]:
    files = _visdrone_tree_files(root)
    if set(files) != set(planned):
        missing = sorted(set(planned) - set(files))[:5]
        extra = sorted(set(files) - set(planned))[:5]
        raise AcquisitionError(
            f"{context} file set differs from its archive (missing={missing}, extra={extra})"
        )
    wrong_size = [
        relative
        for relative, path in files.items()
        if path.stat().st_size != int(planned[relative])
    ]
    if wrong_size:
        raise AcquisitionError(
            f"{context} file size differs from its archive: {wrong_size[:5]}"
        )
    return files


def _extract_visdrone_component(
    archive_path: Path,
    destination: Path,
    item: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Safely normalize one archive and atomically publish its component tree."""

    directory, _, _ = _visdrone_component_layout(item)
    layout, planned = _visdrone_archive_plan(archive_path, item)
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.{directory}.", dir=destination.parent,
    ) as temporary:
        temporary_root = Path(temporary)
        extraction_root = temporary_root / "extracted"
        safe_extract_zip(archive_path, extraction_root)
        payload = extraction_root / directory if layout == "wrapped" else extraction_root
        staged_files = _assert_visdrone_tree_matches(
            payload, planned, context=f"Staged VisDrone component {item['name']}",
        )

        # Older acquisition code extracted rootless archives directly into the
        # dataset root. Migrate that layout only when it is an exact byte-for-byte
        # copy of this archive; otherwise fail without touching user data.
        legacy_roots: list[Path] = []
        if layout == "rootless":
            top_levels = sorted({PurePosixPath(relative).parts[0] for relative in planned})
            present = [destination / name for name in top_levels if (destination / name).exists()]
            if present:
                if len(present) != len(top_levels):
                    raise AcquisitionError(
                        f"Incomplete legacy rootless VisDrone layout for {item['name']}"
                    )
                for name in top_levels:
                    live_root = destination / name
                    expected_subset = {
                        PurePosixPath(relative).relative_to(name).as_posix(): size
                        for relative, size in planned.items()
                        if PurePosixPath(relative).parts[0] == name
                    }
                    live_files = _assert_visdrone_tree_matches(
                        live_root,
                        expected_subset,
                        context=f"Legacy rootless VisDrone path {live_root}",
                    )
                    for relative, live_path in live_files.items():
                        staged_path = staged_files[f"{name}/{relative}"]
                        if _sha256(live_path) != _sha256(staged_path):
                            raise AcquisitionError(
                                f"Legacy rootless VisDrone content differs: {live_path}"
                            )
                    legacy_roots.append(live_root)

        target = destination / directory
        previous = temporary_root / "previous-component"
        legacy_backup = temporary_root / "legacy-rootless"
        legacy_backup.mkdir()
        moved_legacy: list[tuple[Path, Path]] = []
        moved_previous = False
        try:
            if target.exists() or target.is_symlink():
                if target.is_symlink() or not target.is_dir():
                    raise AcquisitionError(
                        f"Refusing to replace non-directory VisDrone target: {target}"
                    )
                os.replace(target, previous)
                moved_previous = True
            for live_root in legacy_roots:
                backup = legacy_backup / live_root.name
                os.replace(live_root, backup)
                moved_legacy.append((live_root, backup))
            os.replace(payload, target)
        except Exception:
            for live_root, backup in reversed(moved_legacy):
                if backup.exists() and not live_root.exists():
                    os.replace(backup, live_root)
            if moved_previous and previous.exists() and not target.exists():
                os.replace(previous, target)
            raise

    if not _visdrone_component_tree_present(destination, item):
        raise AcquisitionError(f"Normalized VisDrone component is incomplete: {item['name']}")
    return {
        "archive_layout": layout,
        "normalized_directory": directory,
        "extracted_files": len(planned),
        "extracted_bytes": sum(planned.values()),
    }


def _visdrone_inventory_roots(destination: Path) -> dict[str, set[str]] | None:
    manifest = destination / "SOURCE_FILES_SHA256.jsonl"
    roots: dict[str, set[str]] = {}
    try:
        with manifest.open("r", encoding="utf-8") as handle:
            for raw in handle:
                record = json.loads(raw)
                value = record.get("path") if isinstance(record, Mapping) else None
                if not isinstance(value, str) or not value:
                    return None
                relative = PurePosixPath(value)
                if relative.is_absolute() or relative.as_posix() != value or ".." in relative.parts:
                    return None
                roots.setdefault(relative.parts[0], set()).add(value)
    except (OSError, UnicodeError, json.JSONDecodeError, IndexError):
        return None
    return roots


def _visdrone_receipt_covers(destination: Path, components: Sequence[Mapping[str, Any]]) -> bool:
    receipt_path = destination / "SOURCE_RECEIPT.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version") != "semuav.dataset-source.v1"
        or receipt.get("dataset") != "visdrone"
    ):
        return False
    source = receipt.get("source")
    if (
        not isinstance(source, Mapping)
        or source.get("scope") != "publisher_acquisition_official_release_components"
    ):
        return False
    artifacts = receipt.get("artifacts", ())
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        return False
    observed = {
        (str(item.get("component", "")), str(item.get("google_drive_id", "")))
        for item in artifacts
        if isinstance(item, Mapping) and "component" in item
    }
    required = {
        (str(item["name"]), str(item["google_drive_id"]))
        for item in components
    }
    if not required.issubset(observed):
        return False

    inventory_path = destination / "SOURCE_FILES_SHA256.jsonl"
    content_manifest = receipt.get("content_hash_manifest")
    if not isinstance(content_manifest, Mapping):
        return False
    try:
        recorded_manifest = Path(str(content_manifest["path"])).expanduser()
        if not recorded_manifest.is_absolute():
            recorded_manifest = destination / recorded_manifest
        if recorded_manifest.resolve(strict=False) != inventory_path.resolve(strict=False):
            return False
        if str(content_manifest.get("sha256", "")).lower() != _sha256(inventory_path):
            return False
    except (KeyError, OSError, ValueError):
        return False

    roots = _visdrone_inventory_roots(destination)
    if roots is None:
        return False
    official_directories = {
        _visdrone_component_layout(item)[0]
        for item in load_dataset_registry()["datasets"]["visdrone"]["components"]
    }
    if set(roots) - official_directories:
        return False
    for item in components:
        directory, _, _ = _visdrone_component_layout(item)
        if not roots.get(directory) or not _visdrone_component_tree_present(destination, item):
            return False
        actual = {
            f"{directory}/{relative}"
            for relative in _visdrone_tree_files(destination / directory)
        }
        if actual != roots[directory]:
            return False
    return True


def _preserved_visdrone_artifacts(
    destination: Path,
    spec: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Preserve only previously receipted components with normalized coverage."""

    receipt_path = destination / "SOURCE_RECEIPT.json"
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if not isinstance(receipt, Mapping):
        return {}
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        return {}
    roots = _visdrone_inventory_roots(destination)
    if roots is None:
        return {}
    registry = {str(item["name"]): item for item in spec["components"]}
    preserved: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or "component" not in artifact:
            continue
        component = str(artifact["component"])
        item = registry.get(component)
        if item is None:
            continue
        directory, _, _ = _visdrone_component_layout(item)
        if (
            str(artifact.get("google_drive_id", "")) == str(item["google_drive_id"])
            and Path(str(artifact.get("archive", ""))).name == str(item["archive"])
            and roots.get(directory)
            and _visdrone_component_tree_present(destination, item)
        ):
            preserved[component] = dict(artifact)
    return preserved


def _acquire_visdrone(data_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import gdown
    except ImportError as exc:
        raise AcquisitionError("Install gdown before downloading VisDrone") from exc
    destination = data_root / str(spec["destination"])
    download_dir = data_root / ".downloads" / "visdrone"
    destination.mkdir(parents=True, exist_ok=True)
    artifacts_by_component = _preserved_visdrone_artifacts(destination, spec)
    components = _selected_visdrone_components(spec)
    for item in components:
        archive = download_dir / str(item["archive"])
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.is_file():
            result = gdown.download(id=str(item["google_drive_id"]), output=str(archive), quiet=False, resume=True)
            if not result or not archive.is_file():
                raise AcquisitionError(f"VisDrone download failed: {item['name']}")
        if not zipfile.is_zipfile(archive):
            raise AcquisitionError(f"VisDrone archive is invalid: {archive}")
        extraction = _extract_visdrone_component(archive, destination, item)
        artifacts_by_component[str(item["name"])] = {
            "component": item["name"], "archive": str(archive),
            "google_drive_id": item["google_drive_id"], "sha256": _sha256(archive),
            "bytes": archive.stat().st_size,
            **extraction,
        }
    artifacts = [
        artifacts_by_component[str(item["name"])]
        for item in spec["components"]
        if str(item["name"]) in artifacts_by_component
    ]
    selected_names = {str(item["name"]) for item in components}
    if selected_names - set(artifacts_by_component):  # pragma: no cover - defensive
        raise AcquisitionError("VisDrone acquisition did not cover every requested component")
    receipt = _write_receipt(destination, "visdrone", {
        "landing_page": spec["landing_page"], "provider": spec["provider"],
        "scope": "publisher_acquisition_official_release_components",
        "license_or_terms": spec["license"],
        "confirmation_note": spec.get("confirmation_note", ""),
    }, artifacts)
    if not _visdrone_receipt_covers(destination, components):
        raise AcquisitionError(
            "VisDrone receipt, inventory, and normalized tree do not cover every requested component"
        )
    return {"dataset": "visdrone", "status": "downloaded", "destination": str(destination), "receipt": str(receipt)}


def _acquire_uavid(data_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    dataverse = spec["dataverse"]
    destination = data_root / str(spec["destination"])
    download_dir = data_root / ".downloads" / "uavid"
    archive = download_dir / str(dataverse["archive"])
    persistent_id = str(dataverse["persistent_id"])
    version = str(dataverse["version"])
    url = (
        f"{str(dataverse['server']).rstrip('/')}/api/access/dataset/:persistentId/versions/"
        f"{quote(version, safe='')}?persistentId={quote(persistent_id, safe=':')}"
    )
    manual_archive = os.environ.get("SEMUAV_UAVID_ARCHIVE", "").strip()
    if manual_archive:
        supplied = Path(manual_archive).expanduser().resolve()
        if not supplied.is_file():
            raise AcquisitionError(f"SEMUAV_UAVID_ARCHIVE does not exist: {supplied}")
        archive.parent.mkdir(parents=True, exist_ok=True)
        if supplied != archive.resolve():
            shutil.copy2(supplied, archive)
    if not archive.is_file():
        try:
            token = os.environ.get("DANS_API_TOKEN", "").strip()
            guestbook_path = os.environ.get("SEMUAV_UAVID_GUESTBOOK_JSON", "").strip()
            headers = {"X-Dataverse-key": token} if token else {}
            if guestbook_path:
                guestbook_file = Path(guestbook_path).expanduser().resolve()
                if not guestbook_file.is_file():
                    raise AcquisitionError(
                        f"SEMUAV_UAVID_GUESTBOOK_JSON does not exist: {guestbook_file}"
                    )
                guestbook = json.loads(guestbook_file.read_text(encoding="utf-8"))
                if not isinstance(guestbook, Mapping):
                    raise AcquisitionError("UAVid guestbook JSON must be an object")
                _download_http(
                    url, archive, method="POST", headers=headers,
                    json_payload=guestbook,
                )
            else:
                _download_http(url, archive, headers=headers)
        except Exception as exc:
            raise AcquisitionError(
                "UAVid's official DANS deposit may require its download guestbook. "
                "Supply DANS_API_TOKEN and SEMUAV_UAVID_GUESTBOOK_JSON, or download "
                "version 1.0 from the configured DOI and set "
                "SEMUAV_UAVID_ARCHIVE=/absolute/path/to/the/archive."
            ) from exc
    if not zipfile.is_zipfile(archive):
        raise AcquisitionError(f"UAVid Dataverse bundle is invalid: {archive}")
    safe_extract_zip(archive, destination)
    receipt = _write_receipt(destination, "uavid", {
        "landing_page": spec["landing_page"], "persistent_id": persistent_id,
        "version": version, "download_url": url, "provider": spec["provider"],
        "scope": "publisher_archive_dans_v1_confirmatory",
        "license_or_terms": spec["license"],
        "confirmation_note": spec.get("confirmation_note", ""),
    }, [{"archive": str(archive), "sha256": _sha256(archive), "bytes": archive.stat().st_size}])
    return {"dataset": "uavid", "status": "downloaded", "destination": str(destination), "receipt": str(receipt)}


def _hf_snapshot(
    repo_id: str,
    destination: Path,
    revision: str | None,
    *,
    allow_patterns: Sequence[str] | None = None,
) -> tuple[str, str]:
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as exc:
        raise AcquisitionError("Install huggingface-hub before downloading UAVReason") from exc
    api = HfApi()
    info = api.dataset_info(repo_id, revision=revision or "main")
    resolved = str(info.sha)
    if not re.fullmatch(r"[0-9a-f]{40}", resolved):
        raise AcquisitionError(f"Hugging Face did not resolve an immutable commit for {repo_id}")
    snapshot_download(
        repo_id=repo_id, repo_type="dataset", revision=resolved,
        local_dir=str(destination),
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )
    return repo_id, resolved


def _assistant_answer(record: Mapping[str, Any]) -> tuple[str, str]:
    conversations = record.get("conversations", [])
    if not isinstance(conversations, Sequence) or isinstance(conversations, (str, bytes)):
        return "", ""
    question = ""
    answer = ""
    for turn in conversations:
        if not isinstance(turn, Mapping):
            continue
        speaker = str(turn.get("from", turn.get("role", ""))).lower()
        value = str(turn.get("value", turn.get("content", ""))).strip()
        if speaker in {"human", "user"} and not question:
            question = value
        if speaker in {"gpt", "assistant"}:
            answer = value
    return question, answer


def _image_list(record: Mapping[str, Any]) -> list[str]:
    value = record.get("image", record.get("images", record.get("image_path", [])))
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return [str(item) for item in value if str(item).strip()]
    return []


def _resolve_uavreason_media(root: Path, image: str) -> Path:
    """Resolve publisher JSONL paths against the staged UAVScenes subset."""

    raw = Path(str(image).replace("\\", "/"))
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend((root / raw, root / "UAVScenes" / raw))
    lowered = [part.lower() for part in raw.parts]
    if "uavscenes" in lowered:
        index = lowered.index("uavscenes")
        candidates.append(root / Path(*raw.parts[index:]))
    if "interval5_cam_lidar" in lowered:
        index = lowered.index("interval5_cam_lidar")
        candidates.append(root / "UAVScenes" / Path(*raw.parts[index:]))
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    # Return the canonical staged location so the error report remains useful.
    return (root / "UAVScenes" / raw.name).resolve()


def prepare_uavreason_manifests(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    annotations_root = root / "annotations"
    sources = sorted(path for path in annotations_root.rglob("*.jsonl") if "for_unified_edit" not in path.parts)
    if not sources:
        raise AcquisitionError(f"No UAVReason VQA/caption JSONL files found under {annotations_root}")
    single_path = root / "manifest.jsonl"
    temporal_path = root / "temporal_manifest.jsonl"
    missing_media = 0
    single_count = temporal_count = invalid_count = 0
    single_tmp = single_path.with_suffix(".jsonl.tmp")
    temporal_tmp = temporal_path.with_suffix(".jsonl.tmp")
    with single_tmp.open("w", encoding="utf-8") as single_handle, temporal_tmp.open("w", encoding="utf-8") as temporal_handle:
        for source in sources:
            relative_source = source.relative_to(root).as_posix()
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_count += 1
                        continue
                    if not isinstance(record, Mapping):
                        invalid_count += 1
                        continue
                    images = _image_list(record)
                    question, answer = _assistant_answer(record)
                    if not images or not answer:
                        invalid_count += 1
                        continue
                    meta = record.get("meta", {}) if isinstance(record.get("meta", {}), Mapping) else {}
                    scene = str(meta.get("scene", "")).strip()
                    if not scene:
                        image_parts = [part for value in images for part in Path(value).parts]
                        scene = next(
                            (
                                part for part in image_parts
                                if part.lower().startswith("interval5_")
                                and part.lower() not in {"interval5_cam", "interval5_cam_lidar"}
                                and "." not in part
                            ),
                            "",
                        )
                    resolved_images = [_resolve_uavreason_media(root, image) for image in images]
                    missing_media += sum(not path.is_file() for path in resolved_images)
                    sample_id = _stable_id(relative_source, line_number, *images)
                    output = {
                        "sample_id": sample_id,
                        "source_uid": f"{relative_source}:{line_number}",
                        "image_path": str(resolved_images[0]),
                        "sequence_id": scene,
                        "location_id": scene,
                        "split_group": f"uavreason:{scene or sample_id}",
                        "task": str(meta.get("task", "uavreason")),
                        "question": question,
                        "answer": answer,
                        "label": answer,
                        "has_ground_truth": True,
                        "critical_event": False,
                        "media_paths_json": json.dumps([str(path) for path in resolved_images], separators=(",", ":")),
                        "source_annotation_file": relative_source,
                        "source_annotation_line": line_number,
                        "source_meta_json": json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
                    }
                    target = single_handle if len(resolved_images) == 1 else temporal_handle
                    target.write(json.dumps(output, sort_keys=True, ensure_ascii=False) + "\n")
                    if len(resolved_images) == 1:
                        single_count += 1
                    else:
                        temporal_count += 1
    if missing_media:
        single_tmp.unlink(missing_ok=True)
        temporal_tmp.unlink(missing_ok=True)
        raise AcquisitionError(
            f"UAVReason normalization found {missing_media} missing UAVScenes RGB reference(s); "
            "the annotation repository alone is not a usable image dataset"
        )
    os.replace(single_tmp, single_path)
    os.replace(temporal_tmp, temporal_path)
    report = {
        "single_frame_rows": single_count,
        "temporal_rows_excluded_from_single_image_pipeline": temporal_count,
        "invalid_rows": invalid_count,
        "missing_media_references": missing_media,
        "single_manifest": str(single_path),
        "temporal_manifest": str(temporal_path),
    }
    _atomic_json(root / "MANIFEST_PREPARATION.json", report)
    return report


def _acquire_uavreason(data_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    destination = data_root / str(spec["destination"])
    annotations_dir = destination / "annotations"
    images_dir = destination / "UAVScenes"
    hf = spec["huggingface"]
    annotations_revision = os.environ.get("SEMUAV_UAVREASON_REVISION") or str(hf["annotations_revision"])
    images_revision = os.environ.get("SEMUAV_UAVSCENES_REVISION") or str(hf["images_revision"])
    annotations_repo, annotations_sha = _hf_snapshot(
        str(hf["annotations_repo"]), annotations_dir, annotations_revision,
        allow_patterns=hf.get("annotations_allow_patterns"),
    )
    images_repo, images_sha = _hf_snapshot(
        str(hf["images_repo"]), images_dir, images_revision,
        allow_patterns=hf.get("images_allow_patterns"),
    )
    images_archive = images_dir / str(hf["images_archive"])
    extracted_marker = images_dir / ".interval5_cam_lidar_extracted"
    if not extracted_marker.is_file():
        if not zipfile.is_zipfile(images_archive):
            raise AcquisitionError(f"Pinned UAVScenes archive is missing or invalid: {images_archive}")
        safe_extract_zip(images_archive, images_dir)
        extracted_marker.write_text(f"{images_sha}\n", encoding="utf-8")
    manifest_report = prepare_uavreason_manifests(destination)
    receipt = _write_receipt(destination, "uavreason", {
        "landing_page": spec["landing_page"],
        "scope": "publisher_acquisition_pinned_huggingface_snapshots",
        "annotations_repo": annotations_repo, "annotations_revision": annotations_sha,
        "images_repo": images_repo, "images_revision": images_sha,
        "provider": spec["provider"], "license_or_terms": spec["license"],
        "confirmation_note": spec.get("confirmation_note", ""),
    }, [{"manifest_report": manifest_report}])
    return {"dataset": "uavreason", "status": "downloaded", "destination": str(destination), "receipt": str(receipt)}


def _flame_event(path: Path, root: Path) -> str:
    relative = path.relative_to(root).as_posix().lower()
    for event in FIRE_EVENTS:
        if event in relative:
            return event.replace("_", "-")
    parts = path.relative_to(root).parts
    return parts[0] if len(parts) > 1 else path.parent.name


def _normalized_sensor_stem(path: Path) -> str:
    text = path.stem.lower()
    for token in ("radiometric", "thermal", "visual", "visible", "colour", "color", "rgb", "ir", "lwir"):
        text = re.sub(rf"(?:^|[_\- ]){token}(?:$|[_\- ])", "_", text)
    return re.sub(r"[^a-z0-9]+", "", text)


def prepare_flame3_manifest(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    images = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise AcquisitionError(f"No FLAME 3 imagery found under {root}")
    thermal = [path for path in images if path.suffix.lower() in {".tif", ".tiff"} or any(token in path.as_posix().lower() for token in ("thermal", "radiometric", "lwir"))]
    rgb = [path for path in images if path not in thermal and not any(token in path.as_posix().lower() for token in ("mask", "label", "annotation"))]
    by_event_stem: dict[tuple[str, str], list[Path]] = {}
    for path in thermal:
        by_event_stem.setdefault((_flame_event(path, root), _normalized_sensor_stem(path)), []).append(path)
    paired_rows: list[dict[str, Any]] = []
    unpaired_rows: list[dict[str, Any]] = []
    for image in rgb:
        event = _flame_event(image, root)
        matches = by_event_stem.get((event, _normalized_sensor_stem(image)), [])
        pair = matches[0] if len(matches) == 1 else None
        relative = image.relative_to(root).as_posix()
        row = {
            "sample_id": _stable_id("flame3", relative),
            "source_uid": relative,
            "image_path": str(image.resolve()),
            "annotation_path": str(pair.resolve()) if pair else "",
            "has_ground_truth": bool(pair),
            "event_id": event,
            "sequence_id": image.parent.name,
            "split_group": f"flame3:{event}",
            "critical_event": False,
            "label": "",
            "paired_thermal_path": str(pair.resolve()) if pair else "",
            "source_role": "paired_rgb_radiometric" if pair else "rgb_unpaired",
            "requires_mission_annotation": True,
        }
        if pair is not None:
            paired_rows.append(row)
        else:
            row["exclusion_reason"] = (
                "no_unique_radiometric_pair; excluded from scientific manifest without "
                "fabricating source ground truth"
            )
            unpaired_rows.append(row)
    manifest = root / "manifest.csv"
    columns = (
        "sample_id", "source_uid", "image_path", "annotation_path",
        "has_ground_truth", "event_id", "sequence_id", "split_group",
        "critical_event", "label", "paired_thermal_path", "source_role",
        "requires_mission_annotation",
    )
    frame = pd.DataFrame.from_records(paired_rows, columns=columns)
    temporary = manifest.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(temporary, manifest)
    unpaired_manifest = root / "unpaired_rgb_manifest.csv"
    unpaired_columns = (*columns, "exclusion_reason")
    unpaired_frame = pd.DataFrame.from_records(unpaired_rows, columns=unpaired_columns)
    unpaired_temporary = unpaired_manifest.with_suffix(".csv.tmp")
    unpaired_frame.to_csv(unpaired_temporary, index=False, quoting=csv.QUOTE_MINIMAL)
    os.replace(unpaired_temporary, unpaired_manifest)
    report = {
        "rgb_rows": len(rgb),
        "paired_rgb_thermal_rows": len(frame),
        "scientific_manifest_rows": len(frame),
        "unpaired_rgb_rows": len(unpaired_frame),
        "excluded_unpaired_rgb_rows": len(unpaired_frame),
        "thermal_files": len(thermal),
        "event_groups": sorted({str(row["event_id"]) for row in paired_rows + unpaired_rows}),
        "mission_labels_inferred": False,
        "unpaired_ground_truth_fabricated": False,
        "manifest": str(manifest),
        "unpaired_manifest": str(unpaired_manifest),
        "status": (
            "paired_rows_ready_unpaired_rgb_routed"
            if len(frame) and len(unpaired_frame)
            else "paired_rows_ready"
            if len(frame)
            else "no_unique_rgb_radiometric_pairs"
        ),
    }
    _atomic_json(root / "MANIFEST_PREPARATION.json", report)
    return report


def _acquire_flame3(data_root: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    destination = data_root / str(spec["destination"])
    destination.mkdir(parents=True, exist_ok=True)
    full_archive_value = os.environ.get("SEMUAV_FLAME3_FULL_ARCHIVE", "").strip()
    if full_archive_value:
        supplied = Path(full_archive_value).expanduser().resolve()
        if not supplied.is_file() or not zipfile.is_zipfile(supplied):
            raise AcquisitionError(
                "SEMUAV_FLAME3_FULL_ARCHIVE must point to the complete IEEE DataPort ZIP"
            )
        download_dir = data_root / ".downloads" / "flame3"
        download_dir.mkdir(parents=True, exist_ok=True)
        archive = download_dir / supplied.name
        if supplied != archive.resolve():
            shutil.copy2(supplied, archive)
        safe_extract_zip(archive, destination / "full_ieee_dataport")
        manifest_report = prepare_flame3_manifest(destination)
        receipt = _write_receipt(destination, "flame3", {
            "landing_page": spec["landing_page"], "doi": spec.get("doi"),
            "scope": "complete_ieee_dataport_full_confirmatory_user_staged",
            "provider": spec["provider"], "license_or_terms": spec["license"],
            "confirmation_note": spec.get("confirmation_note", ""),
        }, [{
            "archive": str(archive), "sha256": _sha256(archive),
            "bytes": archive.stat().st_size, "manifest_report": manifest_report,
        }])
        return {
            "dataset": "flame3", "status": "downloaded_full_confirmatory_scope",
            "destination": str(destination), "receipt": str(receipt),
        }
    try:
        import kagglehub
    except ImportError as exc:
        raise AcquisitionError("Install kagglehub and configure Kaggle credentials before FLAME 3 acquisition") from exc
    artifacts = []
    for item in spec["kaggle"]:
        component_dir = destination / str(item["name"])
        resolved = kagglehub.dataset_download(str(item["handle"]), output_dir=str(component_dir))
        artifacts.append({"component": item["name"], "handle": item["handle"], "resolved_path": str(resolved)})
    manifest_report = prepare_flame3_manifest(destination)
    artifacts.append({"manifest_report": manifest_report})
    receipt = _write_receipt(destination, "flame3", {
        "landing_page": spec["landing_page"], "doi": spec.get("doi"),
        "scope": "official_kaggle_pilot_subsets_not_full_confirmatory",
        "provider": spec["provider"], "license_or_terms": spec["license"],
        "confirmation_note": spec.get("confirmation_note", ""),
    }, artifacts)
    return {"dataset": "flame3", "status": "downloaded_pilot_scope", "destination": str(destination), "receipt": str(receipt)}


def prepare_dataset_manifests(data_root: str | Path, selected: Sequence[str] = SUPPORTED_DATASETS) -> pd.DataFrame:
    names = _validate_selection(selected)
    data_root = Path(data_root).expanduser().resolve()
    registry = load_dataset_registry()["datasets"]
    rows = []
    for name in names:
        try:
            destination = data_root / str(registry[name]["destination"])
            if name == "uavreason":
                report = prepare_uavreason_manifests(destination)
            elif name == "flame3":
                report = prepare_flame3_manifest(destination)
            else:
                report = {
                    "layout": "native_publisher_layout",
                    "normalization_performed": False,
                }
            ready, detail = _dataset_checks(name, destination)
            if not ready:
                raise AcquisitionError(f"Prepared dataset is not ingest-ready: {detail}")
            receipt, receipt_status = _refresh_prepared_receipt(
                destination, name, registry[name], report,
            )
            rows.append({
                "dataset": name,
                "status": "prepared_and_hashed",
                "destination": str(destination),
                "receipt": str(receipt),
                "receipt_status": receipt_status,
                "detail": json.dumps(report, sort_keys=True),
            })
        except Exception as exc:
            rows.append({"dataset": name, "status": "blocked", "detail": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def acquire_datasets(
    data_root: str | Path,
    selected: Sequence[str] = SUPPORTED_DATASETS,
    *,
    accept_terms: bool = False,
) -> pd.DataFrame:
    names = _validate_selection(selected)
    if not accept_terms:
        raise AcquisitionError(
            "Dataset terms were not acknowledged. Read every official landing page, "
            "then set SEMUAV_ACCEPT_DATASET_TERMS=1 or pass accept_terms=True."
        )
    data_root = Path(data_root).expanduser().resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    disk = disk_preflight(data_root, names)
    if disk["available_free_gb"] < disk["recommended_free_gb"]:
        raise AcquisitionError(
            f"Insufficient free space: {disk['available_free_gb']:.1f} GB available; "
            f"{disk['recommended_free_gb']:.1f} GB recommended while archives coexist."
        )
    registry = load_dataset_registry()["datasets"]
    handlers = {
        "visdrone": _acquire_visdrone,
        "uavid": _acquire_uavid,
        "uavreason": _acquire_uavreason,
        "flame3": _acquire_flame3,
    }
    rows = []
    for name in names:
        destination = data_root / str(registry[name]["destination"])
        ready, detail = _dataset_checks(name, destination)
        upgrade_flame_scope = name == "flame3" and bool(
            os.environ.get("SEMUAV_FLAME3_FULL_ARCHIVE", "").strip()
        )
        receipt_covers_request = (
            _visdrone_receipt_covers(
                destination, _selected_visdrone_components(registry[name]),
            )
            if name == "visdrone"
            else True
        )
        ready_for_request = receipt_covers_request if name == "visdrone" else ready
        if (
            ready_for_request
            and (destination / "SOURCE_RECEIPT.json").is_file()
            and (destination / "SOURCE_FILES_SHA256.jsonl").is_file()
            and receipt_covers_request
            and not upgrade_flame_scope
        ):
            rows.append({
                "dataset": name, "status": "already_present", "destination": str(destination),
                "receipt": str(destination / "SOURCE_RECEIPT.json"), "detail": detail,
            })
            continue
        result = handlers[name](data_root, registry[name])
        rows.append(result)
    return pd.DataFrame(rows)


def _parse_names(text: str) -> tuple[str, ...]:
    return _validate_selection(text.split(","))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--action", choices=("status", "download", "prepare"), default="status")
    parser.add_argument("--datasets", default=",".join(SUPPORTED_DATASETS))
    parser.add_argument("--accept-terms", action="store_true")
    args = parser.parse_args(argv)
    selected = _parse_names(args.datasets)
    if args.action == "status":
        print(dataset_status(args.data_root).to_string(index=False))
    elif args.action == "prepare":
        prepared = prepare_dataset_manifests(args.data_root, selected)
        print(prepared.to_string(index=False))
        if prepared["status"].eq("blocked").any():
            blocked = prepared.loc[prepared["status"].eq("blocked"), "dataset"].tolist()
            raise AcquisitionError(f"Dataset preparation blocked for: {blocked}")
    else:
        print(acquire_datasets(args.data_root, selected, accept_terms=args.accept_terms).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

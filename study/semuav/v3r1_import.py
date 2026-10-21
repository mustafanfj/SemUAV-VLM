"""Hash-bound reuse of quiescent v3r1 validation computations in v3r2.

v3r2 changes the controller, the calibration-gate rule and four receipted
bookkeeping/decoding corrections.  It does not change perception inference,
detector measurement, native receiver or feature measurement, the Experiment 2
frontier, or (after the receipted v3r1 corrections) the Experiment 3 static
transport.  Those validation computations are reused here instead of being
measured again for several days.  Experiment 3 shards are reused only when
their inputs reproduce byte-identically and they show no pre-fix L0-L2
decode rows; otherwise they are recomputed.

Reuse is exact and fail-safe:

* The deployment script inventories the quiescent v3r1 run (every reusable
  artifact with its byte count and SHA-256) and pins that inventory's digest in
  the v3r2 deployment receipt before any v3r2 validation starts.
* A stage is reused only when the v3r2 call presents byte-identical key
  material, i.e. the same inputs (images, model lock, detector proposals,
  candidate rows, manifest timing, link grid, seeds).  The predecessor cache
  key is recomputed from that key material with the predecessor's recorded
  config/code identity; on any mismatch the stage is computed normally.
* Experiment 4 partitions (``04_*``) are never reused: the v3r2 controller
  revision changes them.  Calibration, model selection, candidate profiles,
  the policy specification, Experiments 4-7 and every gate are recomputed.
* Untouched-test material is never read.  Import is enabled only in the
  validation phase and refuses a run that has a test-access ledger.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .registry import RunContext, StudyGateError, hash_object


IMPORT_SCHEMA = "semuav.visdrone-v3r1-validation-import.v1"
IMPORT_STATUS = "QUIESCENT_V3R1_VALIDATION_SOURCE_FROZEN"
ENABLE_VARIABLE = "SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT"
RECEIPT_VARIABLE = "SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT"
RECEIPT_SHA256_VARIABLE = "SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT_SHA256"
SUCCESSOR_RUN_NAME = "semuav_visdrone_v3r2"
PREDECESSOR_RUN_NAME = "semuav_visdrone_v3r1"
PREDECESSOR_CODE_SHA256 = (
    "7bb899d11e95b5780278fe38ca653dad34e71137e2f296c7a72056586711db98"
)
PREDECESSOR_CONFIG_PREFIX = "448d523d65"
PREDECESSOR_RUN_ID = (
    f"{PREDECESSOR_RUN_NAME}-full-{PREDECESSOR_CONFIG_PREFIX}-"
    f"{PREDECESSOR_CODE_SHA256[:10]}"
)
# Stage families whose producers are unchanged in v3r2.  ``04_`` (dynamic
# controller trials) is deliberately absent.
REUSABLE_STAGE_PATTERN = re.compile(
    r"^(01_vlm_[A-Za-z0-9_.-]+_shard_[0-9]{5}|"
    r"validation(_temporal)?_02_[A-Za-z0-9_]+_shard_[0-9]{5}|"
    r"03_static_transport_seed_[0-9]+_shard_[0-9]{5}|validation_detector)$"
)
REUSABLE_RESULT_PATTERN = re.compile(
    r"^validation[a-z0-9_]*_(native_receiver_cache|registered_feature_cache)\.csv\.gz$"
)
# Configuration fields that identify the release rather than the computation.
SHARD_SUFFIX = re.compile(r"_shard_[0-9]{5}$")
IDENTITY_ONLY_CONFIG_FIELDS = ("paths", "study_id")
IDENTITY_ONLY_RUNTIME_FIELDS = ("pre_test_calibration_gate_amendment_record",)

_STATE: dict[str, Any] = {"receipt": None, "imported": [], "computed": [], "results": []}


def enabled() -> bool:
    return os.environ.get(ENABLE_VARIABLE) == "1"


def _read_regular(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise StudyGateError(f"Missing {label}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise StudyGateError(f"Unsafe {label}: {path}")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise StudyGateError(f"{label} changed while opening: {path}")
        payload = handle.read()
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ):
        raise StudyGateError(f"{label} changed while reading: {path}")
    return payload


def _digest(value: Any, label: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise StudyGateError(f"{label} is not one lowercase SHA-256")
    return text


def _relative(value: Any) -> PurePosixPath:
    text = str(value)
    relative = PurePosixPath(text)
    if (
        relative.is_absolute()
        or relative.as_posix() != text
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise StudyGateError(f"Unsafe v3r1 import path: {text}")
    return relative


def _normalized_config(config: Mapping[str, Any]) -> dict[str, Any]:
    plain = json.loads(json.dumps(config, sort_keys=True, default=str))
    paths = plain.get("paths")
    project_root = str(paths.get("project_root", "")) if isinstance(paths, dict) else ""
    if project_root:
        # Paths resolved beneath the release directory differ only by the
        # release identity; replace that prefix before comparing.
        def strip_root(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: strip_root(item) for key, item in value.items()}
            if isinstance(value, list):
                return [strip_root(item) for item in value]
            if isinstance(value, str) and value.startswith(project_root):
                return "<PROJECT_ROOT>" + value[len(project_root):]
            return value
        plain = strip_root(plain)
    for field in IDENTITY_ONLY_CONFIG_FIELDS:
        plain.pop(field, None)
    runtime = plain.get("runtime")
    if isinstance(runtime, dict):
        for field in IDENTITY_ONLY_RUNTIME_FIELDS:
            runtime.pop(field, None)
    return plain


def _config_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                differences.append(path)
            else:
                differences.extend(_config_differences(left[key], right[key], path))
        return differences
    return [] if left == right else [prefix or "<root>"]


def load_receipt(ctx: RunContext) -> Mapping[str, Any]:
    """Authenticate the deployment-pinned v3r1 inventory for this v3r2 run."""
    cached = _STATE.get("receipt")
    if cached is not None:
        return cached
    if not enabled():
        raise StudyGateError("v3r1 validation import is not enabled for this phase")
    if not ctx.is_scientific or ctx.run_name != SUCCESSOR_RUN_NAME:
        raise StudyGateError("v3r1 validation import is restricted to the v3r2 full run")
    if (ctx.run_root / "test_access_ledger.json").exists():
        raise StudyGateError("v3r1 validation import refuses a run that opened the test set")
    receipt_path = Path(os.environ.get(RECEIPT_VARIABLE, "")).absolute()
    expected = _digest(os.environ.get(RECEIPT_SHA256_VARIABLE, ""), "v3r1 import receipt digest")
    payload = _read_regular(receipt_path, "v3r1 import receipt")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise StudyGateError("v3r1 import receipt differs from its deployment-pinned digest")
    receipt = json.loads(payload.decode("utf-8"))
    if (
        receipt.get("schema_version") != IMPORT_SCHEMA
        or receipt.get("status") != IMPORT_STATUS
        or receipt.get("predecessor_run_name") != PREDECESSOR_RUN_NAME
        or receipt.get("predecessor_code_sha256") != PREDECESSOR_CODE_SHA256
        or not str(receipt.get("predecessor_config_sha256", "")).startswith(PREDECESSOR_CONFIG_PREFIX)
        or receipt.get("untouched_test_accessed") is not False
        or receipt.get("experiment4_partitions_imported") is not False
    ):
        raise StudyGateError("v3r1 import receipt identity or scope changed")
    run_root = Path(str(receipt.get("predecessor_run_root", ""))).absolute()
    if run_root.name != PREDECESSOR_RUN_ID or run_root.is_symlink() or not run_root.is_dir():
        raise StudyGateError("v3r1 import run root is unsafe or changed")
    if (run_root / "test_access_ledger.json").exists():
        raise StudyGateError("The v3r1 import source opened the untouched test set")
    manifest_record = receipt.get("run_manifest", {})
    manifest_bytes = _read_regular(run_root / "run_manifest.json", "v3r1 run manifest")
    if (
        hashlib.sha256(manifest_bytes).hexdigest() != manifest_record.get("sha256")
        or len(manifest_bytes) != manifest_record.get("bytes")
    ):
        raise StudyGateError("v3r1 run manifest differs from the import receipt")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("run_name") != PREDECESSOR_RUN_NAME
        or manifest.get("profile") != "full"
        or manifest.get("code_hash") != PREDECESSOR_CODE_SHA256
        or manifest.get("config_hash") != receipt.get("predecessor_config_sha256")
    ):
        raise StudyGateError("v3r1 run manifest identity changed")
    if hash_object(manifest.get("config")) != manifest.get("config_hash"):
        raise StudyGateError("v3r1 run manifest config does not match its hash")
    differences = _config_differences(
        _normalized_config(manifest.get("config", {})), _normalized_config(ctx.config),
    )
    if differences:
        raise StudyGateError(
            "v3r2 configuration differs from v3r1 beyond release identity; "
            f"validation reuse is not exact: {differences[:10]}"
        )
    stages: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in receipt.get("stages", []):
        name = str(record.get("name", ""))
        key = _digest(record.get("cache_key"), f"v3r1 stage key {name}")
        relative = _relative(record.get("relative_path"))
        if (
            REUSABLE_STAGE_PATTERN.fullmatch(name) is None
            or relative.parts != ("stages", name, f"{key}.csv.gz")
            or (name, key) in stages
        ):
            raise StudyGateError(f"v3r1 import stage record is outside the reusable scope: {name}")
        _digest(record.get("sha256"), f"v3r1 stage digest {name}")
        stages[(name, key)] = record
    results: dict[str, Mapping[str, Any]] = {}
    for record in receipt.get("results", []):
        relative = _relative(record.get("relative_path"))
        if (
            len(relative.parts) != 2 or relative.parts[0] != "results"
            or REUSABLE_RESULT_PATTERN.fullmatch(relative.parts[1]) is None
        ):
            raise StudyGateError(f"v3r1 import result is outside the reusable scope: {relative}")
        _digest(record.get("sha256"), f"v3r1 result digest {relative}")
        results[relative.parts[1]] = record
    if hash_object({"stages": receipt.get("stages", []), "results": receipt.get("results", [])}) != receipt.get(
        "inventory_sha256"
    ):
        raise StudyGateError("v3r1 import inventory digest changed")
    families: dict[str, list[Mapping[str, Any]]] = {}
    for record in stages.values():
        families.setdefault(SHARD_SUFFIX.sub("", str(record["name"])), []).append(record)
    verified = {
        "receipt": receipt, "receipt_path": str(receipt_path), "receipt_sha256": expected,
        "run_root": run_root, "stages": stages, "results": results, "families": families,
    }
    _STATE["receipt"] = verified
    return verified


def _verified_bytes(run_root: Path, record: Mapping[str, Any], label: str) -> bytes:
    relative = _relative(record["relative_path"])
    payload = _read_regular(run_root.joinpath(*relative.parts), label)
    if len(payload) != record.get("bytes") or hashlib.sha256(payload).hexdigest() != record["sha256"]:
        raise StudyGateError(f"{label} changed since the v3r2 deployment inventory")
    return payload


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _has_pre_cbor_fix_rows(payload: bytes) -> bool:
    """Defensive: an Experiment-3 shard still showing the v3r1 L0-L2 decode defect."""
    import pandas as pd

    frame = pd.read_csv(io.BytesIO(payload), compression="gzip", low_memory=False)
    if "variant" not in frame.columns:
        return False
    level = frame["variant"].astype(str).str.split("__").str[0].isin({"l0_cbor", "l1_cbor", "l2_cbor"})
    if not level.any():
        return False
    if "wire_format" in frame.columns and frame.loc[level, "wire_format"].astype(str).ne("cbor").any():
        return True
    if "apply_status" in frame.columns and frame.loc[level, "apply_status"].astype(str).eq("decode_error").any():
        return True
    return False


def import_stage(ctx: RunContext, name: str, key_material: Any, *, current_key: str) -> bool:
    """Install one exact v3r1 stage under its v3r2 key; ``False`` means compute it."""
    if not enabled() or REUSABLE_STAGE_PATTERN.fullmatch(str(name)) is None:
        return False
    verified = load_receipt(ctx)
    config = verified["receipt"]["predecessor_config_sha256"]

    def predecessor_key(stage_name: str) -> str:
        return hash_object({
            "name": stage_name, "key": key_material, "config": config, "code": PREDECESSOR_CODE_SHA256,
        })

    key = predecessor_key(str(name))
    record = verified["stages"].get((str(name), key))
    if record is None:
        # A sharded stage may carry a different shard index when an earlier
        # stream was added or removed (e.g. a different fixed comparator).
        # The shard's computation does not depend on its index, so an exact
        # key match under any sibling name of the same family is reused.
        for sibling in verified["families"].get(SHARD_SUFFIX.sub("", str(name)), []):
            if sibling["name"] != name and predecessor_key(str(sibling["name"])) == sibling["cache_key"]:
                record, key = sibling, sibling["cache_key"]
                break
    if record is None:
        _STATE["computed"].append(str(name))
        print(f"[v3r1-import] compute {name}: no exact v3r1 match", flush=True)
        return False
    payload = _verified_bytes(verified["run_root"], record, f"v3r1 stage {name}")
    try:
        with gzip.open(io.BytesIO(payload), "rt", newline="") as handle:
            rows = sum(1 for _ in handle) - 1
    except OSError as exc:
        raise StudyGateError(f"v3r1 stage {name} is not readable gzip") from exc
    if record.get("rows") is not None and rows < int(record["rows"]):
        raise StudyGateError(f"v3r1 stage {name} row count changed")
    if str(name).startswith("03_static_transport_") and _has_pre_cbor_fix_rows(payload):
        _STATE["computed"].append(str(name))
        print(f"[v3r1-import] compute {name}: v3r1 rows predate the CBOR wire-format fix", flush=True)
        return False
    destination = ctx.stage_dir(str(name)) / f"{current_key}.csv.gz"
    _atomic_write(destination, payload)
    ctx.set_stage(
        str(name), current_key, "COMPLETE", destination, record.get("rows"),
        message=(
            f"IMPORTED_V3R1 from={record['name']} key={key} sha256={record['sha256']} "
            f"receipt={verified['receipt_sha256']}"
        ),
    )
    _STATE["imported"].append(str(name))
    print(f"[v3r1-import] reused {name}", flush=True)
    return True


def import_result_caches(ctx: RunContext) -> list[str]:
    """Copy exact v3r1 native-receiver/feature caches into this run's results."""
    if not enabled():
        return []
    verified = load_receipt(ctx)
    installed: list[str] = []
    for name, record in sorted(verified["results"].items()):
        payload = _verified_bytes(verified["run_root"], record, f"v3r1 result {name}")
        destination = ctx.run_root / "results" / name
        if destination.exists():
            if _read_regular(destination, f"v3r2 result {name}") != payload:
                raise StudyGateError(
                    f"v3r2 result {name} already exists and differs from the v3r1 cache"
                )
            continue
        _atomic_write(destination, payload)
        installed.append(name)
        print(f"[v3r1-import] reused result cache {name}", flush=True)
    _STATE["results"].extend(installed)
    return installed


def import_report() -> dict[str, Any]:
    """Counts for the notebook's visible reuse audit."""
    receipt = _STATE.get("receipt")
    return {
        "enabled": enabled(),
        "receipt_sha256": receipt["receipt_sha256"] if receipt else None,
        "stages_reused": len(_STATE["imported"]),
        "stages_computed": len(_STATE["computed"]),
        "result_caches_reused": len(_STATE["results"]),
        "computed_examples": sorted(set(_STATE["computed"]))[:20],
    }


__all__ = [
    "ENABLE_VARIABLE", "IMPORT_SCHEMA", "IMPORT_STATUS", "PREDECESSOR_CODE_SHA256",
    "PREDECESSOR_RUN_ID", "RECEIPT_SHA256_VARIABLE", "RECEIPT_VARIABLE",
    "REUSABLE_RESULT_PATTERN", "REUSABLE_STAGE_PATTERN", "enabled", "import_report",
    "import_result_caches", "import_stage", "load_receipt",
]

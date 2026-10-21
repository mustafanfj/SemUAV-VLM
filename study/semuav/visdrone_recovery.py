"""Hash-bound v2r9 Experiment-1 shard recovery for the v3 successor.

Only raw detector/VLM inference frames are eligible.  Derived metrics,
selection, semantic packets, receiver caches, and later experiments are always
recomputed by v3.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import pandas as pd

from .registry import RunContext, StudyGateError, hash_file, hash_object


RECOVERY_SCHEMA = "semuav.visdrone-v2r9-exp1-recovery.v1"
RECOVERY_STATUS = "QUIESCENT_V2R9_EXP1_SOURCE_FROZEN"
PREDECESSOR_RUN_NAME = "semuav_visdrone_v2r9"
PREDECESSOR_RELEASE_ID = "release_20260920_visdrone_paper_v2r9"
PREDECESSOR_CODE_SHA256 = (
    "fbf80e5ec3067ff8ca9c150c71a0ec2a3bcf5cffe9ec3663bb79c973789bdf76"
)
PREDECESSOR_CONFIG_PREFIX = "d799763aef"
PREDECESSOR_MODEL_LOCK_SHA256 = (
    "340dd59ca55a9be491ac806b50f9778a12d2b0b4a447485627aaf422dda05353"
)
EXPECTED_ARTIFACT_COUNT = 132
EXPECTED_INFERENCE_ROWS = 4152
REGISTERED_MODELS = {
    "detector_only",
    "qwen3_vl_2b",
    "smolvlm2_2b",
    "qwen3_vl_8b_upper_bound",
}


def _exact_digest(value: Any, label: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise StudyGateError(f"{label} is not one lowercase SHA-256")
    return digest


def _read_regular(path: Path, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as exc:
        raise StudyGateError(f"Missing {label}: {path}") from exc
    if path.is_symlink() or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise StudyGateError(f"Unsafe {label}: {path}")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        if (
            opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
            or opened.st_nlink != 1
        ):
            raise StudyGateError(f"{label} changed while opening: {path}")
        payload = handle.read()
    after = path.lstat()
    if (
        after.st_dev != before.st_dev
        or after.st_ino != before.st_ino
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_nlink != 1
    ):
        raise StudyGateError(f"{label} changed while reading: {path}")
    return payload


def _recovery_receipt() -> tuple[dict[str, Any], str, Path]:
    raw_path = os.environ.get("SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT", "")
    expected_sha256 = os.environ.get(
        "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256", "",
    )
    if not raw_path or not expected_sha256:
        raise StudyGateError(
            "The authenticated v2r9 Experiment-1 recovery receipt is not configured"
        )
    configured_path = Path(raw_path).expanduser()
    if configured_path.is_symlink():
        raise StudyGateError(
            "The configured v2r9 Experiment-1 recovery receipt is a symlink"
        )
    path = configured_path.resolve()
    payload = _read_regular(path, "v2r9 Experiment-1 recovery receipt")
    digest = hashlib.sha256(payload).hexdigest()
    if digest != _exact_digest(expected_sha256, "recovery receipt digest"):
        raise StudyGateError("The v2r9 Experiment-1 recovery receipt changed")
    try:
        receipt = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise StudyGateError("The v2r9 Experiment-1 recovery receipt is invalid") from exc
    if not isinstance(receipt, dict):
        raise StudyGateError("The v2r9 Experiment-1 recovery receipt is not an object")
    config_hash = _exact_digest(
        receipt.get("predecessor_config_sha256"), "predecessor config hash",
    )
    if (
        receipt.get("schema_version") != RECOVERY_SCHEMA
        or receipt.get("status") != RECOVERY_STATUS
        or receipt.get("predecessor_release_id") != PREDECESSOR_RELEASE_ID
        or receipt.get("predecessor_run_name") != PREDECESSOR_RUN_NAME
        or receipt.get("predecessor_code_sha256") != PREDECESSOR_CODE_SHA256
        or not config_hash.startswith(PREDECESSOR_CONFIG_PREFIX)
        or receipt.get("predecessor_model_lock_sha256")
        != PREDECESSOR_MODEL_LOCK_SHA256
        or receipt.get("untouched_test_accessed") is not False
        or receipt.get("protocol_frozen") is not False
        or receipt.get("native_cache_completed") is not False
        or receipt.get("artifact_count") != EXPECTED_ARTIFACT_COUNT
        or receipt.get("inference_row_count") != EXPECTED_INFERENCE_ROWS
    ):
        raise StudyGateError("The v2r9 Experiment-1 recovery contract changed")
    configured_run_root = Path(str(receipt.get("predecessor_run_root", "")))
    if configured_run_root.is_symlink():
        raise StudyGateError("The v2r9 recovery run root is a symlink")
    run_root = configured_run_root.resolve()
    expected_name = (
        f"{PREDECESSOR_RUN_NAME}-full-{config_hash[:10]}-"
        f"{PREDECESSOR_CODE_SHA256[:10]}"
    )
    if run_root.name != expected_name or run_root.is_symlink() or not run_root.is_dir():
        raise StudyGateError("The v2r9 recovery run root is unsafe or changed")
    run_manifest = run_root / "run_manifest.json"
    manifest_payload = _read_regular(run_manifest, "v2r9 run manifest")
    manifest_record = receipt.get("run_manifest")
    if (
        not isinstance(manifest_record, Mapping)
        or Path(str(manifest_record.get("path", ""))).resolve() != run_manifest
        or len(manifest_payload) != manifest_record.get("bytes")
        or hashlib.sha256(manifest_payload).hexdigest()
        != manifest_record.get("sha256")
    ):
        raise StudyGateError("The v2r9 run manifest differs from the recovery receipt")
    manifest = json.loads(manifest_payload.decode("utf-8"))
    if (
        manifest.get("run_name") != PREDECESSOR_RUN_NAME
        or manifest.get("profile") != "full"
        or manifest.get("config_hash") != config_hash
        or manifest.get("code_hash") != PREDECESSOR_CODE_SHA256
    ):
        raise StudyGateError("The v2r9 run manifest identity changed")
    if (run_root / "test_access_ledger.json").exists():
        raise StudyGateError("The v2r9 recovery source unexpectedly accessed test data")
    return receipt, digest, run_root


def _artifact_inventory(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != EXPECTED_ARTIFACT_COUNT:
        raise StudyGateError("The v2r9 recovery artifact inventory changed")
    records: dict[str, Mapping[str, Any]] = {}
    rows = 0
    for record in artifacts:
        if not isinstance(record, Mapping) or set(record) != {
            "relative_path", "bytes", "sha256", "rows",
        }:
            raise StudyGateError("A v2r9 recovery artifact record is malformed")
        relative_text = str(record["relative_path"])
        relative = PurePosixPath(relative_text)
        if (
            relative.is_absolute()
            or relative.as_posix() != relative_text
            or any(part in {"", ".", ".."} for part in relative.parts)
            or len(relative.parts) != 3
            or relative.parts[0] != "stages"
            or not relative.parts[1].startswith("01_vlm_")
            or not relative.parts[2].endswith(".csv.gz")
            or relative_text in records
            or type(record["bytes"]) is not int
            or record["bytes"] <= 0
            or type(record["rows"]) is not int
            or record["rows"] <= 0
        ):
            raise StudyGateError("A v2r9 recovery artifact path/count is unsafe")
        _exact_digest(record["sha256"], f"recovery artifact {relative_text}")
        records[relative_text] = record
        rows += int(record["rows"])
    if rows != EXPECTED_INFERENCE_ROWS:
        raise StudyGateError("The v2r9 recovery inventory row total changed")
    expected_inventory_hash = _exact_digest(
        receipt.get("artifact_inventory_sha256"), "artifact inventory hash",
    )
    if hash_object(artifacts) != expected_inventory_hash:
        raise StudyGateError("The v2r9 recovery inventory digest changed")
    return records


def _atomic_copy_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def import_v2r9_inference_stage(
    ctx: RunContext,
    name: str,
    key_material: Mapping[str, Any],
    *,
    requests: Sequence[Any],
    model_id: str,
    model_revision: str,
    prompt: str,
) -> bool:
    """Install one verified predecessor shard under its current successor key.

    Returns ``True`` when predecessor bytes were imported and ``False`` when
    the current successor stage was already complete.
    """

    if not ctx.is_scientific or ctx.run_name not in {"semuav_visdrone_v2r13", "semuav_visdrone_v3", "semuav_visdrone_v3r1", "semuav_visdrone_v3r2"}:
        raise StudyGateError("v2r9 inference recovery is restricted to the authenticated v2r13/v3/v3r1/v3r2 full runs")
    current_key = hash_object({
        "name": name,
        "key": key_material,
        "config": ctx.config_hash,
        "code": ctx.code_hash,
    })
    if ctx.stage_complete(name, current_key):
        return False

    receipt, receipt_sha256, predecessor_root = _recovery_receipt()
    inventory = _artifact_inventory(receipt)
    predecessor_key_material = copy.deepcopy(dict(key_material))
    execution = predecessor_key_material.get("execution_context")
    if not isinstance(execution, dict):
        raise StudyGateError("The inference shard lacks a recovery execution context")
    current_model_lock = _exact_digest(
        execution.get("model_lock_sha256"), "current model-lock hash",
    )
    if current_model_lock == PREDECESSOR_MODEL_LOCK_SHA256:
        raise StudyGateError("Successor unexpectedly reused the predecessor model-lock identity")
    execution["model_lock_sha256"] = PREDECESSOR_MODEL_LOCK_SHA256
    predecessor_key = hash_object({
        "name": name,
        "key": predecessor_key_material,
        "config": receipt["predecessor_config_sha256"],
        "code": PREDECESSOR_CODE_SHA256,
    })
    relative = f"stages/{name}/{predecessor_key}.csv.gz"
    record = inventory.get(relative)
    if record is None:
        raise StudyGateError(
            f"The authenticated v2r9 recovery inventory lacks expected shard {name}"
        )
    source = predecessor_root.joinpath(*PurePosixPath(relative).parts)
    payload = _read_regular(source, f"v2r9 inference shard {name}")
    if (
        len(payload) != record["bytes"]
        or hashlib.sha256(payload).hexdigest() != record["sha256"]
    ):
        raise StudyGateError(f"The v2r9 inference shard changed: {name}")

    try:
        # Parse the bytes that were actually authenticated above.  Reopening
        # ``source`` here would introduce a mutation window between the digest
        # check and semantic validation.
        frame = pd.read_csv(io.BytesIO(payload), compression="gzip")
    except Exception as exc:
        raise StudyGateError(f"The v2r9 inference shard is unreadable: {name}") from exc
    expected_ids = [str(request.observation_id) for request in requests]
    required_columns = {
        "observation_id", "model_id", "model_revision", "prompt_hash",
        "parsed_json", "schema_valid", "raw_text", "latency_ms",
    }
    if (
        required_columns.difference(frame.columns)
        or len(frame) != record["rows"]
        or len(frame) != len(expected_ids)
        or frame["observation_id"].astype(str).tolist() != expected_ids
        or frame["observation_id"].astype(str).duplicated().any()
        or set(frame["model_id"].astype(str)) != {str(model_id)}
        or str(model_id) not in REGISTERED_MODELS
        or set(frame["model_revision"].astype(str)) != {str(model_revision)}
        or set(frame["prompt_hash"].astype(str)) != {hash_object(prompt)}
        or any(column.startswith("moondream_") for column in frame.columns)
    ):
        raise StudyGateError(f"The v2r9 inference shard content changed: {name}")
    for value in frame["latency_ms"]:
        try:
            latency = float(value)
        except (TypeError, ValueError) as exc:
            raise StudyGateError(f"The v2r9 shard has invalid latency: {name}") from exc
        if not math.isfinite(latency) or latency < 0:
            raise StudyGateError(f"The v2r9 shard has invalid latency: {name}")

    destination = ctx.stage_dir(name) / f"{current_key}.csv.gz"
    _atomic_copy_bytes(destination, payload)
    if hash_file(destination) != record["sha256"]:
        raise StudyGateError(f"The imported successor shard copy changed: {name}")
    ctx.set_stage(
        name,
        current_key,
        "COMPLETE",
        destination,
        len(frame),
        message=(
            "verified_v2r9_development_inference_reuse:"
            f"{receipt_sha256}:{record['sha256']}"
        ),
    )
    return True


__all__ = [
    "EXPECTED_ARTIFACT_COUNT",
    "EXPECTED_INFERENCE_ROWS",
    "PREDECESSOR_CODE_SHA256",
    "PREDECESSOR_MODEL_LOCK_SHA256",
    "RECOVERY_SCHEMA",
    "RECOVERY_STATUS",
    "import_v2r9_inference_stage",
]

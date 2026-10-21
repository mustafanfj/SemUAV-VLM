"""Auditable, standard-library-only deployment and pre-import verifier for V3R2.

This successor authenticates the last v2r13 deployment and the unchanged v2r3
candidate. It never upgrades an old smoke receipt into scientific authorization.
The notebook creates a fresh V3R2 smoke/environment seal before scientific work.
V3R2 uses a new source/receipt/run/kernel sibling.  It also inventories the
quiescent v3r1 validation run so that unchanged validation computations can be
reused exactly (see ``semuav/v3r1_import.py``); that inventory's digest is
pinned in the deployment receipt before any v3r2 validation starts.
The v3 schema/API family intentionally stays stable.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone

DEFAULT_STUDY_ROOT = Path('/data/Lane Detection Datasets/musta/semuav_vlm')
PACKAGE_NAME = 'SemUAV-VLM-VisDrone-Paper-Study'
RELEASE_ID = 'release_20261013_visdrone_paper_v3r2'
RUN_NAME = 'semuav_visdrone_v3r2'
ARCHIVE_NAME = 'SemUAV-VLM-VisDrone-Paper-Study-20261013-v3r2.zip'
NOTEBOOK_NAME = 'SemUAV_VLM_VisDrone_Paper_Study.ipynb'
KERNEL_NAME = 'semuav-vlm-20261013-v3r2'
KERNEL_DISPLAY_NAME = 'Python (SemUAV-VLM 2026-10-13 v3r2)'
V3R1_RUN_NAME = 'semuav_visdrone_v3r1'
V3R1_CODE_SHA256 = '7bb899d11e95b5780278fe38ca653dad34e71137e2f296c7a72056586711db98'
V3R1_CONFIG_PREFIX = '448d523d65'
V3R1_RUN_ID = f'{V3R1_RUN_NAME}-full-{V3R1_CONFIG_PREFIX}-{V3R1_CODE_SHA256[:10]}'
V3R1_IMPORT_SCHEMA = 'semuav.visdrone-v3r1-validation-import.v1'
V3R1_IMPORT_STATUS = 'QUIESCENT_V3R1_VALIDATION_SOURCE_FROZEN'
V3R1_STAGE_PATTERN = re.compile(
    r'^(01_vlm_[A-Za-z0-9_.-]+_shard_[0-9]{5}|'
    r'validation(_temporal)?_02_[A-Za-z0-9_]+_shard_[0-9]{5}|'
    r'03_static_transport_seed_[0-9]+_shard_[0-9]{5}|validation_detector)$'
)
V3R1_RESULT_PATTERN = re.compile(
    r'^validation[a-z0-9_]*_(native_receiver_cache|registered_feature_cache)\.csv\.gz$'
)
CANDIDATE_ID = 'release_20260912_visdrone_inference_candidate_v2r3'
PREDECESSOR_ID = 'release_20260921_visdrone_paper_v2r13'
PREDECESSOR_ARCHIVE_SHA256 = '5c5191b0bf439623bd868acc0a6222de44573c0c46416131e4842fff5b2f5636'
PREDECESSOR_MANIFEST_SHA256 = '53dcc739804542fda76a65ce32d88d654e49e1d09f518c35de93ef18b407d8b3'
PREDECESSOR_RECEIPT_SHA256 = '110fb6d8c061639746b807b76a5069ab76ceb453a88d4d90941ca368c9ae7c4e'
BASE_MANIFEST_SHA256 = 'b5ab89fa98b562f668ca243ebccbd273b1793ca8606d0f4075c5a89ce069bee9'
BASE_CHECKSUM_SHA256 = 'a42d4d22f7bfe2cd37ec2e03b7c452b65184a2c8a519baedb2aeb1c7990c6baa'
REQUIREMENTS_SHA256 = '028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5'
PIP_FREEZE_SHA256 = '5b6aacbcd2b891c1bf968511d17b3385aff2233e99d0bae440f81bc7f977d5d2'
RECOVERY_SHA256 = '6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608'
CONTROLS = (
    'PHASE', 'EXECUTE_PHASE', 'FREEZE_PROTOCOL', 'AUTHORIZE_TEST_ONCE',
    'RUN_NAME', 'RESUME', 'DEVICE', 'RUN_MODEL_SMOKE',
    'BUILD_NATIVE_RECEIVER_CACHE', 'BUILD_FEATURE_CACHE',
)
PHASES = {'prepare', 'validation', 'test', 'report', 'embedded'}


class DeploymentError(RuntimeError):
    """Authentication, reproducibility, or deployment verification failed."""


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set('0123456789abcdef'):
        raise DeploymentError('Expected one lowercase SHA-256 digest')
    return value


def _json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise DeploymentError(f'Duplicate JSON key: {key}')
            result[key] = value
        return result
    try:
        result = json.loads(data, object_pairs_hook=pairs)
    except (ValueError, UnicodeError) as exc:
        raise DeploymentError(f'Invalid JSON: {exc}') from exc
    if not isinstance(result, dict):
        raise DeploymentError('Expected a JSON object')
    return result


def _read(path):
    path = Path(path).absolute()
    before = path.lstat()
    if path.resolve() != path or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise DeploymentError(f'Expected exact single-link regular file: {path}')
    with path.open('rb') as handle:
        opened = os.fstat(handle.fileno())
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise DeploymentError(f'File changed while opening: {path}')
        result = handle.read()
    after = path.lstat()
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_nlink) != (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_nlink
    ):
        raise DeploymentError(f'File changed while reading: {path}')
    return result


def _checked(path, expected):
    data = _read(path)
    if _sha(data) != _digest(expected):
        raise DeploymentError(f'File digest mismatch: {path}')
    return data


def _record(path):
    data = _read(path)
    return {'path': str(Path(path).absolute()), 'bytes': len(data), 'sha256': _sha(data)}


def _relative(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise DeploymentError('Unsafe archive/inventory path')
    rel = PurePosixPath(value)
    if rel.is_absolute() or rel.as_posix() != value or any(x in ('', '.', '..') for x in rel.parts):
        raise DeploymentError(f'Unsafe archive/inventory path: {value}')
    return rel


def _paths(study_root):
    root = Path(study_root).absolute()
    if root.resolve() != root:
        raise DeploymentError('Study root must be an exact directory, without symlinks')
    return {
        'study': root, 'release': root / 'releases' / RELEASE_ID / PACKAGE_NAME,
        'archive': root / 'incoming' / ARCHIVE_NAME,
        'private': root / 'private' / 'visdrone_v3r2_deployment',
        'receipt': root / 'private' / 'visdrone_v3r2_deployment' / 'DEPLOYMENT_READY.json',
        'v3r1_run': root / '.cache' / 'visdrone_paper_v3r1' / 'runs' / V3R1_RUN_ID,
        'v3r1_import': root / 'private' / 'visdrone_v3r2_deployment' / 'V3R1_VALIDATION_IMPORT_SOURCE.json',
        'candidate': root / 'venvs' / CANDIDATE_ID,
        'recovery': root / 'private' / 'visdrone_v2r10_deployment' / 'V2R9_EXP1_RECOVERY_SOURCE.json',
        'kernel': root / '.cache' / 'jupyter' / 'data' / 'kernels' / KERNEL_NAME / 'kernel.json',
    }


def _parameter_source(source, sealed=None):
    try:
        tree = ast.parse(source)
        expected_tree = ast.parse(sealed) if sealed is not None else tree
    except SyntaxError as exc:
        raise DeploymentError(f'Invalid parameter Python: {exc}') from exc
    def assignments(parsed):
        found = {}
        for node in parsed.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONTROLS:
                name = node.targets[0].id
                if name in found:
                    raise DeploymentError(f'Repeated operator control: {name}')
                found[name] = node.value
        stores = [node.id for node in ast.walk(parsed) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in CONTROLS]
        if tuple(found) != CONTROLS or sorted(stores) != sorted(CONTROLS):
            raise DeploymentError('Operator control names, order, or assignments changed')
        return found
    observed, expected = assignments(tree), assignments(expected_tree)
    encoded = source.encode('utf-8')
    lines = encoded.splitlines(keepends=True)
    substitutions = []
    for name, node in observed.items():
        unchanged = ast.dump(node) == ast.dump(expected[name])
        value = node.value if isinstance(node, ast.Constant) else None
        safe = isinstance(node, ast.Constant) and (
            (name not in ('PHASE', 'RUN_NAME', 'DEVICE') and type(value) is bool)
            or (name == 'PHASE' and isinstance(value, str) and value in PHASES)
            or (name == 'RUN_NAME' and value == RUN_NAME)
            or (name == 'DEVICE' and value == 'cuda:0')
        )
        if not unchanged and not safe:
            raise DeploymentError(f'Unsafe operator control RHS: {name}')
        start = sum(map(len, lines[:node.lineno - 1])) + node.col_offset
        end = sum(map(len, lines[:node.end_lineno - 1])) + node.end_col_offset
        substitutions.append((start, end))
    for start, end in sorted(substitutions, reverse=True):
        encoded = encoded[:start] + b'<APPROVED_PARAMETER_RHS>' + encoded[end:]
    return encoded.decode('utf-8')


def _notebook_contract(data, sealed_parameter=None):
    notebook = copy.deepcopy(_json(data))
    if not isinstance(notebook.get('cells'), list):
        raise DeploymentError('Notebook cells are malformed')
    metadata = notebook.get('metadata', {})
    if not isinstance(metadata, dict):
        raise DeploymentError('Notebook metadata is malformed')
    for key in ('language_info', 'widgets', 'signature'):
        metadata.pop(key, None)
    count, parameter = 0, None
    for cell in notebook['cells']:
        if not isinstance(cell, dict) or not isinstance(cell.get('metadata', {}), dict):
            raise DeploymentError('Notebook cell is malformed')
        metadata = cell.setdefault('metadata', {})
        for key in ('trusted', 'execution', 'ExecuteTime', 'collapsed', 'scrolled'):
            metadata.pop(key, None)
        tags = metadata.get('tags', [])
        if not isinstance(tags, list) or any(not isinstance(x, str) for x in tags):
            raise DeploymentError('Notebook tags are malformed')
        source = cell.get('source', '')
        if isinstance(source, list) and all(isinstance(x, str) for x in source):
            source = ''.join(source)
        if not isinstance(source, str):
            raise DeploymentError('Notebook cell source is malformed')
        if 'parameters' in tags:
            if cell.get('cell_type') != 'code':
                raise DeploymentError('Parameter cell is not code')
            count += 1
            parameter = source
            source = _parameter_source(source, sealed_parameter)
        cell['source'] = source
        if cell.get('cell_type') == 'code':
            cell.pop('outputs', None)
            cell.pop('execution_count', None)
    if count != 1:
        raise DeploymentError('Exactly one parameter cell is required')
    return notebook, parameter


def verify_notebook(live, sealed):
    """Allow runtime output/metadata and only the ten safe parameter RHS edits."""
    expected, parameter = _notebook_contract(sealed)
    observed, _ = _notebook_contract(live, parameter)
    if observed != expected:
        raise DeploymentError('Notebook changed outside permitted parameter RHS/output state')


def _archive_contents(archive, digest):
    raw = _checked(archive, digest)
    contents = {}
    with zipfile.ZipFile(io.BytesIO(raw)) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)):
            raise DeploymentError('Archive contains duplicate members')
        total = 0
        for info in bundle.infolist():
            name = info.filename.rstrip('/') if info.is_dir() else info.filename
            rel = _relative(name)
            if rel.parts[0] != PACKAGE_NAME or len(rel.parts) < 2:
                if info.is_dir() and name == PACKAGE_NAME:
                    continue
                raise DeploymentError('Archive has an unexpected package prefix')
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))):
                raise DeploymentError('Archive contains a nonregular member')
            if info.is_dir():
                continue
            total += info.file_size
            if info.file_size > 64 * 1024 * 1024 or total > 128 * 1024 * 1024:
                raise DeploymentError('Source archive exceeds source-only size budget')
            contents['/'.join(rel.parts[1:])] = bundle.read(info)
    manifest = _json(contents['RELEASE_MANIFEST.json'])
    if (manifest.get('schema_version') != 'semuav.visdrone-v3-source-release.v1'
            or manifest.get('release_id') != RELEASE_ID or manifest.get('run_name') != RUN_NAME
            or manifest.get('archive_name') != ARCHIVE_NAME
            or manifest.get('empirical_results_included') is not False
            or manifest.get('dataset_material_included') is not False):
        raise DeploymentError('V3 source manifest identity changed')
    inventory = manifest.get('payload_inventory')
    if not isinstance(inventory, list) or manifest.get('payload_file_count') != len(inventory):
        raise DeploymentError('Malformed source payload inventory')
    records, checksum_lines = {}, []
    for entry in inventory:
        if not isinstance(entry, dict):
            raise DeploymentError('Malformed source payload record')
        rel = str(_relative(entry.get('path')))
        if rel in records or rel in ('RELEASE_MANIFEST.json', 'RELEASE_MANIFEST_SHA256.txt'):
            raise DeploymentError('Duplicate or recursive source payload record')
        if type(entry.get('bytes')) is not int or entry['bytes'] < 0:
            raise DeploymentError('Invalid source payload byte count')
        payload = contents.get(rel)
        if payload is None or len(payload) != entry['bytes'] or _sha(payload) != _digest(entry.get('sha256')):
            raise DeploymentError(f'Archive payload mismatch: {rel}')
        records[rel] = entry
        checksum_lines.append(f"{entry['sha256']}  {rel}")
    if set(contents) != set(records) | {'RELEASE_MANIFEST.json', 'RELEASE_MANIFEST_SHA256.txt'}:
        raise DeploymentError('Archive contains unsealed or missing files')
    checksum_lines.append(f"{_sha(contents['RELEASE_MANIFEST.json'])}  RELEASE_MANIFEST.json")
    if contents['RELEASE_MANIFEST_SHA256.txt'] != ('\n'.join(sorted(checksum_lines)) + '\n').encode():
        raise DeploymentError('Manifest/checksum inventory binding changed')
    if not {NOTEBOOK_NAME, 'v3_deployment.py', 'requirements.txt'}.issubset(records):
        raise DeploymentError('V3 archive lacks required runtime files')
    if _sha(contents['requirements.txt']) != REQUIREMENTS_SHA256:
        raise DeploymentError('V3 would require a different candidate environment')
    return contents, manifest


def _verify_tree(project_root, contents):
    project_root = Path(project_root).absolute()
    if project_root.resolve() != project_root or not project_root.is_dir():
        raise DeploymentError('Source root must be a real directory')
    checkpoint = '.ipynb_checkpoints/' + NOTEBOOK_NAME.replace('.ipynb', '-checkpoint.ipynb')
    observed = set()
    for item in project_root.rglob('*'):
        rel = item.relative_to(project_root).as_posix()
        if item.is_symlink() or item.resolve() != item:
            raise DeploymentError(f'Source symlink rejected: {rel}')
        if item.is_dir():
            if rel == '__pycache__' or '/__pycache__' in rel:
                raise DeploymentError('Source bytecode directory rejected')
            if rel == '.ipynb_checkpoints' or any(key.startswith(rel + '/') for key in contents):
                continue
            raise DeploymentError(f'Unsealed source directory: {rel}')
        data = _read(item)
        if rel == checkpoint:
            verify_notebook(data, contents[NOTEBOOK_NAME])
            continue
        if rel not in contents:
            raise DeploymentError(f'Unsealed source file: {rel}')
        observed.add(rel)
        if rel == NOTEBOOK_NAME:
            verify_notebook(data, contents[rel])
        elif data != contents[rel]:
            raise DeploymentError(f'Live source digest mismatch: {rel}')
    if observed != set(contents):
        raise DeploymentError('Sealed source files are missing')


def verify_predecessor(study_root=DEFAULT_STUDY_ROOT):
    """Authenticate immutable v2r13 anchors without asserting its runtime state."""
    paths = _paths(study_root)
    study = paths['study']
    archive = study / 'incoming' / 'SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip'
    manifest = study / 'releases' / PREDECESSOR_ID / PACKAGE_NAME / 'RELEASE_MANIFEST.json'
    receipt = study / 'private' / 'visdrone_v2r13_deployment' / 'DEPLOYMENT_READY.json'
    _checked(archive, PREDECESSOR_ARCHIVE_SHA256)
    _checked(manifest, PREDECESSOR_MANIFEST_SHA256)
    payload = _json(_checked(receipt, PREDECESSOR_RECEIPT_SHA256))
    if payload.get('schema_version') != 'semuav.visdrone-v2r13-source-deployment.v1' or payload.get('release_id') != PREDECESSOR_ID:
        raise DeploymentError('Pinned predecessor receipt identity changed')
    return {'archive': _record(archive), 'manifest': _record(manifest), 'receipt': _record(receipt)}, payload


def verify_recovery(study_root=DEFAULT_STUDY_ROOT):
    """Recheck exact frozen raw E1 files; no old decision or native cache is reused."""
    paths = _paths(study_root)
    payload = _json(_checked(paths['recovery'], RECOVERY_SHA256))
    if (payload.get('schema_version') != 'semuav.visdrone-v2r9-exp1-recovery.v1'
            or payload.get('status') != 'QUIESCENT_V2R9_EXP1_SOURCE_FROZEN'
            or payload.get('artifact_count') != 132 or payload.get('inference_row_count') != 4152
            or any(payload.get(key) is not False for key in ('untouched_test_accessed', 'protocol_frozen', 'native_cache_completed'))):
        raise DeploymentError('Frozen raw Experiment-1 recovery boundary changed')
    run_root = Path(payload['predecessor_run_root'])
    expected_root = paths['study'] / '.cache' / 'visdrone_paper_v2r9' / 'runs' / 'semuav_visdrone_v2r9-full-d799763aef-fbf80e5ec3'
    if run_root != expected_root:
        raise DeploymentError('Frozen recovery root changed')
    run_manifest = payload['run_manifest']
    if Path(run_manifest['path']) != run_root / 'run_manifest.json':
        raise DeploymentError('Recovery run manifest path changed')
    _checked(run_manifest['path'], run_manifest['sha256'])
    artifacts = payload['artifacts']
    if not isinstance(artifacts, list) or len(artifacts) != 132:
        raise DeploymentError('Recovery artifact count changed')
    seen, rows = set(), 0
    for entry in artifacts:
        rel = _relative(entry['relative_path'])
        if len(rel.parts) != 3 or rel.parts[0] != 'stages' or not rel.parts[1].startswith('01_vlm_') or not rel.parts[2].endswith('.csv.gz') or str(rel) in seen:
            raise DeploymentError('Unsafe recovery artifact')
        seen.add(str(rel))
        data = _checked(run_root / str(rel), entry['sha256'])
        if len(data) != entry['bytes'] or type(entry['rows']) is not int or entry['rows'] < 1:
            raise DeploymentError('Recovery size/row binding changed')
        rows += entry['rows']
    if rows != 4152:
        raise DeploymentError('Recovery inference row total changed')
    return _record(paths['recovery'])


def _canonical_hash(value):
    """Same canonical JSON digest as ``semuav.registry.hash_object`` for plain JSON."""
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


def _v3r1_status_rows(run_root):
    database = run_root / 'status.sqlite'
    before = database.lstat()
    connection = sqlite3.connect(f'file:{database}?mode=ro', uri=True)
    try:
        rows = connection.execute(
            'SELECT name, cache_key, status, artifact, checksum, rows FROM stages ORDER BY name'
        ).fetchall()
    finally:
        connection.close()
    return rows, (before.st_size, before.st_mtime_ns)


def build_v3r1_import_inventory(study_root=DEFAULT_STUDY_ROOT):
    """Inventory exact reusable v3r1 validation artifacts (read-only)."""
    paths = _paths(study_root)
    run_root = paths['v3r1_run']
    if run_root.resolve() != run_root or not run_root.is_dir():
        raise DeploymentError(f'v3r1 run root is missing or a symlink: {run_root}')
    for forbidden in ('test_access_ledger.json', 'frozen_protocol.json'):
        if (run_root / forbidden).exists():
            raise DeploymentError(f'v3r1 run is not a validation-only source: {forbidden} exists')
    manifest_record = _record(run_root / 'run_manifest.json')
    manifest = _json(_read(run_root / 'run_manifest.json'))
    config_hash = _digest(manifest.get('config_hash'))
    if (manifest.get('run_name') != V3R1_RUN_NAME or manifest.get('profile') != 'full'
            or manifest.get('code_hash') != V3R1_CODE_SHA256
            or not config_hash.startswith(V3R1_CONFIG_PREFIX)):
        raise DeploymentError('v3r1 run manifest identity changed')
    rows, stamp = _v3r1_status_rows(run_root)
    stages = []
    for name, cache_key, status, artifact, checksum, row_count in rows:
        if status != 'COMPLETE' or not V3R1_STAGE_PATTERN.fullmatch(str(name)):
            continue
        cache_key = _digest(cache_key)
        expected = run_root / 'stages' / name / f'{cache_key}.csv.gz'
        if Path(str(artifact)) != expected:
            raise DeploymentError(f'v3r1 stage artifact path is unexpected: {name}')
        record = _record(expected)
        if record['sha256'] != checksum:
            raise DeploymentError(f'v3r1 stage artifact differs from its status checksum: {name}')
        stages.append({'name': name, 'cache_key': cache_key,
                       'relative_path': f'stages/{name}/{cache_key}.csv.gz',
                       'bytes': record['bytes'], 'sha256': record['sha256'],
                       'rows': int(row_count) if row_count is not None else None})
    results = []
    for item in sorted((run_root / 'results').iterdir()):
        if item.is_file() and V3R1_RESULT_PATTERN.fullmatch(item.name):
            record = _record(item)
            results.append({'relative_path': f'results/{item.name}',
                            'bytes': record['bytes'], 'sha256': record['sha256']})
    _, after = _v3r1_status_rows(run_root)
    if after != stamp:
        raise DeploymentError('v3r1 status database changed during inventory; shut down the v3r1 kernel')
    if not stages or not results:
        raise DeploymentError('v3r1 run has no reusable validation stages or result caches')
    return {
        'schema_version': V3R1_IMPORT_SCHEMA, 'status': V3R1_IMPORT_STATUS,
        'predecessor_run_name': V3R1_RUN_NAME, 'predecessor_run_root': str(run_root),
        'predecessor_code_sha256': V3R1_CODE_SHA256, 'predecessor_config_sha256': config_hash,
        'run_manifest': manifest_record, 'untouched_test_accessed': False,
        'protocol_frozen': False, 'experiment4_partitions_imported': False,
        'stages': stages, 'results': results,
        'inventory_sha256': _canonical_hash({'stages': stages, 'results': results}),
        'stage_count': len(stages), 'result_count': len(results),
    }


def verify_v3r1_import(study_root=DEFAULT_STUDY_ROOT, expected=None):
    """Recheck the frozen v3r1 inventory file and every listed artifact."""
    paths = _paths(study_root)
    payload = _json(_read(paths['v3r1_import']) if expected is None else _checked(paths['v3r1_import'], expected))
    if (payload.get('schema_version') != V3R1_IMPORT_SCHEMA or payload.get('status') != V3R1_IMPORT_STATUS
            or payload.get('predecessor_run_root') != str(paths['v3r1_run'])
            or payload.get('inventory_sha256') != _canonical_hash({'stages': payload['stages'], 'results': payload['results']})):
        raise DeploymentError('v3r1 import inventory identity changed')
    root = paths['v3r1_run']
    if (root / 'test_access_ledger.json').exists() or (root / 'frozen_protocol.json').exists():
        raise DeploymentError('v3r1 import source is no longer validation-only')
    for entry in [*payload['stages'], *payload['results']]:
        rel = _relative(entry['relative_path'])
        record = _record(root / str(rel))
        if record['bytes'] != entry['bytes'] or record['sha256'] != entry['sha256']:
            raise DeploymentError(f'v3r1 import artifact changed: {rel}')
    return _record(paths['v3r1_import'])


def configure_v3r1_import(study_root=DEFAULT_STUDY_ROOT, *, phase, receipt_sha256):
    """Enable exact v3r1 reuse for v3r2 validation only."""
    for name in ('SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT', 'SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT',
                 'SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT_SHA256'):
        os.environ.pop(name, None)
    if phase != 'validation':
        return None
    paths = _paths(study_root)
    receipt = _json(_checked(paths['receipt'], receipt_sha256))
    pinned = receipt.get('v3r1_validation_import')
    if not isinstance(pinned, dict) or pinned.get('path') != str(paths['v3r1_import']):
        raise DeploymentError('Deployment receipt lacks the pinned v3r1 import inventory')
    _checked(paths['v3r1_import'], pinned['sha256'])
    os.environ['SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT'] = pinned['path']
    os.environ['SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT_SHA256'] = pinned['sha256']
    os.environ['SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT'] = '1'
    return pinned


def configure_recovery(study_root=DEFAULT_STUDY_ROOT, *, phase):
    if phase != 'validation':
        raise DeploymentError('Raw Experiment-1 recovery is available only after the validation seal')
    record = verify_recovery(study_root)
    os.environ['SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT'] = record['path']
    os.environ['SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256'] = record['sha256']
    if phase == 'validation':
        os.environ['SEMUAV_ENABLE_V2R9_EXP1_RECOVERY'] = '1'
    else:
        os.environ.pop('SEMUAV_ENABLE_V2R9_EXP1_RECOVERY', None)
    return record


def _candidate_facts(study_root, predecessor_payload):
    paths = _paths(study_root)
    candidate = paths['candidate']
    source = paths['study'] / 'releases' / 'release_20260912_visdrone_paper_v2r3' / PACKAGE_NAME
    _checked(source / 'RELEASE_MANIFEST.json', BASE_MANIFEST_SHA256)
    _checked(source / 'RELEASE_MANIFEST_SHA256.txt', BASE_CHECKSUM_SHA256)
    _checked(source / 'requirements.txt', REQUIREMENTS_SHA256)
    prior = predecessor_payload['candidate_environment']
    ready_path = candidate / 'SEMUAV_CANDIDATE_READY.json'
    if Path(prior['path']) != candidate or Path(prior['ready_receipt']) != ready_path or prior['pip_freeze_sha256'] != PIP_FREEZE_SHA256:
        raise DeploymentError('Predecessor candidate binding changed')
    ready = _json(_checked(ready_path, prior['ready_receipt_sha256']))
    if (ready.get('schema_version') != 'semuav.visdrone-inference-candidate-environment.v1'
            or ready.get('status') != 'READY_FOR_DEVELOPMENT_SMOKE_ONLY'
            or ready.get('candidate_only') is not True
            or ready.get('candidate_environment') != str(candidate)
            or any(ready.get(key) is not False for key in ('scientific_results_authorized', 'validation_or_test_data_read', 'legacy_environment_invoked', 'canonical_release_modified'))):
        raise DeploymentError('Candidate ready receipt identity changed')
    expected_source = {'path': str(source), 'release_id': 'release_20260912_visdrone_paper_v2r3', 'run_name': 'semuav_visdrone_v2r3', 'manifest_sha256': BASE_MANIFEST_SHA256, 'checksum_inventory_sha256': BASE_CHECKSUM_SHA256, 'requirements_sha256': REQUIREMENTS_SHA256}
    observed_source = ready.get('source_release')
    if not isinstance(observed_source, dict) or any(observed_source.get(key) != value for key, value in expected_source.items()):
        raise DeploymentError('Candidate source binding changed')
    freeze_record = ready['artifacts']['pip_freeze']
    freeze = _checked(freeze_record['path'], PIP_FREEZE_SHA256)
    if freeze_record['sha256'] != PIP_FREEZE_SHA256:
        raise DeploymentError('Candidate recorded distribution inventory changed')
    python = candidate / 'bin' / 'python'
    if not python.is_file() or candidate.resolve() != candidate:
        raise DeploymentError('Candidate Python is missing or candidate directory is a symlink')
    observed = subprocess.run([str(python), '-I', '-B', '-c', 'import sys; print(sys.prefix)'], check=True, capture_output=True, text=True).stdout.strip()
    if observed != str(candidate):
        raise DeploymentError('Candidate Python prefix changed')
    current = subprocess.run([str(python), '-I', '-B', '-m', 'pip', 'freeze', '--all'], check=True, capture_output=True).stdout
    if current != freeze:
        raise DeploymentError('Installed distributions differ from the frozen v2r3 candidate')
    facts = {'path': str(candidate), 'ready_receipt': str(ready_path), 'ready_receipt_sha256': _sha(_read(ready_path)), 'pip_freeze_sha256': _sha(current), 'requirements_sha256': REQUIREMENTS_SHA256}
    return facts, ready


def _kernel_payload(paths):
    return {'argv': [str(paths['candidate'] / 'bin' / 'python'), '-I', '-B', '-m', 'ipykernel_launcher', '-f', '{connection_file}'], 'display_name': KERNEL_DISPLAY_NAME, 'language': 'python', 'metadata': {'debugger': True}}


def verify_source(project_root, receipt_sha256, *, study_root=DEFAULT_STUDY_ROOT):
    """Authenticate V3 archive, receipt, bootstrap, source tree and kernel."""
    paths = _paths(study_root)
    if Path(project_root).absolute() != paths['release']:
        raise DeploymentError('Project root differs from the exact V3 release')
    receipt = _json(_checked(paths['receipt'], receipt_sha256))
    if (receipt.get('schema_version') != 'semuav.visdrone-v3-source-deployment.v1'
            or receipt.get('status') != 'READY_FOR_DEVELOPMENT_SMOKE_ONLY'
            or receipt.get('release_id') != RELEASE_ID or receipt.get('run_name') != RUN_NAME
            or receipt.get('release_root') != str(paths['release'])
            or receipt.get('runtime_smoke_run') is not False
            or receipt.get('scientific_validation_authorized') is not False
            or receipt.get('package_installation_performed') is not False):
        raise DeploymentError('V3 deployment receipt identity/scope changed')
    binding = receipt['source_release']
    if binding['archive_path'] != str(paths['archive']):
        raise DeploymentError('V3 sealed archive path changed')
    contents, manifest = _archive_contents(paths['archive'], binding['archive_sha256'])
    if _sha(contents['RELEASE_MANIFEST.json']) != binding['manifest_sha256'] or _sha(contents['RELEASE_MANIFEST_SHA256.txt']) != binding['checksum_inventory_sha256']:
        raise DeploymentError('V3 deployment manifest binding changed')
    bootstrap = receipt['bootstrap']
    if bootstrap != {'path': str(paths['release'] / 'v3_deployment.py'), 'sha256': _sha(contents['v3_deployment.py'])}:
        raise DeploymentError('V3 pre-import verifier binding changed')
    _verify_tree(paths['release'], contents)
    predecessor, _ = verify_predecessor(study_root)
    if receipt.get('authenticated_predecessor_v2r13') != predecessor:
        raise DeploymentError('Predecessor provenance binding changed')
    if receipt.get('experiment1_recovery') != _record(paths['recovery']) or _sha(_read(paths['recovery'])) != RECOVERY_SHA256:
        raise DeploymentError('Recovery receipt binding changed')
    if _json(_read(paths['kernel'])) != _kernel_payload(paths) or receipt.get('jupyter_kernel') != _record(paths['kernel']):
        raise DeploymentError('V3 isolated Jupyter kernel changed')
    if receipt.get('v3r1_validation_import') != _record(paths['v3r1_import']):
        raise DeploymentError('Pinned v3r1 validation import inventory changed')
    tests = receipt['unit_tests']
    summary = tests['summary']
    if (summary.get('passed') is not True or type(summary.get('tests_run')) is not int
            or summary['tests_run'] < 1 or summary.get('failures') != 0 or summary.get('errors') != 0):
        raise DeploymentError('V3 receipt lacks successful CPU unit-test evidence')
    test_log = Path(tests['log']['path'])
    if test_log.parent != paths['private'] or _record(test_log) != tests['log']:
        raise DeploymentError('Deployment unit-test log binding changed')
    return {'receipt': receipt, 'receipt_sha256': receipt_sha256, 'manifest': manifest, 'project_root': str(paths['release'])}


def verify_candidate(project_root, receipt_sha256, *, study_root=DEFAULT_STUDY_ROOT, require_inference_lock=True, inference_pip_freeze=None):
    verified = verify_source(project_root, receipt_sha256, study_root=study_root)
    paths = _paths(study_root)
    if (Path(sys.executable).absolute() != paths['candidate'] / 'bin' / 'python'
            or Path(sys.prefix).absolute() != paths['candidate']
            or sys.flags.isolated != 1 or sys.flags.dont_write_bytecode != 1):
        raise DeploymentError('Select the exact isolated V3 candidate kernel')
    _, prior = verify_predecessor(study_root)
    facts, ready = _candidate_facts(study_root, prior)
    if facts != verified['receipt']['candidate_environment']:
        raise DeploymentError('V3 candidate binding changed since deployment')
    if require_inference_lock or (inference_pip_freeze is not None and Path(inference_pip_freeze).exists()):
        if inference_pip_freeze is None:
            raise DeploymentError('Fresh V3 inference pip freeze is required')
        _checked(inference_pip_freeze, PIP_FREEZE_SHA256)
    source = verified['receipt']['source_release']
    compatibility = {
        'schema_version': 'semuav.visdrone-v3-source-candidate-binding.v1',
        'release_id': RELEASE_ID, 'run_name': RUN_NAME, 'release_path': str(paths['release']),
        'release_manifest_sha256': source['manifest_sha256'],
        'checksum_inventory_sha256': source['checksum_inventory_sha256'],
        'requirements_sha256': REQUIREMENTS_SHA256, 'candidate_environment': str(paths['candidate']),
        'candidate_pip_freeze_sha256': PIP_FREEZE_SHA256,
        'candidate_requirements_changed': False, 'candidate_environment_reused_without_rebuild': True,
        'package_installation_performed': False,
        'compatibility_scope': 'unchanged_candidate_distributions_and_historical_raw_E1_only; V3_corrected_source_requires_fresh_smoke_and_new_run',
        'fresh_runtime_smoke_required': True, 'predecessor_smoke_rows_reused': False,
        'active_model_ids': ['qwen3_vl_2b', 'smolvlm2_2b', 'qwen3_vl_8b_upper_bound'],
        'excluded_model_ids': ['moondream2'],
        'authenticated_predecessor_v2r13': verified['receipt']['authenticated_predecessor_v2r13'],
        'deployment_receipt_sha256': receipt_sha256, 'archive_sha256': source['archive_sha256'],
    }
    return {'payload': ready, 'path': Path(facts['ready_receipt']), 'sha256': facts['ready_receipt_sha256'], 'artifacts': ready['artifacts'], 'source_compatibility': compatibility}


def _write_new_or_exact(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        if _read(path) != data:
            raise DeploymentError(f'Existing file differs; left unchanged: {path}')
        return
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()


def _run_tests(paths):
    code = """import json, sys, unittest
sys.path.insert(0, sys.argv[1])
suite = unittest.defaultTestLoader.discover(sys.argv[1] + '/tests')
result = unittest.TextTestRunner(verbosity=1).run(suite)
print(json.dumps({'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'skipped': len(result.skipped), 'passed': result.wasSuccessful()}))
sys.exit(0 if result.wasSuccessful() else 1)
"""
    test_environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', NVIDIA_VISIBLE_DEVICES='void', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SEMUAV_MODELS_LOCAL_ONLY='1', YOLO_AUTOINSTALL='false', PIP_NO_INDEX='1', PYTHONDONTWRITEBYTECODE='1')
    for name in ('BASH_ENV', 'ENV', 'LD_PRELOAD', 'LD_LIBRARY_PATH', 'PYTHONPATH', 'PYTHONHOME',
                 'HF_HUB_CACHE', 'HUGGINGFACE_HUB_CACHE', 'TRANSFORMERS_CACHE',
                 'SEMUAV_QWEN_REVISION', 'SEMUAV_SMOLVLM_REVISION', 'SEMUAV_MOONDREAM_REVISION',
                 'SEMUAV_UPPER_BOUND_REVISION', 'SEMUAV_MOONDREAM_TOKENIZER_LOCK',
                 'SEMUAV_ENABLE_V2R9_EXP1_RECOVERY', 'SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT',
                 'SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256'):
        test_environment.pop(name, None)
    process = subprocess.run([str(paths['candidate'] / 'bin' / 'python'), '-I', '-B', '-c', code, str(paths['release'])], capture_output=True, cwd=paths['release'], env=test_environment)
    log_path = paths['private'] / ('deployment_tests_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.log')
    _write_new_or_exact(log_path, process.stdout + b'\n' + process.stderr)
    if process.returncode:
        raise DeploymentError(f'Deployed CPU unit tests failed; inspect {log_path}')
    summary = _json(process.stdout.splitlines()[-1])
    if summary.get('passed') is not True or summary.get('tests_run', 0) < 1:
        raise DeploymentError('No passing deployment test evidence')
    return {'summary': summary, 'log': _record(log_path)}


def deploy(archive, archive_sha256, *, study_root=DEFAULT_STUDY_ROOT):
    """Publish a new immutable source sibling; never install, train or download."""
    paths = _paths(study_root)
    contents, manifest = _archive_contents(archive, archive_sha256)
    predecessor, prior = verify_predecessor(study_root)
    candidate, _ = _candidate_facts(study_root, prior)
    recovery = verify_recovery(study_root)
    if paths['receipt'].exists():
        verified = verify_source(paths['release'], _sha(_read(paths['receipt'])), study_root=study_root)
        if verified['receipt']['source_release']['archive_sha256'] != archive_sha256 or verified['receipt']['candidate_environment'] != candidate:
            raise DeploymentError('Existing V3 deployment differs; nothing overwritten')
        return _record(paths['receipt'])
    _write_new_or_exact(paths['archive'], _read(archive))
    parent = paths['release'].parent
    if paths['release'].exists():
        _verify_tree(paths['release'], contents)
    else:
        parent.parent.mkdir(parents=True, exist_ok=True)
        if parent.exists():
            raise DeploymentError('V3 release parent exists without its complete source')
        temporary = Path(tempfile.mkdtemp(prefix='.v3r2-deploy-', dir=parent.parent))
        try:
            source = temporary / PACKAGE_NAME
            source.mkdir()
            for name, data in contents.items():
                destination = source / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
            _verify_tree(source, contents)
            os.rename(temporary, parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
    _write_new_or_exact(paths['kernel'], _json_bytes(_kernel_payload(paths)))
    _write_new_or_exact(paths['v3r1_import'], _json_bytes(build_v3r1_import_inventory(study_root)))
    v3r1_import = verify_v3r1_import(study_root)
    tests = _run_tests(paths)
    after_predecessor, after_prior = verify_predecessor(study_root)
    after_candidate, _ = _candidate_facts(study_root, after_prior)
    if (after_predecessor != predecessor or after_candidate != candidate
            or verify_recovery(study_root) != recovery or verify_v3r1_import(study_root) != v3r1_import):
        raise DeploymentError('Historical provenance changed during source deployment')
    _verify_tree(paths['release'], contents)
    receipt = {
        'schema_version': 'semuav.visdrone-v3-source-deployment.v1',
        'status': 'READY_FOR_DEVELOPMENT_SMOKE_ONLY',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'release_id': RELEASE_ID, 'run_name': RUN_NAME, 'release_root': str(paths['release']),
        'scope': 'new_source_sibling_CPU_unit_tests_and_isolated_kernel_registration',
        'package_installation_performed': False, 'runtime_smoke_run': False,
        'scientific_validation_authorized': False, 'scientific_experiments_run': False,
        'model_weights_loaded': False, 'gpu_accessed': False,
        'validation_or_test_images_or_labels_read': False,
        'source_release': {'archive_path': str(paths['archive']), 'archive_sha256': archive_sha256,
                           'manifest_sha256': _sha(contents['RELEASE_MANIFEST.json']),
                           'checksum_inventory_sha256': _sha(contents['RELEASE_MANIFEST_SHA256.txt'])},
        'bootstrap': {'path': str(paths['release'] / 'v3_deployment.py'), 'sha256': _sha(contents['v3_deployment.py'])},
        'authenticated_predecessor_v2r13': predecessor,
        'candidate_environment': candidate, 'experiment1_recovery': recovery,
        'unit_tests': tests, 'jupyter_kernel': _record(paths['kernel']),
        'v3r1_validation_import': v3r1_import,
        'next_gate': 'FRESH_V3R2_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE',
    }
    _write_new_or_exact(paths['receipt'], _json_bytes(receipt))
    verify_source(paths['release'], _sha(_read(paths['receipt'])), study_root=study_root)
    return _record(paths['receipt'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('deploy', 'verify'))
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--archive-sha256')
    args = parser.parse_args(argv)
    if args.action == 'deploy':
        if args.archive is None or args.archive_sha256 is None:
            parser.error('deploy requires --archive and --archive-sha256')
        result = deploy(args.archive, _digest(args.archive_sha256))
        print('V3R2_SOURCE_DEPLOYMENT_COMPLETE')
        print(json.dumps(result, sort_keys=True))
    else:
        paths = _paths(DEFAULT_STUDY_ROOT)
        digest = os.environ.get('SEMUAV_V3_DEPLOYMENT_RECEIPT_SHA256', '')
        verify_source(paths['release'], _digest(digest))
        _, prior = verify_predecessor()
        _candidate_facts(DEFAULT_STUDY_ROOT, prior)
        print('V3R2_SOURCE_AND_CANDIDATE_VERIFIED')


if __name__ == '__main__':
    try:
        main()
    except (DeploymentError, OSError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as exc:
        raise SystemExit(f'STOP: {exc}') from exc

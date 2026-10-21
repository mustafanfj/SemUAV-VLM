"""V3R2 source deployment integration tests: no real lab, GPU or datasets needed."""
import hashlib
import sqlite3
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('semuav_v3_deployment_test', ROOT / 'v3_deployment.py')
deployment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deployment)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()


class DeploymentIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.study = Path(self.temporary.name)
        self.paths = deployment._paths(self.study)
        self.requirements = b'fixture-package==1.0\n'
        self.patch_requirements = mock.patch.object(deployment, 'REQUIREMENTS_SHA256', digest(self.requirements))
        self.patch_requirements.start()
        self.addCleanup(self.patch_requirements.stop)
        self.controls = '\n'.join(f'{key} = {value!r}' for key, value in zip(deployment.CONTROLS, ('prepare', False, False, False, deployment.RUN_NAME, True, 'cuda:0', False, False, False))) + '\n'
        self.notebook = {'nbformat': 4, 'nbformat_minor': 5, 'metadata': {'kernelspec': {'name': deployment.KERNEL_NAME}}, 'cells': [
            {'cell_type': 'code', 'metadata': {'tags': ['parameters']}, 'source': self.controls, 'execution_count': None, 'outputs': []},
            {'cell_type': 'code', 'metadata': {}, 'source': 'answer = 42\n', 'execution_count': None, 'outputs': []},
        ]}
        self.payloads = {
            deployment.NOTEBOOK_NAME: json_bytes(self.notebook),
            'requirements.txt': self.requirements,
            'v3_deployment.py': (ROOT / 'v3_deployment.py').read_bytes(),
            'tests/test_cpu_fixture.py': b'import unittest\nclass Fixture(unittest.TestCase):\n    def test_fixture(self):\n        self.assertEqual(6 * 7, 42)\n',
        }
        self.archive = self.study / 'delivery' / deployment.ARCHIVE_NAME
        self._seal_archive()
        candidate = self.paths['candidate']
        (candidate / 'bin').mkdir(parents=True)
        (candidate / 'bin' / 'python').symlink_to(sys.executable)
        self.candidate = {'path': str(candidate), 'ready_receipt': str(candidate / 'SEMUAV_CANDIDATE_READY.json'), 'ready_receipt_sha256': 'd' * 64, 'pip_freeze_sha256': deployment.PIP_FREEZE_SHA256, 'requirements_sha256': digest(self.requirements)}
        predecessor_receipt = {'schema_version': 'semuav.visdrone-v2r13-source-deployment.v1', 'release_id': deployment.PREDECESSOR_ID, 'candidate_environment': self.candidate}
        self.predecessor_files = {
            'PREDECESSOR_ARCHIVE_SHA256': (self.study / 'incoming' / 'SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip', b'fixture old archive'),
            'PREDECESSOR_MANIFEST_SHA256': (self.study / 'releases' / deployment.PREDECESSOR_ID / deployment.PACKAGE_NAME / 'RELEASE_MANIFEST.json', b'fixture old manifest'),
            'PREDECESSOR_RECEIPT_SHA256': (self.study / 'private' / 'visdrone_v2r13_deployment' / 'DEPLOYMENT_READY.json', json_bytes(predecessor_receipt)),
        }
        for constant, (path, data) in self.predecessor_files.items():
            write(path, data)
            patcher = mock.patch.object(deployment, constant, digest(data))
            patcher.start()
            self.addCleanup(patcher.stop)
        recovery_root = self.study / '.cache' / 'visdrone_paper_v2r9' / 'runs' / 'semuav_visdrone_v2r9-full-d799763aef-fbf80e5ec3'
        run_manifest = write(recovery_root / 'run_manifest.json', b'{}\n')
        artifacts = []
        for index in range(132):
            relative = f'stages/01_vlm_fixture/part-{index:03d}.csv.gz'
            artifact = write(recovery_root / relative, f'raw fixture {index}'.encode())
            artifacts.append({'relative_path': relative, 'sha256': digest(artifact.read_bytes()), 'bytes': artifact.stat().st_size, 'rows': 91 if index == 0 else 31})
        recovery = {'schema_version': 'semuav.visdrone-v2r9-exp1-recovery.v1', 'status': 'QUIESCENT_V2R9_EXP1_SOURCE_FROZEN', 'artifact_count': 132, 'inference_row_count': 4152, 'untouched_test_accessed': False, 'protocol_frozen': False, 'native_cache_completed': False, 'predecessor_run_root': str(recovery_root), 'run_manifest': deployment._record(run_manifest), 'artifacts': artifacts}
        data = json_bytes(recovery)
        write(self.paths['recovery'], data)
        patcher = mock.patch.object(deployment, 'RECOVERY_SHA256', digest(data))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.candidate_mock = mock.patch.object(deployment, '_candidate_facts', return_value=(self.candidate, {}))
        self.candidate_mock.start()
        self.addCleanup(self.candidate_mock.stop)
        self._make_v3r1_run()

    def _make_v3r1_run(self):
        root = self.paths['v3r1_run']
        config = {'study_id': 'fixture', 'paths': {'project_root': '/fixture'}}
        config_hash = deployment.V3R1_CONFIG_PREFIX + 'a' * 54
        write(root / 'run_manifest.json', json_bytes({'run_name': deployment.V3R1_RUN_NAME, 'profile': 'full', 'code_hash': deployment.V3R1_CODE_SHA256, 'config_hash': config_hash, 'config': config}))
        rows = []
        def stage(name, status, data):
            key = digest(name.encode())
            artifact = write(root / 'stages' / name / f'{key}.csv.gz', data)
            rows.append((name, key, status, str(artifact), digest(data), 3))
        stage('03_static_transport_seed_1_shard_00000', 'COMPLETE', b'exp3 fixture')
        stage('01_vlm_detector_only_shard_00000', 'COMPLETE', b'inference fixture')
        stage('04_policy_partition_00000', 'COMPLETE', b'controller fixture')
        stage('03_static_transport_seed_1_shard_00001', 'INVALIDATED', b'quarantined fixture')
        write(root / 'results' / 'validation_detector_only_native_receiver_cache.csv.gz', b'native fixture')
        write(root / 'results' / 'validation_exp4_summary.csv.gz', b'not reusable')
        connection = sqlite3.connect(root / 'status.sqlite')
        connection.execute('CREATE TABLE stages (name TEXT PRIMARY KEY, cache_key TEXT NOT NULL, status TEXT NOT NULL, artifact TEXT, checksum TEXT, rows INTEGER, started_utc TEXT, ended_utc TEXT, message TEXT)')
        connection.executemany('INSERT INTO stages VALUES (?,?,?,?,?,?,NULL,NULL,NULL)', rows)
        connection.commit()
        connection.close()

    def _seal_archive(self):
        manifest = {'schema_version': 'semuav.visdrone-v3-source-release.v1', 'release_id': deployment.RELEASE_ID, 'run_name': deployment.RUN_NAME, 'archive_name': deployment.ARCHIVE_NAME, 'empirical_results_included': False, 'dataset_material_included': False, 'payload_file_count': len(self.payloads), 'payload_inventory': [{'path': name, 'bytes': len(data), 'sha256': digest(data)} for name, data in sorted(self.payloads.items())]}
        raw_manifest = json_bytes(manifest)
        lines = [f'{digest(data)}  {name}' for name, data in self.payloads.items()] + [f'{digest(raw_manifest)}  RELEASE_MANIFEST.json']
        contents = dict(self.payloads, **{'RELEASE_MANIFEST.json': raw_manifest, 'RELEASE_MANIFEST_SHA256.txt': ('\n'.join(sorted(lines)) + '\n').encode()})
        self.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(self.archive, 'w') as bundle:
            for name, data in contents.items():
                bundle.writestr(deployment.PACKAGE_NAME + '/' + name, data)
        self.archive_hash = digest(self.archive.read_bytes())

    def _deploy(self):
        return deployment.deploy(self.archive, self.archive_hash, study_root=self.study)

    def test_v3r2_uses_disjoint_source_receipt_run_and_kernel_namespaces(self):
        self.assertEqual(deployment.RELEASE_ID, 'release_20261013_visdrone_paper_v3r2')
        self.assertEqual(deployment.RUN_NAME, 'semuav_visdrone_v3r2')
        self.assertEqual(deployment.ARCHIVE_NAME, 'SemUAV-VLM-VisDrone-Paper-Study-20261013-v3r2.zip')
        self.assertEqual(deployment.KERNEL_NAME, 'semuav-vlm-20261013-v3r2')
        self.assertEqual(self.paths['receipt'].parent.name, 'visdrone_v3r2_deployment')
        old_release = self.study / 'releases/release_20260928_visdrone_paper_v3' / deployment.PACKAGE_NAME
        old_receipt = self.study / 'private/visdrone_v3_deployment/DEPLOYMENT_READY.json'
        write(old_release / 'failed_attempt.txt', b'preserved old source evidence\n')
        write(old_receipt, b'preserved old receipt evidence\n')
        self._deploy()
        self.assertEqual((old_release / 'failed_attempt.txt').read_bytes(), b'preserved old source evidence\n')
        self.assertEqual(old_receipt.read_bytes(), b'preserved old receipt evidence\n')

    def test_deploy_runs_actual_cpu_tests_and_reverify_is_idempotent(self):
        before = {str(path): path.read_bytes() for path, _ in self.predecessor_files.values()}
        record = self._deploy()
        receipt = json.loads(Path(record['path']).read_bytes())
        self.assertEqual(receipt['unit_tests']['summary']['tests_run'], 1)
        self.assertTrue(receipt['unit_tests']['summary']['passed'])
        self.assertFalse(receipt['runtime_smoke_run'])
        self.assertFalse(receipt['scientific_validation_authorized'])
        self.assertEqual(self._deploy(), record)
        self.assertEqual({str(Path(path)): Path(path).read_bytes() for path in before}, before)
        deployment.verify_source(self.paths['release'], record['sha256'], study_root=self.study)

    def test_saved_outputs_and_safe_parameter_rhs_are_allowed(self):
        record = self._deploy()
        notebook = json.loads((self.paths['release'] / deployment.NOTEBOOK_NAME).read_bytes())
        notebook['cells'][0]['source'] = self.controls.replace("PHASE = 'prepare'", "PHASE = 'validation'").replace('EXECUTE_PHASE = False', 'EXECUTE_PHASE = True')
        notebook['cells'][1]['execution_count'] = 7
        notebook['cells'][1]['outputs'] = [{'output_type': 'stream', 'name': 'stdout', 'text': 'done'}]
        write(self.paths['release'] / deployment.NOTEBOOK_NAME, json_bytes(notebook))
        deployment.verify_source(self.paths['release'], record['sha256'], study_root=self.study)

    def test_unsafe_rhs_other_code_and_source_tampering_are_rejected(self):
        record = self._deploy()
        notebook_path = self.paths['release'] / deployment.NOTEBOOK_NAME
        for replacement in ("PHASE = __import__('os').getenv('PHASE')", "PHASE = 'validation'; injected = True"):
            notebook = json.loads(self.payloads[deployment.NOTEBOOK_NAME])
            notebook['cells'][0]['source'] = self.controls.replace("PHASE = 'prepare'", replacement)
            write(notebook_path, json_bytes(notebook))
            with self.assertRaises(deployment.DeploymentError):
                deployment.verify_source(self.paths['release'], record['sha256'], study_root=self.study)
        write(notebook_path, self.payloads[deployment.NOTEBOOK_NAME])
        write(self.paths['release'] / 'v3_deployment.py', b'print("tampered")\n')
        with self.assertRaises(deployment.DeploymentError):
            deployment.verify_source(self.paths['release'], record['sha256'], study_root=self.study)

    def test_v3r1_import_inventory_is_pinned_and_excludes_controller_trials(self):
        record = self._deploy()
        receipt = json.loads(Path(record['path']).read_bytes())
        pinned = receipt['v3r1_validation_import']
        inventory = json.loads(Path(pinned['path']).read_bytes())
        names = sorted(item['name'] for item in inventory['stages'])
        self.assertEqual(names, ['01_vlm_detector_only_shard_00000', '03_static_transport_seed_1_shard_00000'])
        self.assertEqual([item['relative_path'] for item in inventory['results']],
                         ['results/validation_detector_only_native_receiver_cache.csv.gz'])
        self.assertFalse(inventory['experiment4_partitions_imported'])
        binding = deployment.configure_v3r1_import(self.study, phase='validation', receipt_sha256=record['sha256'])
        self.addCleanup(lambda: [deployment.os.environ.pop(name, None) for name in (
            'SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT', 'SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT',
            'SEMUAV_V3R1_VALIDATION_IMPORT_RECEIPT_SHA256')])
        self.assertEqual(binding['sha256'], pinned['sha256'])
        self.assertEqual(deployment.os.environ['SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT'], '1')
        self.assertIsNone(deployment.configure_v3r1_import(self.study, phase='test', receipt_sha256=record['sha256']))
        self.assertNotIn('SEMUAV_ENABLE_V3R1_VALIDATION_IMPORT', deployment.os.environ)
        Path(pinned['path']).write_bytes(b'{}')
        with self.assertRaises(deployment.DeploymentError):
            deployment.verify_source(self.paths['release'], record['sha256'], study_root=self.study)

    def test_v3r1_import_refuses_test_access_and_changed_artifacts(self):
        root = self.paths['v3r1_run']
        write(root / 'test_access_ledger.json', b'{}')
        with self.assertRaises(deployment.DeploymentError):
            self._deploy()
        (root / 'test_access_ledger.json').unlink()
        write(root / 'stages' / '03_static_transport_seed_1_shard_00000' / (digest(b'03_static_transport_seed_1_shard_00000') + '.csv.gz'), b'tampered')
        with self.assertRaises(deployment.DeploymentError):
            self._deploy()

    def test_changed_predecessor_and_recovery_stop_before_deployment(self):
        old_manifest = self.predecessor_files['PREDECESSOR_MANIFEST_SHA256'][0]
        old_manifest.write_bytes(b'tampered predecessor')
        with self.assertRaises(deployment.DeploymentError):
            self._deploy()
        self.assertFalse(self.paths['release'].exists())
        old_manifest.write_bytes(self.predecessor_files['PREDECESSOR_MANIFEST_SHA256'][1])
        recovery = json.loads(self.paths['recovery'].read_bytes())
        raw_path = Path(recovery['predecessor_run_root']) / recovery['artifacts'][0]['relative_path']
        raw_path.write_bytes(b'tampered raw E1')
        with self.assertRaises(deployment.DeploymentError):
            self._deploy()
        self.assertFalse(self.paths['release'].exists())

    def test_archive_digest_and_unsealed_source_extra_rejected(self):
        with self.assertRaises(deployment.DeploymentError):
            deployment.deploy(self.archive, '0' * 64, study_root=self.study)
        record = self._deploy()
        write(self.paths['release'] / 'injected.py', b'pass\n')
        with self.assertRaises(deployment.DeploymentError):
            deployment.verify_source(self.paths['release'], record['sha256'], study_root=self.study)

    def test_symlink_source_rejected(self):
        record = self._deploy()
        (self.paths['release'] / 'injected.py').symlink_to(self.paths['receipt'])
        with self.assertRaises(deployment.DeploymentError):
            deployment.verify_source(self.paths['release'], record['sha256'], study_root=self.study)


class CandidateVerificationTests(unittest.TestCase):
    def test_exact_candidate_provenance_and_package_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            study = Path(temporary)
            paths = deployment._paths(study)
            candidate = paths['candidate']
            source = study / 'releases/release_20260912_visdrone_paper_v2r3' / deployment.PACKAGE_NAME
            manifest = write(source / 'RELEASE_MANIFEST.json', b'{}\n')
            checksum = write(source / 'RELEASE_MANIFEST_SHA256.txt', b'fixture checksums\n')
            requirements = write(source / 'requirements.txt', b'fixture-package==1.0\n')
            freeze = write(study / 'private/candidate/pip_freeze.txt', b'fixture-package==1.0\n')
            candidate.mkdir(parents=True)
            (candidate / 'bin').mkdir()
            (candidate / 'bin/python').symlink_to(sys.executable)
            ready = {
                'schema_version': 'semuav.visdrone-inference-candidate-environment.v1',
                'status': 'READY_FOR_DEVELOPMENT_SMOKE_ONLY', 'candidate_only': True,
                'candidate_environment': str(candidate), 'scientific_results_authorized': False,
                'validation_or_test_data_read': False, 'legacy_environment_invoked': False,
                'canonical_release_modified': False,
                'source_release': {'path': str(source), 'release_id': 'release_20260912_visdrone_paper_v2r3',
                                   'run_name': 'semuav_visdrone_v2r3', 'manifest_sha256': digest(manifest.read_bytes()),
                                   'checksum_inventory_sha256': digest(checksum.read_bytes()),
                                   'requirements_sha256': digest(requirements.read_bytes())},
                'artifacts': {'pip_freeze': deployment._record(freeze)},
            }
            ready_path = write(candidate / 'SEMUAV_CANDIDATE_READY.json', json_bytes(ready))
            prior = {'candidate_environment': {'path': str(candidate), 'ready_receipt': str(ready_path),
                                              'ready_receipt_sha256': digest(ready_path.read_bytes()),
                                              'pip_freeze_sha256': digest(freeze.read_bytes())}}
            with mock.patch.multiple(deployment, BASE_MANIFEST_SHA256=digest(manifest.read_bytes()),
                                     BASE_CHECKSUM_SHA256=digest(checksum.read_bytes()),
                                     REQUIREMENTS_SHA256=digest(requirements.read_bytes()),
                                     PIP_FREEZE_SHA256=digest(freeze.read_bytes())):
                responses = [mock.Mock(stdout=str(candidate) + '\n'), mock.Mock(stdout=freeze.read_bytes())]
                with mock.patch.object(deployment.subprocess, 'run', side_effect=responses) as process:
                    facts, observed_ready = deployment._candidate_facts(study, prior)
                    self.assertEqual(observed_ready, ready)
                    self.assertEqual(facts['pip_freeze_sha256'], digest(freeze.read_bytes()))
                    self.assertEqual(process.call_args_list[1].args[0][-4:], ['-m', 'pip', 'freeze', '--all'])
                responses = [mock.Mock(stdout=str(candidate) + '\n'), mock.Mock(stdout=b'changed-package==2.0\n')]
                with mock.patch.object(deployment.subprocess, 'run', side_effect=responses):
                    with self.assertRaisesRegex(deployment.DeploymentError, 'Installed distributions'):
                        deployment._candidate_facts(study, prior)

    def test_recovery_cannot_be_enabled_before_validation(self):
        for phase in ('prepare', 'test', 'report', 'embedded'):
            with mock.patch.object(deployment, 'verify_recovery') as read_raw:
                with self.assertRaises(deployment.DeploymentError):
                    deployment.configure_recovery(phase=phase)
                read_raw.assert_not_called()


if __name__ == '__main__':
    unittest.main()

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from PIL import Image

from semuav.registry import atomic_write_json, hash_file, hash_object, source_tree_hash
from semuav.visdrone_models import (
    OFFICIAL_COMPACT_MODEL_REPOS,
    OFFICIAL_MODEL_BACKENDS,
    OFFICIAL_UPPER_BOUND_ID,
    OFFICIAL_UPPER_BOUND_REPO,
    RUNTIME_SMOKE_ISOLATION_SCHEMA,
    RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
    RUNTIME_SMOKE_WORKER_RESULT_SCHEMA,
    _execute_runtime_smoke_entry,
    _launch_runtime_smoke_worker,
    _run_runtime_smoke_subprocesses,
    _worker_environment,
    require_runtime_smoke_pass,
    run_one_image_runtime_smoke,
)


ROOT = Path(__file__).resolve().parents[1]


class V2R9ThreeModelSubprocessSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.image = self.root / "frame.jpg"
        Image.new("RGB", (16, 12), (1, 2, 3)).save(self.image)
        self.lock = self.root / "MODEL_LOCK.json"
        self.lock.write_text('{"fixture":true}\n', encoding="utf-8")
        self.request = SimpleNamespace(
            observation_id="visdrone:det_train:subprocess-smoke",
            image_path=str(self.image),
            mission="visdrone_urban",
            capture_time_us=123,
            telemetry={"battery_fraction": 0.9},
            detector_proposals=(),
        )
        compact = [
            {
                "model_id": model_id,
                "repo": repo,
                "resolved_revision": character * 40,
                "role": "compact",
                "backend": OFFICIAL_MODEL_BACKENDS[model_id],
                "dtype": "bfloat16",
            }
            for (model_id, repo), character in zip(
                OFFICIAL_COMPACT_MODEL_REPOS.items(), ("a", "b")
            )
        ]
        self.entries = compact + [{
            "model_id": OFFICIAL_UPPER_BOUND_ID,
            "repo": OFFICIAL_UPPER_BOUND_REPO,
            "resolved_revision": "c" * 40,
            "role": "upper_bound",
            "backend": "qwen3_vl",
            "dtype": "bfloat16",
        }]
        self.input_receipt = {
            "input_contract": "semuav.visdrone-proposal-decision.v2",
            "request_sha256": "e" * 64,
            "detector_proposals_sha256": "f" * 64,
            "detector_proposal_count": 0,
        }

    @staticmethod
    def _successful_launch(
        input_path, result_path, stdout_path, stderr_path, **kwargs
    ):
        del kwargs
        worker_input = json.loads(input_path.read_text(encoding="utf-8"))
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(b"")
        row = {
            "model_id": worker_input["model_id"],
            "status": "PASS",
            "error": None,
        }
        atomic_write_json(result_path, {
            "schema_version": RUNTIME_SMOKE_WORKER_RESULT_SCHEMA,
            "worker_input_sha256": hash_file(input_path),
            "model_id": worker_input["model_id"],
            "model_entry_sha256": worker_input["model_entry_sha256"],
            "model_lock_sha256": worker_input["model_lock_sha256"],
            "code_sha256": worker_input["code_sha256"],
            "worker_script_sha256": worker_input["worker_script_sha256"],
            "python_executable": worker_input["python_executable"],
            "python_executable_sha256": worker_input[
                "python_executable_sha256"
            ],
            "worker_environment_sha256": worker_input[
                "worker_environment_sha256"
            ],
            "row": row,
        })
        return {
            "pid": 100,
            "returncode": 0,
            "signal": None,
            "timed_out": False,
            "wall_ms": 1.0,
            "argv_sha256": "1" * 64,
            "stdout_bytes": 0,
            "stdout_sha256": hash_file(stdout_path),
            "stderr_bytes": 0,
            "stderr_sha256": hash_file(stderr_path),
        }

    def _run(self, launcher):
        with mock.patch(
            "semuav.visdrone_models._launch_runtime_smoke_worker",
            side_effect=launcher,
        ):
            return _run_runtime_smoke_subprocesses(
                self.entries,
                self.request,
                output_path=self.root / "MODEL_RUNTIME_SMOKE.json",
                lock_path=self.lock,
                model_lock_sha256=hash_file(self.lock),
                code_sha256=source_tree_hash(ROOT),
                image_path=self.image.resolve(),
                image_sha256=hash_file(self.image),
                input_receipt=self.input_receipt,
                prompt_text="FROZEN PROMPT",
                device="cuda:0",
                dtype="bfloat16",
                max_new_tokens=256,
            )

    def test_crashed_middle_worker_does_not_skip_later_model(self) -> None:
        calls = []

        def launcher(input_path, result_path, stdout_path, stderr_path, **kwargs):
            model_id = json.loads(input_path.read_text())["model_id"]
            calls.append(model_id)
            if model_id != "smolvlm2_2b":
                return self._successful_launch(
                    input_path, result_path, stdout_path, stderr_path, **kwargs
                )
            stdout_path.write_bytes(b"")
            stderr_path.write_bytes(b"device-side assert")
            return {
                "pid": 103,
                "returncode": -6,
                "signal": 6,
                "timed_out": False,
                "wall_ms": 2.0,
                "argv_sha256": "2" * 64,
                "stdout_bytes": 0,
                "stdout_sha256": hash_file(stdout_path),
                "stderr_bytes": stderr_path.stat().st_size,
                "stderr_sha256": hash_file(stderr_path),
            }

        rows, isolation = self._run(launcher)
        self.assertEqual(calls, [entry["model_id"] for entry in self.entries])
        by_model = {row["model_id"]: row for row in rows}
        self.assertEqual(by_model["smolvlm2_2b"]["status"], "ERROR")
        self.assertEqual(by_model["smolvlm2_2b"]["worker_signal"], 6)
        self.assertEqual(
            by_model[OFFICIAL_UPPER_BOUND_ID]["status"], "PASS"
        )
        self.assertEqual(
            isolation["strategy"], "fresh_exec_per_locked_model"
        )
        self.assertNotIn("moondream_tokenizer", isolation)
        self.assertNotIn("moondream_tokenizer_sha256", isolation)

    def test_invalid_worker_attestation_fails_only_that_row(self) -> None:
        calls = []

        def launcher(input_path, result_path, stdout_path, stderr_path, **kwargs):
            launch = self._successful_launch(
                input_path, result_path, stdout_path, stderr_path, **kwargs
            )
            model_id = json.loads(input_path.read_text())["model_id"]
            calls.append(model_id)
            if model_id == "smolvlm2_2b":
                payload = json.loads(result_path.read_text())
                payload["code_sha256"] = "0" * 64
                atomic_write_json(result_path, payload)
            return launch

        rows, _ = self._run(launcher)
        self.assertEqual(len(calls), 3)
        by_model = {row["model_id"]: row for row in rows}
        self.assertEqual(by_model["smolvlm2_2b"]["status"], "ERROR")
        self.assertEqual(by_model["qwen3_vl_2b"]["status"], "PASS")
        self.assertEqual(by_model[OFFICIAL_UPPER_BOUND_ID]["status"], "PASS")

    def test_passing_report_reverifies_every_worker_artifact(self) -> None:
        rows, isolation = self._run(self._successful_launch)
        report_path = self.root / "PASS.json"
        atomic_write_json(report_path, {
            "schema_version": RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
            "all_passed": True,
            "model_lock": str(self.lock),
            "models": rows,
            "execution_isolation": isolation,
        })
        with mock.patch(
            "semuav.visdrone_models._read_model_lock",
            return_value=(self.lock, {"models": self.entries}),
        ):
            require_runtime_smoke_pass(report_path)

        tampered_report = json.loads(report_path.read_text(encoding="utf-8"))
        tampered_report["all_passed"] = False
        atomic_write_json(report_path, tampered_report)
        with mock.patch(
            "semuav.visdrone_models._read_model_lock",
            return_value=(self.lock, {"models": self.entries}),
        ), self.assertRaisesRegex(Exception, "all_passed|exact 3/3"):
            require_runtime_smoke_pass(report_path)

        tampered_report["all_passed"] = True
        tampered_report["models"][0]["worker_returncode"] = 1
        atomic_write_json(report_path, tampered_report)
        with mock.patch(
            "semuav.visdrone_models._read_model_lock",
            return_value=(self.lock, {"models": self.entries}),
        ), self.assertRaisesRegex(Exception, "attestation"):
            require_runtime_smoke_pass(report_path)

        tampered_report["models"] = rows
        atomic_write_json(report_path, tampered_report)

        stdout = Path(isolation["artifacts"][0]["stdout"]["path"])
        stdout.write_bytes(b"tampered")
        with mock.patch(
            "semuav.visdrone_models._read_model_lock",
            return_value=(self.lock, {"models": self.entries}),
        ), self.assertRaisesRegex(
            Exception, "artifact changed|artifact manifest changed"
        ):
            require_runtime_smoke_pass(report_path)

    def test_launcher_uses_isolated_absolute_exec_without_shell(self) -> None:
        worker_script = ROOT / "semuav" / "runtime_smoke_worker.py"
        input_path = self.root / "input.json"
        result_path = self.root / "result.json"
        stdout_path = self.root / "stdout.log"
        stderr_path = self.root / "stderr.log"
        input_path.write_text("{}\n", encoding="utf-8")
        process = SimpleNamespace(pid=123, returncode=0, wait=lambda timeout: 0)
        with mock.patch(
            "semuav.visdrone_models.subprocess.Popen", return_value=process
        ) as popen:
            launch = _launch_runtime_smoke_worker(
                input_path,
                result_path,
                stdout_path,
                stderr_path,
                worker_script=worker_script,
                environment={"SAFE": "1"},
                timeout_seconds=10,
            )
        argv = popen.call_args.args[0]
        options = popen.call_args.kwargs
        self.assertTrue(Path(argv[0]).is_absolute())
        self.assertEqual(argv[1:3], ["-I", "-B"])
        self.assertEqual(Path(argv[3]), worker_script)
        self.assertIs(options["shell"], False)
        self.assertIs(options["start_new_session"], True)
        self.assertEqual(options["stdin"], -3)  # subprocess.DEVNULL
        self.assertEqual(launch["returncode"], 0)

    def test_worker_environment_removes_import_injection(self) -> None:
        contaminated = {
            "PYTHONPATH": "/attacker",
            "LD_PRELOAD": "/attacker.so",
            "BASH_ENV": "/attacker.sh",
            "SEMUAV_MOONDREAM_REVISION": "f" * 40,
            "SEMUAV_MOONDREAM_TOKENIZER_LOCK": "/legacy/tokenizer-lock.json",
        }
        with mock.patch.dict(os.environ, contaminated, clear=False):
            environment, attested = _worker_environment()
        for name in contaminated:
            self.assertNotIn(name, environment)
            if name.startswith("SEMUAV_MOONDREAM"):
                self.assertNotIn(name, attested)
            else:
                self.assertIsNone(attested[name])
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
        self.assertEqual(
            hash_object(attested), hash_object(dict(attested))
        )

    def test_active_smoke_rejects_historical_moondream_tokenizer_arguments(self) -> None:
        verified = {
            "models": self.entries,
            "model_set_sha256": "9" * 64,
        }
        for argument in (
            {"moondream_tokenizer_lock_path": self.root / "legacy.json"},
            {"moondream_tokenizer_offline_checker": lambda *args: None},
        ):
            with self.subTest(argument=next(iter(argument))), mock.patch(
                "semuav.visdrone_models.verify_huggingface_model_lock",
                return_value=verified,
            ), self.assertRaisesRegex(
                Exception, "v2r13 runtime excludes Moondream",
            ):
                run_one_image_runtime_smoke(
                    self.lock,
                    self.request,
                    output_path=self.root / "NEVER_WRITTEN.json",
                    **argument,
                )

    def test_disposable_worker_never_calls_cuda_cleanup_after_error(self) -> None:
        cleanup_calls = []

        class FailingBackend:
            def infer(self, request, prompt, max_new_tokens):
                del request, prompt, max_new_tokens
                raise RuntimeError("simulated device-side assert")

        fake_torch = SimpleNamespace(cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: cleanup_calls.append("called"),
        ))
        with mock.patch.dict("sys.modules", {"torch": fake_torch}):
            row, cleanup_error = _execute_runtime_smoke_entry(
                self.entries[2],
                self.request,
                image_path=self.image.resolve(),
                image_sha256=hash_file(self.image),
                input_receipt=self.input_receipt,
                prompt_text="FROZEN PROMPT",
                device="cuda:0",
                dtype="bfloat16",
                max_new_tokens=256,
                expected_decision_context_sha256="0" * 64,
                expected_generation_policy_sha256="1" * 64,
                backend_factory=lambda entry, device, dtype: FailingBackend(),
                cleanup_cuda=False,
            )
        self.assertEqual(row["status"], "ERROR")
        self.assertIn("simulated device-side assert", row["error"])
        self.assertIsNone(cleanup_error)
        self.assertEqual(cleanup_calls, [])


if __name__ == "__main__":
    unittest.main()

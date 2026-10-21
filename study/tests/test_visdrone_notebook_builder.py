from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import re
import sys
import tempfile
import unittest
import zipfile
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import yaml


# Historical bootstrap contract: v3 orchestration has separate tests.
ROOT = Path(__file__).resolve().parents[1] / "legacy" / "v2r13"
BUILDER = ROOT / "build_visdrone_paper_notebook.py"
NOTEBOOK = ROOT / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
RUN_GUIDE = ROOT / "VISDRONE_PAPER_RUN_GUIDE.md"


APPROVED_PARAMETER_NAMES = (
    "PHASE",
    "EXECUTE_PHASE",
    "FREEZE_PROTOCOL",
    "AUTHORIZE_TEST_ONCE",
    "RUN_NAME",
    "RESUME",
    "DEVICE",
    "RUN_MODEL_SMOKE",
    "BUILD_NATIVE_RECEIVER_CACHE",
    "BUILD_FEATURE_CACHE",
)


def _normalized_parameter_source(source: str, expected_source: str) -> str:
    """Normalize only safe literal RHS edits to the ten approved controls."""

    def assignments(text: str, label: str) -> dict[str, ast.Assign]:
        try:
            module = ast.parse(text)
        except SyntaxError as exc:
            raise ValueError(f"{label} parameter cell is invalid Python") from exc
        result: dict[str, ast.Assign] = {}
        for statement in module.body:
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and statement.targets[0].id in APPROVED_PARAMETER_NAMES
            ):
                name = statement.targets[0].id
                if name in result:
                    raise ValueError(f"{label} repeats approved parameter {name}")
                result[name] = statement
        stored_names = [
            node.id for node in ast.walk(module)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id in APPROVED_PARAMETER_NAMES
        ]
        if (
            tuple(result) != APPROVED_PARAMETER_NAMES
            or sorted(stored_names) != sorted(APPROVED_PARAMETER_NAMES)
        ):
            raise ValueError(f"{label} approved parameter assignments changed")
        return result

    observed = assignments(source, "observed")
    expected = assignments(expected_source, "generated")
    boolean_names = {
        "EXECUTE_PHASE", "FREEZE_PROTOCOL", "AUTHORIZE_TEST_ONCE", "RESUME",
        "RUN_MODEL_SMOKE", "BUILD_NATIVE_RECEIVER_CACHE", "BUILD_FEATURE_CACHE",
    }
    for name, statement in observed.items():
        value = statement.value
        unchanged = ast.dump(value, include_attributes=False) == ast.dump(
            expected[name].value, include_attributes=False,
        )
        safe_literal = (
            isinstance(value, ast.Constant)
            and (
                (name in boolean_names and type(value.value) is bool)
                or (
                    name == "PHASE"
                    and value.value in {
                        "prepare", "validation", "test", "report", "embedded",
                    }
                )
                or (name == "RUN_NAME" and value.value == "semuav_visdrone_v2r13")
                or (name == "DEVICE" and value.value == "cuda:0")
            )
        )
        if not unchanged and not safe_literal:
            raise ValueError(f"observed parameter {name} has an unsafe right-hand side")

    encoded = source.encode("utf-8")
    lines = encoded.splitlines(keepends=True)
    replacements: list[tuple[int, int]] = []
    for statement in observed.values():
        value = statement.value
        start = sum(len(line) for line in lines[: value.lineno - 1]) + value.col_offset
        end = sum(len(line) for line in lines[: value.end_lineno - 1]) + value.end_col_offset
        replacements.append((start, end))
    for start, end in sorted(replacements, reverse=True):
        encoded = encoded[:start] + b"<APPROVED_PARAMETER_RHS>" + encoded[end:]
    return encoded.decode("utf-8")


def _authored_contract(
    notebook: dict[str, object], *, expected_parameter_source: str,
) -> dict[str, object]:
    """Ignore outputs/counts and normalize only approved parameter RHS edits."""

    contract = copy.deepcopy(notebook)
    metadata = contract.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("notebook metadata is malformed")
    for transient in ("language_info", "widgets", "signature"):
        metadata.pop(transient, None)
    parameter_cells = 0
    for cell in contract.get("cells", []):
        if not isinstance(cell, dict):
            raise ValueError("notebook contains a non-object cell")
        cell_metadata = cell.get("metadata", {})
        if not isinstance(cell_metadata, dict):
            raise ValueError("cell metadata is malformed")
        for transient in (
            "trusted", "execution", "ExecuteTime", "collapsed", "scrolled",
        ):
            cell_metadata.pop(transient, None)
        tags = cell_metadata.get("tags", [])
        raw_source = cell.get("source", "")
        if isinstance(raw_source, list):
            if any(not isinstance(part, str) for part in raw_source):
                raise ValueError("cell source list contains a non-string")
            raw_source = "".join(raw_source)
        if not isinstance(raw_source, str):
            raise ValueError("cell source is malformed")
        cell["source"] = raw_source
        if "parameters" in tags:
            parameter_cells += 1
            cell["source"] = _normalized_parameter_source(
                raw_source, expected_parameter_source,
            )
        if cell.get("cell_type") == "code":
            cell.pop("execution_count", None)
            cell.pop("outputs", None)
    if parameter_cells != 1:
        raise ValueError("notebook does not have one parameter cell")
    return contract


def _require_authored_contract(
    observed: dict[str, object], expected: dict[str, object],
) -> None:
    parameter_source = next(
        str(cell["source"])
        for cell in expected["cells"]  # type: ignore[index]
        if "parameters" in cell.get("metadata", {}).get("tags", [])
    )
    if _authored_contract(
        observed, expected_parameter_source=parameter_source,
    ) != _authored_contract(
        expected, expected_parameter_source=parameter_source,
    ):
        raise ValueError("notebook authored contract differs")


class VisDronePaperNotebookBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_builder = Path(directory) / BUILDER.name
            namespace = {
                "__file__": str(temporary_builder),
                "__name__": "visdrone_notebook_builder_test",
                "print": lambda *args, **kwargs: None,
            }
            source = BUILDER.read_text(encoding="utf-8")
            exec(compile(source, str(BUILDER), "exec"), namespace)
            cls.authenticated_v2r12_predecessor = namespace[
                "_v2r13_authenticated_predecessor_fixture"
            ]()
            cls.generated_text = (
                Path(directory) / NOTEBOOK.name
            ).read_text(encoding="utf-8")
        cls.notebook = json.loads(cls.generated_text)

    def test_checked_in_notebook_is_exact_builder_output(self) -> None:
        self.assertTrue(NOTEBOOK.is_file())
        checked_text = NOTEBOOK.read_text(encoding="utf-8")
        if os.environ.get("SEMUAV_LIVE_NOTEBOOK_SELFTEST") == "1":
            _require_authored_contract(json.loads(checked_text), self.notebook)
        else:
            self.assertEqual(checked_text, self.generated_text)

        # Runtime state and safe literal control edits are the only tolerated
        # live-notebook differences; authored statements remain immutable.
        live = json.loads(self.generated_text)
        parameter_cell = next(
            cell for cell in live["cells"]
            if "parameters" in cell.get("metadata", {}).get("tags", [])
        )
        parameter_cell["source"] = parameter_cell["source"].replace(
            "EXECUTE_PHASE = False", "EXECUTE_PHASE = True", 1,
        ).replace(
            'os.environ.get("SEMUAV_VISDRONE_PHASE", "prepare").strip().lower()',
            '"validation"', 1,
        )
        parameter_cell["execution_count"] = 7
        parameter_cell["outputs"] = [{"output_type": "stream", "text": "runtime\n"}]
        live["metadata"]["language_info"] = {"version": "runtime-only"}
        live["metadata"]["widgets"] = {"application/vnd.jupyter.widget-state+json": {}}
        live["metadata"]["signature"] = "legacy-runtime-signature"
        parameter_cell["metadata"].update({
            "trusted": True,
            "execution": {"iopub.status.busy": "runtime"},
            "ExecuteTime": {"start_time": "runtime"},
            "collapsed": False,
            "scrolled": True,
        })
        _require_authored_contract(live, self.notebook)
        list_source_notebook = copy.deepcopy(live)
        for cell in list_source_notebook["cells"]:
            cell["source"] = str(cell["source"]).splitlines(keepends=True)
        _require_authored_contract(list_source_notebook, self.notebook)

        rejected_sources = {
            "injected statement": parameter_cell["source"] + "\nprint('injected')\n",
            "nonliteral RHS": parameter_cell["source"].replace(
                "EXECUTE_PHASE = True", "EXECUTE_PHASE = bool(1)", 1,
            ),
            "changed target": parameter_cell["source"].replace(
                "BUILD_FEATURE_CACHE = False", "UNLISTED_CONTROL = False", 1,
            ),
            "reordered assignments": parameter_cell["source"].replace(
                "EXECUTE_PHASE = True\nFREEZE_PROTOCOL = False",
                "FREEZE_PROTOCOL = False\nEXECUTE_PHASE = True", 1,
            ),
        }
        for source in rejected_sources.values():
            changed = json.loads(json.dumps(live))
            next(
                cell for cell in changed["cells"]
                if "parameters" in cell.get("metadata", {}).get("tags", [])
            )["source"] = source
            with self.assertRaises(ValueError):
                _require_authored_contract(changed, self.notebook)

        structural_mutations = {
            "cell id": lambda payload: payload["cells"][0].__setitem__(
                "id", "changed-cell-id"
            ),
            "arbitrary cell metadata": lambda payload: payload["cells"][0][
                "metadata"
            ].__setitem__("arbitrary", True),
            "attachment": lambda payload: payload["cells"][0].__setitem__(
                "attachments", {"evidence.txt": {"text/plain": "changed"}}
            ),
            "arbitrary notebook metadata": lambda payload: payload[
                "metadata"
            ].__setitem__("arbitrary", True),
            "top-level extra": lambda payload: payload.__setitem__(
                "unexpected", True
            ),
        }
        for label, mutate in structural_mutations.items():
            with self.subTest(mutation=label):
                changed = copy.deepcopy(live)
                mutate(changed)
                with self.assertRaises(ValueError):
                    _require_authored_contract(changed, self.notebook)

    def test_delivery_is_clean_and_every_code_cell_compiles(self) -> None:
        identifiers: set[str] = set()
        for index, cell in enumerate(self.notebook["cells"]):
            cell_id = cell.get("id")
            self.assertIsInstance(cell_id, str)
            self.assertNotIn(cell_id, identifiers)
            identifiers.add(cell_id)
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell.get("execution_count"), f"cell {index}")
            self.assertEqual(cell.get("outputs"), [], f"cell {index}")
            compile(str(cell.get("source", "")), f"notebook-cell-{index}", "exec")

    def test_scientific_phase_entrypoints_are_imported_and_resolve(self) -> None:
        import_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "from semuav.visdrone_runner import" in str(cell.get("source", ""))
        )
        namespace = {
            "PROJECT_ROOT": ROOT, "PHASE": "validation",
            "EXECUTE_PHASE": False,
            "V2R13_PREIMPORT_SOURCE_VERIFIED": True,
        }
        exec(compile(import_source, "notebook-import-cell", "exec"), namespace)
        for name in (
            "prepare_visdrone_bundle",
            "run_validation_perception_and_frontier",
            "run_validation_network_and_controller",
            "run_confirmatory_test_once",
            "recompute_hypotheses_from_frozen_results",
            "evaluate_optional_embedded_h5",
            "experiment6_receiver_robustness",
            "experiment6_registered_campaign",
            "experiment7_readiness",
            "require_exclusive_physical_gpu",
        ):
            with self.subTest(name=name):
                self.assertTrue(callable(namespace.get(name)), name)
        self.assertIs(namespace.get("hashlib"), hashlib)

    def test_parameter_cell_exposes_all_irreversible_controls(self) -> None:
        parameter_cells = [
            cell for cell in self.notebook["cells"]
            if "parameters" in cell.get("metadata", {}).get("tags", [])
        ]
        self.assertEqual(len(parameter_cells), 1)
        source = str(parameter_cells[0]["source"])
        for token in (
            "PHASE =", "EXECUTE_PHASE = False", "FREEZE_PROTOCOL = False",
            "AUTHORIZE_TEST_ONCE = False", "RUN_NAME =", "RESUME = True",
            "DEVICE =", "RUN_MODEL_SMOKE = False",
            '"IPYTHONDIR"',
            "prepare", "validation", "test", "report", "embedded",
        ):
            self.assertIn(token, source)
        for prohibited in (
            "PREPARE_DATA", "PREPARE_DETECTOR_DATA", "PREFETCH_MODELS",
            "TRAIN_FEATURE_HEAD",
        ):
            self.assertNotIn(prohibited, source)
        for namespace in (
            '"JUPYTER_DATA_DIR": STUDY_STORAGE_ROOT / ".cache" / "jupyter" / "data"',
            '"config-v2r13"',
            '"runtime-v2r13"',
            '"ipython-v2r13"',
            '"matplotlib-v2r13"',
            '"modules-v2r13"',
            "V2R13_PREIMPORT_SOURCE_VERIFIED = False",
            "V2R13_BOOTSTRAP_PHASE = PHASE",
            "V2R13_BOOTSTRAP_EXECUTE_PHASE = EXECUTE_PHASE",
            "V2R13_PREIMPORT_LIVE_SOURCE_INVENTORY_VERIFIED",
            "preimported_release_modules",
            "source payload bytes changed",
            "source payload set differs from release inventory",
            "live notebook differs outside approved parameter RHS values",
            "zipfile.ZipFile(io.BytesIO(archive_bytes), \"r\")",
            'b"<APPROVED_PARAMETER_RHS>"',
            "sys.executable != V2R13_EXPECTED_CANDIDATE_PYTHON_TEXT",
            "sys.prefix != V2R13_EXPECTED_CANDIDATE_PREFIX_TEXT",
            "sys.flags.isolated != 1",
            "sys.flags.dont_write_bytecode != 1",
            'if DEVICE != "cuda:0":',
            "official v2r13 notebook requires DEVICE=cuda:0",
            '"PYTHONHOME"',
        ):
            self.assertIn(namespace, source)
        self.assertLess(source.index("import sys"), source.index("import ast"))
        kernel_identity_gate = source.index(
            "sys.executable != V2R13_EXPECTED_CANDIDATE_PYTHON_TEXT"
        )
        self.assertLess(
            kernel_identity_gate,
            source.index("V2R13_EXPECTED_RELEASE_ROOT_TEXT"),
        )
        self.assertLess(
            source.index("sys.path[:]") , source.index("import ast"),
        )
        self.assertLess(kernel_identity_gate, source.index("sys.path[:]"))
        self.assertNotIn(
            "if EXECUTE_PHASE:",
            source[:source.index("V2R13_PREIMPORT_SOURCE_VERIFIED = True")],
        )
        parameter_module = ast.parse(source)
        top_level_assignments = [
            statement for statement in parameter_module.body
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ]
        authenticated_line = next(
            statement.lineno for statement in top_level_assignments
            if statement.targets[0].id == "V2R13_PREIMPORT_SOURCE_VERIFIED"
            and isinstance(statement.value, ast.Constant)
            and statement.value.value is True
        )
        approved_assignments = [
            statement for statement in top_level_assignments
            if statement.targets[0].id in APPROVED_PARAMETER_NAMES
        ]
        self.assertEqual(
            [statement.targets[0].id for statement in approved_assignments],
            list(APPROVED_PARAMETER_NAMES),
        )
        self.assertTrue(all(
            statement.lineno > authenticated_line
            for statement in approved_assignments
        ))
        self.assertIn('"semuav_visdrone_v2r13"', source)
        run_name_assignment = source.index(
            'RUN_NAME = os.environ.get("SEMUAV_RUN_NAME", "semuav_visdrone_v2r13")'
        )
        exact_identity_gate = source.index(
            'if RUN_NAME != "semuav_visdrone_v2r13":', run_name_assignment,
        )
        cache_creation = source.index("for variable, default_path in cache_defaults.items()")
        self.assertLess(run_name_assignment, exact_identity_gate)
        self.assertLess(exact_identity_gate, cache_creation)
        self.assertNotIn("RUN_NAME.startswith(", source)
        kernelspec = self.notebook["metadata"]["kernelspec"]
        self.assertEqual(kernelspec["name"], "semuav-vlm-20260921-v2r13")
        self.assertEqual(
            kernelspec["display_name"], "Python (SemUAV-VLM 2026-09-21 v2r13)",
        )

        expected_python = (
            "/data/Lane Detection Datasets/musta/semuav_vlm/venvs/"
            "release_20260912_visdrone_inference_candidate_v2r3/bin/python"
        )
        expected_prefix = str(Path(expected_python).parent.parent)
        flag_values = {
            name: getattr(sys.flags, name)
            for name in dir(sys.flags)
            if not name.startswith("_")
        }
        flag_values.update(isolated=1, dont_write_bytecode=1)
        isolated_flags = SimpleNamespace(**flag_values)

        def execute_parameter(source_text, environment, *, modules=None):
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(sys, "path", list(sys.path)),
                mock.patch.object(sys, "executable", expected_python),
                mock.patch.object(sys, "prefix", expected_prefix),
                mock.patch.object(sys, "flags", isolated_flags),
                mock.patch.object(sys, "dont_write_bytecode", True),
            ):
                if modules is None:
                    exec(compile(source_text, "parameter-cell", "exec"), {})
                else:
                    with mock.patch.dict(sys.modules, modules, clear=False):
                        exec(compile(source_text, "parameter-cell", "exec"), {})

        environment = {
            "PYTHONNOUSERSITE": "1",
            "SEMUAV_PROJECT_ROOT": "/not/reached",
            "SEMUAV_STUDY_ROOT": "/not/reached",
        }
        with self.assertRaisesRegex(
            ValueError, "SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256",
        ):
            execute_parameter(source, environment)

        preimport_environment = {
            **environment,
            "SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256": "0" * 64,
        }
        with self.assertRaisesRegex(ValueError, "release modules were imported"):
            execute_parameter(
                source, preimport_environment, modules={"semuav": object()},
            )

        with mock.patch.object(sys, "executable", "/usr/bin/python3"):
            with self.assertRaisesRegex(RuntimeError, "exact isolated v2r13"):
                exec(compile(source, "parameter-cell", "exec"), {})

    def test_preimport_gate_cannot_be_bypassed_by_late_parameter_toggle(self) -> None:
        import_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "def _require_v2r13_preimport_source_gate" in str(
                cell.get("source", "")
            )
        )
        module = ast.parse(import_source)
        guard = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_require_v2r13_preimport_source_gate"
        )
        guard_module = ast.Module(body=[guard], type_ignores=[])
        controls = {
            "PHASE": "validation",
            "EXECUTE_PHASE": True,
            "FREEZE_PROTOCOL": False,
            "AUTHORIZE_TEST_ONCE": False,
            "RUN_NAME": "semuav_visdrone_v2r13",
            "RESUME": True,
            "DEVICE": "cuda:0",
            "RUN_MODEL_SMOKE": False,
            "BUILD_NATIVE_RECEIVER_CACHE": False,
            "BUILD_FEATURE_CACHE": False,
        }
        snapshot = tuple(
            (name, type(value).__name__, value)
            for name, value in controls.items()
        )
        namespace = {
            **controls,
            "V2R13_PREIMPORT_SOURCE_VERIFIED": False,
            "V2R13_BOOTSTRAP_EXECUTE_PHASE": False,
            "V2R13_BOOTSTRAP_PHASE": "prepare",
            "V2R13_APPROVED_CONTROL_SNAPSHOT": snapshot,
        }
        exec(compile(guard_module, "preimport-guard", "exec"), namespace)
        with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
            namespace["_require_v2r13_preimport_source_gate"]()
        namespace.update({
            "V2R13_PREIMPORT_SOURCE_VERIFIED": True,
            "V2R13_BOOTSTRAP_EXECUTE_PHASE": True,
            "V2R13_BOOTSTRAP_PHASE": "validation",
        })
        namespace["_require_v2r13_preimport_source_gate"]()
        changed_values = {
            "PHASE": "test",
            "EXECUTE_PHASE": 1,
            "FREEZE_PROTOCOL": True,
            "AUTHORIZE_TEST_ONCE": True,
            "RUN_NAME": "semuav_visdrone_other",
            "RESUME": False,
            "DEVICE": "cpu",
            "RUN_MODEL_SMOKE": True,
            "BUILD_NATIVE_RECEIVER_CACHE": True,
            "BUILD_FEATURE_CACHE": True,
        }
        for name, changed in changed_values.items():
            with self.subTest(control=name):
                original = namespace[name]
                namespace[name] = changed
                with self.assertRaisesRegex(RuntimeError, "Restart the kernel"):
                    namespace["_require_v2r13_preimport_source_gate"]()
                namespace[name] = original

    def test_preimport_inventory_gate_covers_source_config_and_prompt_drift(self) -> None:
        parameter_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "parameters" in cell.get("metadata", {}).get("tags", [])
        )
        for token in (
            "for relative_text, record in records.items():",
            "if relative_text == notebook_relative:",
            "payload = read_exact_regular(",
            'len(payload) != record["bytes"]',
            'hashlib.sha256(payload).hexdigest() != record["sha256"]',
            "source payload bytes changed",
        ):
            self.assertIn(token, parameter_source)
        self.assertNotIn(
            'relative_text.startswith(("semuav/", "configs/", "prompts/"))',
            parameter_source,
        )
        for relative in (
            "semuav/visdrone_workflow.py",
            "configs/visdrone_paper.yaml",
            "prompts/visdrone_proposal_decision_v2.txt",
        ):
            with self.subTest(relative=relative):
                sealed = (relative + "\n").encode("utf-8")
                record = {
                    "bytes": len(sealed),
                    "sha256": hashlib.sha256(sealed).hexdigest(),
                }
                changed = bytearray(sealed)
                changed[0] ^= 1
                changed_bytes = bytes(changed)
                self.assertEqual(len(changed_bytes), record["bytes"])
                self.assertNotEqual(
                    hashlib.sha256(changed_bytes).hexdigest(), record["sha256"],
                )

    def test_preimport_bootstrap_end_to_end_rejects_representative_drift(self) -> None:
        parameter_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "parameters" in cell.get("metadata", {}).get("tags", [])
        )
        with tempfile.TemporaryDirectory() as directory:
            study = Path(directory).resolve() / "study"
            release = (
                study / "releases" / "release_20260921_visdrone_paper_v2r13"
                / "SemUAV-VLM-VisDrone-Paper-Study"
            )
            authenticated_v2r12_predecessor = {
                key: value.replace("/data/Lane Detection Datasets/musta/semuav_vlm", str(study))
                if isinstance(value, str) else copy.deepcopy(value)
                for key, value in self.authenticated_v2r12_predecessor.items()
            }
            archive_path = (
                study / "incoming"
                / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"
            )
            receipt_path = (
                study / "private" / "visdrone_v2r13_deployment"
                / "DEPLOYMENT_READY.json"
            )
            recovery_path = (
                study / "private" / "visdrone_v2r10_deployment"
                / "V2R9_EXP1_RECOVERY_SOURCE.json"
            )
            source_predecessor_release = (
                study / "releases"
                / "release_20260921_visdrone_paper_v2r10"
                / "SemUAV-VLM-VisDrone-Paper-Study"
            )
            source_predecessor_archive = (
                study / "incoming"
                / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r10.zip"
            )
            source_predecessor_deployer = (
                study / "incoming"
                / "deploy_semuav_visdrone_v2r10_20260921.sh"
            )
            source_predecessor_receipt = (
                study / "private" / "visdrone_v2r10_deployment"
                / "DEPLOYMENT_READY.json"
            )
            immediate_predecessor_release = (
                study / "releases"
                / "release_20260921_visdrone_paper_v2r11"
                / "SemUAV-VLM-VisDrone-Paper-Study"
            )
            immediate_predecessor_archive = (
                study / "incoming"
                / "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r11.zip"
            )
            immediate_predecessor_deployer = (
                study / "incoming"
                / "deploy_semuav_visdrone_v2r11_20260921.sh"
            )
            immediate_predecessor_receipt = (
                study / "private" / "visdrone_v2r11_deployment"
                / "DEPLOYMENT_READY.json"
            )
            source_predecessor = {
                "release_id": "release_20260921_visdrone_paper_v2r10",
                "run_name": "semuav_visdrone_v2r10",
                "release_path": str(source_predecessor_release),
                "archive_path": str(source_predecessor_archive),
                "archive_sha256": (
                    "f35723efcbda801bc1c3a8796e2fa6603b8d8bcacdcf66b7186df8d7a314af6a"
                ),
                "release_manifest_sha256": (
                    "99323dd6109eca6d4942a120a3bb383f077d4a8f95869169f57f75e57574ac3c"
                ),
                "checksum_inventory_sha256": (
                    "53bd76f3ffd5510b0d8c8cef6bfb82a6bdf63562c6dadd08c849d224e3513649"
                ),
                "requirements_sha256": (
                    "028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5"
                ),
                "deployer_path": str(source_predecessor_deployer),
                "deployer_sha256": (
                    "e8b49ea192b955bf07f8b53a0f6130f3eb0bd182c3a1ea28ede6f10e89bf79cf"
                ),
                "deployment_receipt": str(source_predecessor_receipt),
                "deployment_receipt_sha256": (
                    "95a64dad33ae9fa21974b8ba629bc9ec972ad989245bef5b38df42b00b2fb2e3"
                ),
                "payload_file_count": 108,
                "archive_member_count": 110,
                "payload_total_uncompressed_bytes": 4009068,
                "builder_sha256": (
                    "09478e1f93768d80b589110e45edc05fa0c1cb3cd63bbf6e6e29ed15a27a73ab"
                ),
                "notebook_sha256": (
                    "f35bde4c03536b97bf82ab719ad5a516225bcc321d5a878971410b4e86a16108"
                ),
                "experiment1_recovery_receipt": str(recovery_path),
                "experiment1_recovery_receipt_sha256": (
                    "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608"
                ),
                "experiment1_recovery_receipt_bytes": 37416,
                "experiment1_recovery_artifact_count": 132,
                "experiment1_recovery_row_count": 4152,
                "active_model_set_sha256": (
                    "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c"
                ),
                "runtime_smoke_all_passed": True,
                "runtime_smoke_pass_count": 3,
                "scientific_environment_sealed": True,
                "validation_experiments_started": False,
                "untouched_test_accessed": False,
                "unchanged_after_deployment_actions": True,
            }
            immediate_predecessor = {
                "release_id": "release_20260921_visdrone_paper_v2r11",
                "run_name": "semuav_visdrone_v2r11",
                "release_path": str(immediate_predecessor_release),
                "archive_path": str(immediate_predecessor_archive),
                "archive_sha256": (
                    "737c168c8710d9f111f13ce138fefc9b3dc2f2bc4da9fc858a3cc11e76c465b8"
                ),
                "release_manifest_sha256": (
                    "b98c11aefe1617e6d2bffcaf9107b0708d31f45040db76d959a8c6664d7ee2ec"
                ),
                "checksum_inventory_sha256": (
                    "a761c0f0ae1f8d1ded79d388885f25c7dc4766cf94fe41b7120aa24d620e6ab0"
                ),
                "payload_file_count": 109,
                "archive_member_count": 111,
                "payload_total_uncompressed_bytes": 4063633,
                "builder_sha256": (
                    "c2edf8ebec76727fcaf2d785c01a3555704d4b5b639b8c373021588b1acd1447"
                ),
                "notebook_sha256": (
                    "9b6b4d4585036d0d2c45df1fce21af0f4a08fbcc90f9406cdeefb4c690b2154c"
                ),
                "deployer_path": str(immediate_predecessor_deployer),
                "deployer_sha256": (
                    "ee4ce4ad394e46eb85d4c83040f9f95020bf685436e06a2dfc1b8411d5905bb5"
                ),
                "deployment_receipt": str(immediate_predecessor_receipt),
                "deployment_receipt_sha256": (
                    "befdce3067b571933dd94d35758c2c29385a06a8bb02ea93ae681117dcc17f3e"
                ),
                "experiment1_recovery_receipt": str(recovery_path),
                "experiment1_recovery_receipt_sha256": (
                    "6ca88cb5e82c8b33e27fd229443e233a48323d91fd73d0e79ec51ff8fc61a608"
                ),
                "experiment1_recovery_receipt_bytes": 37416,
                "experiment1_recovery_artifact_count": 132,
                "experiment1_recovery_row_count": 4152,
                "active_model_set_sha256": (
                    "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c"
                ),
                "requirements_sha256": (
                    "028a2757d2e1565db90ff26a26fd6e564bd592c868ed093d632cb930d438dfa5"
                ),
                "source_deployment_ready": True,
                "runtime_smoke_run": False,
                "scientific_environment_sealed": False,
                "scientific_validation_authorized": False,
                "scientific_experiments_started": False,
                "validation_experiments_started": False,
                "untouched_test_accessed": False,
                "unchanged_after_deployment_actions": True,
            }
            release.mkdir(parents=True)
            archive_path.parent.mkdir(parents=True)
            receipt_path.parent.mkdir(parents=True)
            recovery_path.parent.mkdir(parents=True)

            source = re.sub(
                r"V2R13_EXPECTED_RELEASE_ROOT_TEXT = \(.*?^\s*\)",
                f"V2R13_EXPECTED_RELEASE_ROOT_TEXT = {str(release)!r}",
                parameter_source,
                count=1,
                flags=re.MULTILINE | re.DOTALL,
            )
            source = source.replace(
                '"/data/Lane Detection Datasets/musta/semuav_vlm"',
                repr(str(study)),
            ).replace(
                'Path("/data")', "EXPECTED_STUDY_STORAGE_ROOT.parent",
            )
            compile(source, "synthetic-v2r13-parameter-cell", "exec")

            payloads = {
                "semuav/example.py": b"SOURCE = 'sealed'\n",
                "configs/visdrone_paper.yaml": b"study: sealed\n",
                "prompts/example.txt": b"sealed prompt\n",
                "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md": (
                    b"# Synthetic v2r13 native-receiver correction\n"
                ),
                "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md": (
                    b"# Synthetic v2r13 acceptance isolation correction\n"
                ),
                "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md": (
                    b"# Synthetic v2r13 predecessor boundary routing correction\n"
                ),
                "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md": (
                    b"# Synthetic v2r13 low-evidence packet correction\n"
                ),
            }
            notebook = {
                "cells": [{
                    "cell_type": "code",
                    "execution_count": None,
                    "id": "parameters",
                    "metadata": {"tags": ["parameters"]},
                    "outputs": [],
                    "source": source,
                }],
                "metadata": {"kernelspec": {
                    "display_name": "Python (SemUAV-VLM 2026-09-21 v2r13)",
                    "language": "python",
                    "name": "semuav-vlm-20260921-v2r13",
                }},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
            notebook_bytes = (
                json.dumps(notebook, indent=1, sort_keys=True) + "\n"
            ).encode("utf-8")
            payloads["SemUAV_VLM_VisDrone_Paper_Study.ipynb"] = notebook_bytes
            roles = {
                "semuav/example.py": "scientific_source",
                "configs/visdrone_paper.yaml": "configuration",
                "prompts/example.txt": "prompt",
                "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md": (
                    "pre_validation_native_receiver_correction"
                ),
                "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md": (
                    "pre_validation_acceptance_recovery_isolation"
                ),
                "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md": (
                    "pre_smoke_predecessor_boundary_routing_correction"
                ),
                "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md": (
                    "pre_validation_low_evidence_packet_correction"
                ),
                "SemUAV_VLM_VisDrone_Paper_Study.ipynb": "notebook_entrypoint",
            }
            for relative, content in payloads.items():
                destination = release / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            inventory = [
                {
                    "bytes": len(content),
                    "path": relative,
                    "role": roles[relative],
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for relative, content in sorted(payloads.items())
            ]
            manifest = {
                "archive_member_count": len(inventory) + 2,
                "archive_name": (
                    "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"
                ),
                "notebook_builder": "build_visdrone_paper_notebook.py",
                "package_name": "SemUAV-VLM-VisDrone-Paper-Study",
                "payload_file_count": len(inventory),
                "payload_inventory": inventory,
                "payload_total_uncompressed_bytes": sum(
                    item["bytes"] for item in inventory
                ),
                "release_id": "release_20260921_visdrone_paper_v2r13",
                "run_name": "semuav_visdrone_v2r13",
                "schema_version": "semuav.visdrone-paper-release.v1",
                "study_entrypoint": "SemUAV_VLM_VisDrone_Paper_Study.ipynb",
                "pre_validation_native_receiver_correction": (
                    "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md"
                ),
                "pre_validation_acceptance_recovery_isolation": (
                    "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md"
                ),
                "pre_smoke_predecessor_boundary_routing_correction": (
                    "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md"
                ),
                "authenticated_predecessor_v2r11": immediate_predecessor,
                "authenticated_predecessor_v2r10": source_predecessor,
                "authenticated_predecessor_v2r12": authenticated_v2r12_predecessor,
                "pre_validation_low_evidence_packet_correction": (
                    "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md"
                ),
                "methodology": {
                    "native_receiver_correction": (
                        "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md"
                    ),
                },
            }
            manifest_bytes = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
            checksum_bytes = (
                "\n".join(sorted([
                    *(f"{item['sha256']}  {item['path']}" for item in inventory),
                    f"{manifest_sha}  RELEASE_MANIFEST.json",
                ])) + "\n"
            ).encode("utf-8")
            (release / "RELEASE_MANIFEST.json").write_bytes(manifest_bytes)
            (release / "RELEASE_MANIFEST_SHA256.txt").write_bytes(checksum_bytes)

            prefix = "SemUAV-VLM-VisDrone-Paper-Study/"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for relative, content in payloads.items():
                    archive.writestr(prefix + relative, content)
                archive.writestr(prefix + "RELEASE_MANIFEST.json", manifest_bytes)
                archive.writestr(
                    prefix + "RELEASE_MANIFEST_SHA256.txt", checksum_bytes,
                )
            archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            recovery_bytes = b"{}\n"
            recovery_path.write_bytes(recovery_bytes)
            receipt = {
                "created_at_utc": "2026-09-21T00:00:00Z",
                "deployment_mode": "NEW_ATOMIC_DEPLOYMENT",
                "scope": (
                    "immutable_v2r13_source_sibling_tests_and_"
                    "kernel_registration_only"
                ),
                "scientific_contract_changed": False,
                "candidate_requirements_changed": False,
                "package_installation_performed": False,
                "candidate_builder_invoked": False,
                "candidate_environment_modified": False,
                "base_v2r3_release_modified": False,
                "predecessor_v2r9_release_modified": False,
                "predecessor_v2r10_release_modified": False,
                "predecessor_v2r11_release_modified": False,
                "predecessor_v2r12_release_modified": False,
                "predecessor_v2r12_state": {
                    "schema_version": "semuav.visdrone-v2r12-preserved-state.v1",
                    "release_root": str(study / "releases/release_20260921_visdrone_paper_v2r12/SemUAV-VLM-VisDrone-Paper-Study"),
                    "output_root": str(study / "outputs/visdrone_paper/semuav_visdrone_v2r12"),
                    "cache_root": str(study / ".cache/visdrone_paper_v2r12"),
                    "trusted_code_cache_root": str(study / ".cache/huggingface/modules-v2r12"),
                    "release_tree": "0" * 64 + " 1 1",
                    "output_tree": "1" * 64 + " 1 1",
                    "cache_tree": "2" * 64 + " 1 1",
                    "trusted_code_cache_tree": "3" * 64 + " 0 0",
                    "unchanged_after_deployment_actions": True,
                },
                "scientific_validation_authorized": False,
                "next_gate": "FRESH_V2R13_THREE_MODEL_DEVELOPMENT_RUNTIME_SMOKE",
                "release_id": "release_20260921_visdrone_paper_v2r13",
                "release_root": str(release),
                "run_name": "semuav_visdrone_v2r13",
                "runtime_smoke_run": False,
                "gpu_accessed": False,
                "model_weights_loaded": False,
                "jupyter_launched": False,
                "schema_version": "semuav.visdrone-v2r13-source-deployment.v1",
                "scientific_experiments_run": False,
                "scientific_validation_run": False,
                "status": "READY_FOR_DEVELOPMENT_SMOKE_ONLY",
                "untouched_test_run": False,
                "validation_or_test_images_or_labels_read": False,
                "release_revision": {},
                "installer": {},
                "deployment_script": {},
                "builder": {},
                "authenticated_base_v2r3": {},
                "authenticated_predecessor_v2r9": {},
                "authenticated_predecessor_v2r10": source_predecessor,
                "authenticated_predecessor_v2r11": immediate_predecessor,
                "authenticated_predecessor_v2r12": authenticated_v2r12_predecessor,
                "predecessor_observed_terminal_boundary": {},
                "candidate_environment": {},
                "model_set_amendment": {},
                "temporal_feasibility_amendment": {},
                "source_compatibility": {},
                "unit_tests": {},
                "jupyter_kernel": {},
                "jupyter_launch_target": {},
                "trusted_code_cache_namespace": {},
                "attempt_directory": {},
                "deployment_log": {},
                "experiment1_recovery": {
                    "path": str(recovery_path),
                    "bytes": len(recovery_bytes),
                    "sha256": hashlib.sha256(recovery_bytes).hexdigest(),
                    "schema_version": "semuav.visdrone-v2r9-exp1-recovery.v1",
                    "status": "QUIESCENT_V2R9_EXP1_SOURCE_FROZEN",
                    "artifact_count": 132,
                    "inference_row_count": 4152,
                },
                "release": {
                    "archive_name": (
                        "SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"
                    ),
                    "archive_sha256": archive_sha,
                    "archive_source": str(archive_path),
                    "checksum_inventory_sha256": hashlib.sha256(
                        checksum_bytes
                    ).hexdigest(),
                    "manifest_sha256": manifest_sha,
                    "path": str(release),
                    "release_id": "release_20260921_visdrone_paper_v2r13",
                    "run_name": "semuav_visdrone_v2r13",
                },
                "source_release": {
                    "archive_path": str(archive_path),
                    "archive_sha256": archive_sha,
                    "checksum_inventory_sha256": hashlib.sha256(
                        checksum_bytes
                    ).hexdigest(),
                    "manifest_sha256": manifest_sha,
                },
            }
            receipt_bytes = (
                json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            receipt_path.write_bytes(receipt_bytes)
            environment = {
                "PYTHONNOUSERSITE": "1",
                "SEMUAV_PROJECT_ROOT": str(release),
                "SEMUAV_STUDY_ROOT": str(study),
                "SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256": hashlib.sha256(
                    receipt_bytes
                ).hexdigest(),
            }

            expected_python = (
                "/data/Lane Detection Datasets/musta/semuav_vlm/venvs/"
                "release_20260912_visdrone_inference_candidate_v2r3/bin/python"
            )
            expected_prefix = str(Path(expected_python).parent.parent)
            flag_values = {
                name: getattr(sys.flags, name)
                for name in dir(sys.flags)
                if not name.startswith("_")
            }
            flag_values.update(isolated=1, dont_write_bytecode=1)
            isolated_flags = SimpleNamespace(**flag_values)

            def execute_bootstrap(environment_updates=None):
                namespace = {}
                execution_environment = {
                    **environment, **(environment_updates or {}),
                }
                clean_modules = {
                    name: module for name, module in sys.modules.items()
                    if name != "semuav" and not name.startswith("semuav.")
                }
                with (
                    mock.patch.dict(
                        os.environ, execution_environment, clear=True,
                    ),
                    mock.patch.object(sys, "path", list(sys.path)),
                    mock.patch.object(sys, "executable", expected_python),
                    mock.patch.object(sys, "prefix", expected_prefix),
                    mock.patch.object(sys, "flags", isolated_flags),
                    mock.patch.object(sys, "dont_write_bytecode", True),
                    mock.patch.dict(sys.modules, clean_modules, clear=True),
                ):
                    exec(
                        compile(source, "synthetic-v2r13-parameter-cell", "exec"),
                        namespace,
                    )
                self.assertIs(namespace["V2R13_PREIMPORT_SOURCE_VERIFIED"], True)
                self.assertIs(namespace["EXECUTE_PHASE"], False)

            execute_bootstrap()
            tampered_receipt = copy.deepcopy(receipt)
            tampered_receipt["authenticated_predecessor_v2r10"][
                "runtime_smoke_pass_count"
            ] = 2
            tampered_receipt_bytes = (
                json.dumps(tampered_receipt, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            receipt_path.write_bytes(tampered_receipt_bytes)
            with self.assertRaisesRegex(
                ValueError,
                "authenticated v2r10 source predecessor record changed",
            ):
                execute_bootstrap({
                    "SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256": hashlib.sha256(
                        tampered_receipt_bytes
                    ).hexdigest(),
                })
            receipt_path.write_bytes(receipt_bytes)
            with self.assertRaisesRegex(ValueError, "requires DEVICE=cuda:0"):
                execute_bootstrap({"SEMUAV_DEVICE": "cpu"})
            for relative in (
                "semuav/example.py",
                "configs/visdrone_paper.yaml",
                "prompts/example.txt",
            ):
                with self.subTest(drift=relative):
                    path = release / relative
                    original = path.read_bytes()
                    path.write_bytes(original + b"drift")
                    with self.assertRaisesRegex(ValueError, "payload bytes changed"):
                        execute_bootstrap()
                    path.write_bytes(original)
            original_archive = archive_path.read_bytes()
            archive_path.write_bytes(original_archive + b"drift")
            with self.assertRaisesRegex(ValueError, "source archive bytes changed"):
                execute_bootstrap()
            archive_path.write_bytes(original_archive)
            changed_notebook = copy.deepcopy(notebook)
            changed_notebook["metadata"]["arbitrary"] = True
            (release / "SemUAV_VLM_VisDrone_Paper_Study.ipynb").write_text(
                json.dumps(changed_notebook, indent=1, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "outside approved parameter"):
                execute_bootstrap()
            notebook_path = release / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
            notebook_path.write_bytes(notebook_bytes)

            bytecode_dir = release / "semuav" / "__pycache__"
            bytecode_dir.mkdir()
            (bytecode_dir / "injected.pyc").write_bytes(b"unsafe")
            with self.assertRaisesRegex(ValueError, "executable bytecode"):
                execute_bootstrap()
            (bytecode_dir / "injected.pyc").unlink()
            bytecode_dir.rmdir()

            checkpoint_dir = release / ".ipynb_checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "unexpected.ipynb").write_bytes(notebook_bytes)
            with self.assertRaisesRegex(ValueError, "unexpected Jupyter"):
                execute_bootstrap()

    def test_run_all_cannot_cross_scientific_phases(self) -> None:
        sources = [str(cell.get("source", "")) for cell in self.notebook["cells"]]
        barriers = [
            source for cell, source in zip(self.notebook["cells"], sources)
            if "phase-barrier" in cell.get("metadata", {}).get("tags", [])
        ]
        self.assertGreaterEqual(len(barriers), 3)
        combined = "\n".join(sources)
        self.assertIn('if PHASE == "prepare" and EXECUTE_PHASE:', combined)
        self.assertIn('if PHASE == "validation" and EXECUTE_PHASE:', combined)
        self.assertIn('if PHASE == "test" and EXECUTE_PHASE:', combined)
        self.assertIn('if PHASE == "report" and EXECUTE_PHASE:', combined)
        self.assertIn('if PHASE == "embedded" and EXECUTE_PHASE:', combined)
        self.assertNotIn('PHASE = "test"', combined)
        self.assertNotIn('PHASE = "report"', combined)
        action_conditions = re.findall(
            r'(?m)^if (?:PHASE == "(?:prepare|validation|test|report|embedded)" '
            r'and EXECUTE_PHASE|EXECUTE_PHASE and PHASE in \{[^\n]+\}):\n'
            r'(?P<body>[^\n]+)',
            combined,
        )
        self.assertGreaterEqual(len(action_conditions), 12)
        self.assertTrue(all(
            "_require_v2r13_preimport_source_gate()" in body
            for body in action_conditions
        ))

    def test_acceptance_tests_isolate_lock_and_recovery_variables(self) -> None:
        acceptance_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "unittest_lock_specific_variables" in str(cell.get("source", ""))
        )
        expected_lock_specific = (
            "HF_HUB_CACHE",
            "HUGGINGFACE_HUB_CACHE",
            "TRANSFORMERS_CACHE",
            "SEMUAV_QWEN_REVISION",
            "SEMUAV_SMOLVLM_REVISION",
            "SEMUAV_MOONDREAM_REVISION",
            "SEMUAV_UPPER_BOUND_REVISION",
            "SEMUAV_MOONDREAM_TOKENIZER_LOCK",
        )
        expected_recovery_variables = (
            "SEMUAV_ENABLE_V2R9_EXP1_RECOVERY",
            "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT",
            "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256",
        )
        offline_guards = {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "SEMUAV_MODELS_LOCAL_ONLY": "1",
        }
        parent_environment = {
            "PARENT_RUNTIME_SENTINEL": "preserved",
            "SEMUAV_LIVE_NOTEBOOK_SELFTEST": "parent-value",
            **offline_guards,
            **{
                variable: f"parent-lock-value-{index}"
                for index, variable in enumerate(expected_lock_specific)
            },
            "SEMUAV_ENABLE_V2R9_EXP1_RECOVERY": "1",
            "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT": (
                "/authenticated/V2R9_EXP1_RECOVERY_SOURCE.json"
            ),
            "SEMUAV_V2R9_EXP1_RECOVERY_RECEIPT_SHA256": "d" * 64,
        }
        captured: dict[str, object] = {}

        def fake_run(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return SimpleNamespace(stdout="", stderr="", returncode=0)

        namespace = {
            "PROJECT_ROOT": ROOT,
            "os": os,
            "subprocess": SimpleNamespace(run=fake_run),
            "sys": sys,
        }
        with mock.patch.dict(os.environ, parent_environment, clear=True):
            exec(
                compile(acceptance_source, "notebook-acceptance-cell", "exec"),
                namespace,
            )
            self.assertEqual(dict(os.environ), parent_environment)

        self.assertEqual(
            namespace["unittest_lock_specific_variables"],
            expected_lock_specific,
        )
        self.assertEqual(
            namespace["unittest_recovery_variables"],
            expected_recovery_variables,
        )
        child_environment = captured["kwargs"]["env"]
        expected_child = dict(parent_environment)
        for variable in (
            *expected_lock_specific,
            *expected_recovery_variables,
        ):
            expected_child.pop(variable)
        expected_child["SEMUAV_LIVE_NOTEBOOK_SELFTEST"] = "1"
        self.assertEqual(child_environment, expected_child)
        for variable, expected in offline_guards.items():
            self.assertEqual(child_environment[variable], expected)
        for variable in (
            *expected_lock_specific,
            *expected_recovery_variables,
        ):
            self.assertNotIn(variable, child_environment)

    def test_acceptance_tests_fail_closed_before_subprocess_if_offline_guard_differs(self) -> None:
        acceptance_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "unittest_lock_specific_variables" in str(cell.get("source", ""))
        )
        required_guards = (
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "SEMUAV_MODELS_LOCAL_ONLY",
        )
        for differing_guard in required_guards:
            with self.subTest(differing_guard=differing_guard):
                parent_environment = {
                    variable: "1" for variable in required_guards
                }
                parent_environment[differing_guard] = "0"
                run = mock.Mock()
                namespace = {
                    "PROJECT_ROOT": ROOT,
                    "os": os,
                    "subprocess": SimpleNamespace(run=run),
                    "sys": sys,
                }
                with mock.patch.dict(os.environ, parent_environment, clear=True):
                    with self.assertRaisesRegex(RuntimeError, differing_guard):
                        exec(
                            compile(
                                acceptance_source,
                                "notebook-acceptance-cell",
                                "exec",
                            ),
                            namespace,
                        )
                    self.assertEqual(dict(os.environ), parent_environment)
                run.assert_not_called()

    def test_physical_gpu_1_exclusivity_is_fail_closed_at_scientific_boundaries(self) -> None:
        combined = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
        )
        self.assertIn(
            "require_exclusive_physical_gpu(\n        1, allowed_pids=(os.getpid(),)",
            combined,
        )
        self.assertIn(
            "occupancy gate is deliberately separate from the\n"
            "    # immutable environment fingerprint because OS PIDs are volatile",
            combined,
        )

        prepare_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if '_require_exclusive_scientific_gpu("prepare_runtime_smoke")'
            in str(cell.get("source", ""))
        )
        self.assertLess(
            prepare_source.index(
                '_require_exclusive_scientific_gpu("prepare_runtime_smoke")'
            ),
            prepare_source.index("run_one_image_runtime_smoke("),
        )

        validation_boundaries = {
            "validation_input_gate": "_verify_inference_environment_seal()",
            "validation_experiments_1_2": "run_validation_perception_and_frontier(",
            "validation_experiments_3_5": "run_validation_network_and_controller(",
            "validation_experiments_6_7": "experiment6_receiver_robustness()",
            "validation_claim_and_freeze": "validation_gates_passed = bool(",
        }
        validation_sources = [
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "validation-only" in cell.get("metadata", {}).get("tags", [])
        ]
        for stage, operation in validation_boundaries.items():
            with self.subTest(stage=stage):
                source = next(item for item in validation_sources if operation in item)
                self.assertLess(
                    source.index(f'_require_exclusive_scientific_gpu("{stage}")'),
                    source.index(operation),
                )

        test_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if 'if PHASE == "test" and EXECUTE_PHASE:' in str(cell.get("source", ""))
        )
        self.assertLess(
            test_source.index('_require_exclusive_scientific_gpu("test_pre_open")'),
            test_source.index("authorize_test_once(ctx, PROTOCOL)"),
        )
        self.assertLess(
            test_source.index(
                '_require_exclusive_scientific_gpu("test_confirmatory_execution")'
            ),
            test_source.index("run_confirmatory_test_once("),
        )

        guide = RUN_GUIDE.read_text(encoding="utf-8")
        for token in (
            "After deployment, the v2r13 session, TCP port 8888, and physical GPU 1 must\n"
            "still be clean.",
            "If the port or GPU is occupied, identify its owner and full command first.",
            "Never kill an arbitrary PID or another user's process.",
            "V2R13_TMUX_SESSION=ABSENT",
            "PORT_8888=FREE",
            "STOP: v2r13 tmux session appeared after the clean-boundary check",
            "STOP: port 8888 became occupied after the clean-boundary check",
        ):
            self.assertIn(token, guide)

    def test_test_is_exact_once_and_report_is_read_only(self) -> None:
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        for token in (
            "authorize_test_once(ctx, PROTOCOL)",
            "load_untouched_test_permit(",
            'ctx, status="COMPLETE", result_artifacts=result_paths',
            "load_report_safe_finalized_test_handoff(",
        ):
            self.assertIn(token, combined)
        report_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "report-only" in cell.get("metadata", {}).get("tags", [])
        )
        self.assertNotIn("build_untouched_test_manifest", report_sources)
        self.assertNotIn("authorize_test_once", report_sources)
        self.assertNotIn("verify_visdrone_source_provenance", report_sources)
        self.assertNotIn("load_finalized_test_handoff(ctx, PROTOCOL)", report_sources)
        embedded_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "embedded-only" in cell.get("metadata", {}).get("tags", [])
        )
        self.assertIn("verify_paper_completion_receipt(", embedded_sources)
        self.assertIn("handoff={}, cfg=cfg, raw_trial_path=None", embedded_sources)
        self.assertNotIn("load_finalized_test_handoff(ctx, PROTOCOL)", embedded_sources)
        self.assertNotIn('cfg.hardware["target_log"]', embedded_sources)

    def test_full_source_integrity_precedes_validation_and_test_access(self) -> None:
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        self.assertGreaterEqual(combined.count("full_integrity=True"), 3)
        test_start = combined.index('if PHASE == "test" and EXECUTE_PHASE:')
        test_source = combined[test_start:]
        authorization = test_source.index("authorize_test_once(ctx, PROTOCOL)")
        before_open = test_source[:authorization]
        after_open = test_source[authorization:]
        self.assertIn("components=DEVELOPMENT_COMPONENTS", before_open)
        self.assertNotIn("UNTOUCHED_TEST_COMPONENTS", before_open)
        self.assertIn("components=UNTOUCHED_TEST_COMPONENTS", after_open)
        self.assertIn("require_test_permit=True", after_open)

    def test_prepare_reuses_verified_v1_bundle_without_rebuilding_it(self) -> None:
        combined = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
        )
        prepare_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "prepare-only" in cell.get("metadata", {}).get("tags", [])
        )
        self.assertIn(
            "_verify_or_create_reused_v1_artifacts_receipt(", prepare_sources,
        )
        self.assertIn(
            'RUN_NAME != "semuav_visdrone_v2r13"', combined,
        )
        self.assertNotIn("RUN_NAME.startswith(", combined)
        self.assertIn("allow_create=True", prepare_sources)
        self.assertIn("REUSED_V1_FROZEN_ARTIFACTS_VERIFIED", prepare_sources)
        self.assertIn("load_prepared_visdrone_bundle(DEVELOPMENT_BUNDLE_DIR)", combined)
        self.assertIn("verify_artifact_hash_manifest(", combined)
        self.assertNotIn(
            "load_prepared_visdrone_bundle(DEVELOPMENT_BUNDLE_DIR)",
            prepare_sources,
        )
        self.assertNotIn(
            "_verify_full_reused_v1_development_bundle(", prepare_sources,
        )
        # Test-dev is still materialized only after the one-time OPEN ledger,
        # so ``prepare_visdrone_bundle`` legitimately remains in the test cell.
        # The v2r13 prepare cell itself must never rebuild any v1 artifact family.
        for prohibited in (
            "prepare_visdrone_bundle(", "prepare_ultralytics_visdrone_det(",
            "detector_training_command_plan(", "freeze_huggingface_models(",
            "prepare_feature_head_artifact(", "write_detector_training_receipt(",
        ):
            self.assertNotIn(prohibited, prepare_sources)

    def test_preseal_reuse_inventory_excludes_deferred_validation_artifacts(self) -> None:
        setup_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "PRESEAL_REUSED_V1_ARTIFACT_PATHS =" in str(cell.get("source", ""))
        )
        module = ast.parse(setup_source)
        inventory_assignment = next(
            node for node in module.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "PRESEAL_REUSED_V1_ARTIFACT_PATHS"
                for target in node.targets
            )
        )
        self.assertIsInstance(inventory_assignment.value, ast.Dict)
        inventory_names = {
            str(ast.literal_eval(key))
            for key in inventory_assignment.value.keys
            if key is not None
        }
        self.assertEqual(inventory_names, {
            "model_lock",
            "detector_training_environment_fingerprint",
            "detector_training_pip_freeze",
            "detector_training_plan",
            "detector_training_receipt",
            "registered_train_sample",
            "registered_train_sampling_audit",
            "registered_train_sample_artifacts",
            "feature_head_lock",
            "feature_head_weights",
            "feature_encoder_receipt",
        })
        preseal_function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_or_create_reused_v1_artifacts_receipt"
        )
        preseal_source = ast.get_source_segment(setup_source, preseal_function) or ""
        for prohibited in (
            "load_prepared_visdrone_bundle",
            "verify_detector_training_receipt",
            "verify_artifact_hash_manifest",
            "DEVELOPMENT_BUNDLE_DIR",
            "registered_validation_sample",
            "registered_calibration_sample",
            "detector_study_source_manifest",
        ):
            self.assertNotIn(prohibited, preseal_source)
        self.assertIn(
            '"validation_derived_artifacts_read_before_seal": False',
            preseal_source,
        )

        train_manifest_function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_preseal_registered_train_artifact_manifest"
        )
        train_manifest_source = (
            ast.get_source_segment(setup_source, train_manifest_function) or ""
        )
        self.assertNotIn("verify_artifact_hash_manifest(", train_manifest_source)
        for required in (
            'set(payload) != {',
            '"schema_version", "workflow_version", "root", "metadata", "artifacts"',
            'set(metadata) != {"selection_sha256"}',
            'payload.get("artifacts") != expected_artifacts',
            '"registered_analysis_sample.csv.gz"',
            '"registered_analysis_sampling_audit.csv"',
            "manifest_bytes = manifest_path.read_bytes()",
            "hashlib.sha256(manifest_bytes).hexdigest()",
            'manifest_bytes.decode("utf-8")',
            'object_pairs_hook=reject_duplicate_keys',
        ):
            self.assertIn(required, train_manifest_source)
        self.assertNotIn("manifest_path.read_text", train_manifest_source)

        full_function = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_full_reused_v1_development_bundle"
        )
        full_source = ast.get_source_segment(setup_source, full_function) or ""
        for required in (
            "load_prepared_visdrone_bundle(DEVELOPMENT_BUNDLE_DIR)",
            'for split_name in ("train", "calibration", "validation")',
            "verify_detector_training_receipt(",
            "scientific_environment = _verify_inference_environment_seal()",
        ):
            self.assertIn(required, full_source)
        self.assertEqual(len(full_function.args.args), 0)
        self.assertNotIn('scientific_environment.get("payload", {})', full_source)

        identity_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "expected_model_lock_sha256 =" in str(cell.get("source", ""))
        )
        symlink_guard = identity_source.index("SOURCE_MODEL_LOCK.is_symlink()")
        byte_read = identity_source.index(
            "model_lock_bytes = SOURCE_MODEL_LOCK.read_bytes()"
        )
        anchor_check = identity_source.index(
            "hashlib.sha256(model_lock_bytes).hexdigest()"
        )
        model_lock_parse = identity_source.index(
            'json.loads(model_lock_bytes.decode("utf-8"))'
        )
        self.assertLess(symlink_guard, byte_read)
        self.assertLess(byte_read, anchor_check)
        self.assertLess(anchor_check, model_lock_parse)
        self.assertNotIn("MODEL_LOCK.read_text", identity_source)
        self.assertIn(
            'reused_cfg.get("expected_sha256", {}).get("model_lock", "")',
            identity_source,
        )

    def test_preseal_reuse_path_guard_rejects_deferred_roots_and_publisher_val(self) -> None:
        setup_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "def _assert_preseal_reuse_path_allowed" in str(cell.get("source", ""))
        )
        module = ast.parse(setup_source)
        guard = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_assert_preseal_reuse_path_allowed"
        )
        isolated = ast.Module(body=[guard], type_ignores=[])
        ast.fix_missing_locations(isolated)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            development_bundle = root / "development_bundle"
            sample_root = root / "development_registered_samples"
            namespace = {
                "Path": Path,
                "StudyGateError": RuntimeError,
                "PRESEAL_FORBIDDEN_REUSE_ROOTS": (
                    development_bundle,
                    sample_root / "calibration",
                    sample_root / "validation",
                ),
                "PRESEAL_FORBIDDEN_PUBLISHER_COMPONENTS": {
                    "VisDrone2019-DET-val", "VisDrone2019-MOT-val",
                },
            }
            exec(compile(isolated, "preseal-path-guard", "exec"), namespace)
            verify = namespace["_assert_preseal_reuse_path_allowed"]
            allowed = sample_root / "train" / "registered_analysis_sample.csv.gz"
            self.assertEqual(verify(allowed), allowed.resolve())
            for forbidden in (
                development_bundle / "analysis_manifest.csv.gz",
                sample_root / "calibration" / "registered_analysis_sample.csv.gz",
                sample_root / "validation" / "registered_analysis_sample.csv.gz",
                root / "VisDrone2019-DET-val" / "images" / "frame.jpg",
                root / "VisDrone2019-MOT-val" / "annotations" / "frame.txt",
            ):
                with self.assertRaises(RuntimeError):
                    verify(forbidden)

            train_manifest_function = next(
                node for node in module.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "_verify_preseal_registered_train_artifact_manifest"
            )
            manifest_module = ast.Module(
                body=[train_manifest_function], type_ignores=[],
            )
            ast.fix_missing_locations(manifest_module)

            train_root = sample_root / "train"
            validation_root = sample_root / "validation"
            train_root.mkdir(parents=True)
            validation_root.mkdir(parents=True)
            sample_path = train_root / "registered_analysis_sample.csv.gz"
            audit_path = train_root / "registered_analysis_sampling_audit.csv"
            manifest_path = train_root / "REGISTERED_ANALYSIS_SAMPLE_ARTIFACTS.json"
            malicious_validation_path = (
                validation_root / "registered_analysis_sample.csv.gz"
            )
            sample_path.write_bytes(b"registered-train-sample")
            audit_path.write_bytes(b"registered-train-audit")
            malicious_validation_path.write_bytes(b"must-not-be-opened-pre-seal")

            sample_sha = hashlib.sha256(sample_path.read_bytes()).hexdigest()
            audit_sha = hashlib.sha256(audit_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps({
                "schema_version": 1,
                "workflow_version": "semuav.visdrone-paper-workflow.v1",
                "root": ".",
                "metadata": {"selection_sha256": "3" * 64},
                "artifacts": {
                    "registered_analysis_sample": {
                        "path": "registered_analysis_sample.csv.gz",
                        "sha256": sample_sha,
                        "bytes": sample_path.stat().st_size,
                    },
                    "sampling_audit": {
                        "path": "registered_analysis_sampling_audit.csv",
                        "sha256": audit_sha,
                        "bytes": audit_path.stat().st_size,
                    },
                    "redirected_validation": {
                        "path": "../validation/registered_analysis_sample.csv.gz",
                        "sha256": hashlib.sha256(
                            malicious_validation_path.read_bytes()
                        ).hexdigest(),
                        "bytes": malicious_validation_path.stat().st_size,
                    },
                },
            }), encoding="utf-8")
            inventory = {
                "registered_train_sample": {
                    "path": str(sample_path.resolve()),
                    "sha256": sample_sha,
                    "bytes": sample_path.stat().st_size,
                },
                "registered_train_sampling_audit": {
                    "path": str(audit_path.resolve()),
                    "sha256": audit_sha,
                    "bytes": audit_path.stat().st_size,
                },
                "registered_train_sample_artifacts": {
                    "path": str(manifest_path.resolve()),
                    "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    "bytes": manifest_path.stat().st_size,
                },
            }
            manifest_namespace = {
                "Path": Path,
                "hashlib": hashlib,
                "json": json,
                "StudyGateError": RuntimeError,
                "WORKFLOW_VERSION": "semuav.visdrone-paper-workflow.v1",
                "PRESEAL_REUSED_V1_ARTIFACT_PATHS": {
                    "registered_train_sample_artifacts": manifest_path,
                },
            }
            exec(
                compile(manifest_module, "preseal-train-manifest", "exec"),
                manifest_namespace,
            )
            verify_manifest = manifest_namespace[
                "_verify_preseal_registered_train_artifact_manifest"
            ]
            opened_paths = []
            original_path_open = Path.open

            def observed_path_open(path, *args, **kwargs):
                opened_paths.append(Path(path).resolve())
                return original_path_open(path, *args, **kwargs)

            with mock.patch.object(Path, "open", new=observed_path_open):
                with self.assertRaises(RuntimeError):
                    verify_manifest(inventory)
            self.assertEqual(opened_paths, [manifest_path.resolve()])
            self.assertNotIn(malicious_validation_path.resolve(), opened_paths)

        output_guard_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "def _assert_safe_output_directory" in str(cell.get("source", ""))
        )
        output_module = ast.parse(output_guard_source)
        output_guard = next(
            node for node in output_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_assert_safe_output_directory"
        )
        isolated_output_guard = ast.Module(body=[output_guard], type_ignores=[])
        ast.fix_missing_locations(isolated_output_guard)
        output_namespace = {"Path": Path, "StudyGateError": RuntimeError}
        exec(
            compile(isolated_output_guard, "v2r2-output-guard", "exec"),
            output_namespace,
        )
        verify_output = output_namespace["_assert_safe_output_directory"]
        with tempfile.TemporaryDirectory() as directory:
            output_root = Path(directory)
            redirected = output_root / "validation-derived"
            redirected.mkdir()
            run_output = output_root / "semuav_visdrone_v2r2"
            run_output.symlink_to(redirected, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                verify_output(run_output, label="run output", create=False)
            run_output.unlink()
            run_output.mkdir()
            frozen_inputs = run_output / "frozen_inputs"
            frozen_inputs.symlink_to(redirected, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                verify_output(frozen_inputs, label="frozen inputs", create=True)

    def test_generated_config_key_lookups_execute_against_registered_yaml(self) -> None:
        config = yaml.safe_load((ROOT / "configs" / "visdrone_paper.yaml").read_text(encoding="utf-8"))
        cfg = SimpleNamespace(**config)
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        expressions = sorted(set(re.findall(
            r'cfg\.(evaluation|network|controller|benchmark|hardware|models)\["([^"]+)"\]',
            combined,
        )))
        self.assertTrue(expressions)
        for section, key in expressions:
            with self.subTest(section=section, key=key):
                value = eval(f'cfg.{section}[{key!r}]', {"cfg": cfg})
                self.assertIsNotNone(value)
        frozen_input_policy = (
            "common_aspect_preserving_letterbox_then_pinned_native_processor_v1"
        )
        self.assertEqual(config["models"]["input_policy"], frozen_input_policy)
        self.assertIn(frozen_input_policy, combined)

    def test_registered_structural_not_estimable_status_matches_notebook(self) -> None:
        config = yaml.safe_load(
            (ROOT / "configs" / "visdrone_paper.yaml").read_text(encoding="utf-8")
        )
        evaluation = config["evaluation"]
        self.assertEqual(
            evaluation["expected_not_estimable_hypotheses"], ["H2", "H3"],
        )
        reasons = evaluation["expected_not_estimable_reasons"]
        self.assertEqual(set(reasons), {"H2", "H3"})
        self.assertTrue(all(str(reasons[name]).strip() for name in ("H2", "H3")))
        self.assertIn("bounded_15_sequence_temporal", str(reasons["H3"]))
        self.assertEqual(float(evaluation["h4_high_confidence_threshold"]), 0.80)
        combined = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
        )
        for token in (
            '"h2_structural_status": "NOT_ESTIMABLE"',
            '"h3_structural_status": "NOT_ESTIMABLE"',
            '"H2=STRUCTURALLY_NOT_ESTIMABLE"',
            '"H3=STRUCTURALLY_NOT_ESTIMABLE"',
        ):
            self.assertIn(token, combined)

    def test_reused_detector_and_feature_artifacts_are_hash_bound(self) -> None:
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        for token in (
            '"detector_training_plan": DETECTOR_PLAN',
            '"detector_training_receipt": DETECTOR_TRAINING_RECEIPT',
            '"feature_head_lock": FEATURE_HEAD_LOCK',
            '"feature_head_weights": FEATURE_HEAD_WEIGHTS',
            '"feature_encoder_receipt": FEATURE_ENCODER_RECEIPT',
            '"detector_conversion_manifest"',
            '"detector_leakage_audit"',
            '"feature_encoder_checkpoint"',
            '"reuse_mode": "immutable_read_only_reference_without_retraining"',
            'inventory[name]["sha256"]',
        ):
            self.assertIn(token, combined)
        self.assertIn("feature_head_lock=FEATURE_HEAD_LOCK", combined)

    def test_historical_training_and_v2r13_inference_environments_are_separate(self) -> None:
        combined = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
        )
        for token in (
            "DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT =",
            "DETECTOR_TRAINING_PIP_FREEZE =",
            "INFERENCE_ENVIRONMENT_FINGERPRINT =",
            "INFERENCE_PIP_FREEZE =",
            "INFERENCE_ENVIRONMENT_SEAL =",
            "_capture_or_verify_inference_runtime(",
            "_verify_candidate_ready_receipt(",
            "_create_inference_environment_seal(",
            "_verify_inference_environment_seal(",
            "SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256",
            "semuav.visdrone-source-candidate-binding.v1",
            "training_plan_path=DETECTOR_PLAN",
            "DETECTOR_TRAINING_ENVIRONMENT_FINGERPRINT",
            '"historical_detector_training_environment_only": True',
            '"inference_runtime_identity_stored_separately": True',
            '"historical_detector_training_environment_fingerprint"',
            '"inference_environment_fingerprint"',
        ):
            self.assertIn(token, combined)
        self.assertNotIn("environment_fingerprint_path=INFERENCE_ENVIRONMENT_FINGERPRINT", combined)

        helper_cell = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "def _verify_candidate_ready_receipt" in str(cell.get("source", ""))
        )
        helper_module = ast.parse(helper_cell)
        candidate_functions = [
            node for node in helper_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_candidate_ready_receipt"
        ]
        self.assertEqual(len(candidate_functions), 1)
        candidate_function = candidate_functions[0]
        candidate_source = ast.get_source_segment(
            helper_cell, candidate_function,
        ) or ""
        for token in (
            "require_inference_lock=True",
            'release_id = "release_20260921_visdrone_paper_v2r13"',
            'run_name = "semuav_visdrone_v2r13"',
            '"release_20260912_visdrone_inference_candidate_v2r3"',
            'candidate_source_release_id = "release_20260912_visdrone_paper_v2r3"',
            'candidate_source_run_name = "semuav_visdrone_v2r3"',
            '"build_semuav_visdrone_inference_candidate_v2r3_20260912.sh"',
            '"deploy_semuav_visdrone_v2r13_20260921.sh"',
            '"SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r13.zip"',
            '"visdrone_v2r13_deployment"',
            'os.environ.get("SEMUAV_V2R13_DEPLOYMENT_RECEIPT_SHA256", "")',
            '"semuav.visdrone-v2r13-source-deployment.v1"',
            'len(expected_deployment_keys) != 52',
            '"predecessor_v2r9_release_modified"',
            '"predecessor_v2r10_release_modified"',
            '"predecessor_v2r11_release_modified"',
            '"predecessor_v2r12_release_modified"',
            '"predecessor_v2r12_state"',
            'deployment["authenticated_predecessor_v2r9"]',
            'deployment["authenticated_predecessor_v2r10"]',
            'deployment["authenticated_predecessor_v2r11"]',
            'deployment["authenticated_predecessor_v2r12"]',
            'immediate_predecessor_release_id = (',
            '"release_20260921_visdrone_paper_v2r11"',
            'immediate_predecessor_run_name = "semuav_visdrone_v2r11"',
            '"SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r11.zip"',
            '"deploy_semuav_visdrone_v2r11_20260921.sh"',
            'immediate_predecessor_evidence_required = {',
            'set(immediate_predecessor_evidence)',
            '!= immediate_predecessor_evidence_required',
            '!= expected_immediate_predecessor_evidence',
            'source_predecessor_release_id = (',
            '"release_20260921_visdrone_paper_v2r10"',
            'source_predecessor_run_name = "semuav_visdrone_v2r10"',
            '"SemUAV-VLM-VisDrone-Paper-Study-20260921-v2r10.zip"',
            '"deploy_semuav_visdrone_v2r10_20260921.sh"',
            'source_predecessor_evidence_required = {',
            'set(source_predecessor_evidence)',
            '!= source_predecessor_evidence_required',
            '!= expected_source_predecessor_evidence',
            '"payload_file_count": 108',
            '"archive_member_count": 110',
            '"payload_total_uncompressed_bytes": 4009068',
            '"experiment1_recovery_receipt_bytes": 37416',
            '"experiment1_recovery_artifact_count": 132',
            '"experiment1_recovery_row_count": 4152',
            '"runtime_smoke_all_passed": True',
            '"runtime_smoke_pass_count": 3',
            '"scientific_environment_sealed": True',
            '"validation_experiments_started": False',
            '"untouched_test_accessed": False',
            '"unchanged_after_deployment_actions": True',
            'source_predecessor_evidence["builder_sha256"]',
            'source_predecessor_evidence["notebook_sha256"]',
            'source_predecessor_deployment["experiment1_recovery"]',
            'deployment["predecessor_observed_terminal_boundary"]',
            'expected_observed_terminal_boundary = {',
            '"semuav.visdrone-v2r11-operator-observed-terminal-boundary.v1"',
            '"failure_exception_type": "StudyGateError"',
            '"runtime_smoke_entered": False',
            '"validation_software_acceptance_entered": False',
            '"validation_experiments_1_to_7_started": False',
            'expected_inherited_v2r10_boundary = {',
            '"semuav.visdrone-v2r10-operator-observed-terminal-boundary.v1"',
            'manifest_inherited_v2r10_boundary = require_mapping(',
            'receipt_inherited_v2r10_boundary = require_mapping(',
            '"software_acceptance_entered": True',
            'expected_inherited_v2r9_boundary = {',
            'inherited_v2r9_boundary = require_mapping(',
            'receipt_inherited_v2r9_boundary = require_mapping(',
            '"semuav.visdrone-v2r9-operator-observed-terminal-boundary.v1"',
            'predecessor_release_id = "release_20260920_visdrone_paper_v2r9"',
            'predecessor_run_name = "semuav_visdrone_v2r9"',
            '"SemUAV-VLM-VisDrone-Paper-Study-20260920-v2r9.zip"',
            '"deploy_semuav_visdrone_v2r9_20260920.sh"',
            '"visdrone_v2r9_deployment"',
            'deployment["temporal_feasibility_amendment"]',
            '"fresh_runtime_smoke_required": True',
            '"predecessor_smoke_rows_reused": False',
            '"temporal_validation_sequences"',
            '"temporal_test_sequences"',
            '"temporal_observations_per_sequence"',
            '"temporal_source_frame_stride"',
            '"temporal_target_rate_hz"',
            '"temporal_feasibility_amendment_record"',
            '"native_receiver_correction_record"',
            '"pre_validation_native_receiver_correction"',
            'manifest_methodology.get("native_receiver_correction")',
            '"semuav.visdrone-v2r10-source-authenticated-native-"',
            '"receiver-correction.v1"',
            '"authenticated_v2r9_and_v2r10_native_receiver_source"',
            '"actionless_l0_heartbeat_rejected_by_generic_semantic_"',
            '"action_path"',
            '"exception_message_template": (',
            '"affected_entrypoint": (',
            '"RegisteredVisDroneNativeReceiver.__call__"',
            '"affected_variant": "l0_cbor"',
            '"pixel_receiver_request_used_synthetic_time_and_partial_"',
            '"RegisteredVisDroneNativeReceiver._frame_action"',
            '"authoritative_pixel_control_fields": [',
            '"registered_l0_receiver_action": "NO_ACTION"',
            '"pixel_receiver_request_binds_authoritative_time_and_telemetry"',
            '"l3_roi_receiver_preserves_authoritative_telemetry"',
            '"registered_native_cache_l0_through_visual_and_video_end_to_end"',
            '"raw_experiment1_artifact_count": 132',
            '"raw_experiment1_inference_row_count": 4152',
            '"failure_variant": "l0_cbor"',
            'predecessor_evidence["runtime_smoke_schema_version"] != 4',
            '"semuav.runtime-smoke-subprocess.v2"',
            'predecessor_evidence["runtime_smoke_all_passed"] is not True',
            'predecessor_evidence["runtime_smoke_pass_count"] != 3',
            'predecessor_evidence["runtime_smoke_failure_count"] != 0',
            '"validation_input_gate_completed": True',
            '"exact_eligible_capacity_recomputed": True',
            '"v2r10_fresh_runtime_smoke_required": True',
            '"v2r10_raw_experiment1_recovery_authorized": True',
            '"v2r10_derived_results_recomputation_required": True',
            '"v2r10_downstream_results_recomputation_required": True',
            '"v2r9_derived_scientific_outputs_reused": False',
            'set(predecessor_evidence) != predecessor_evidence_required',
            '"authenticated_predecessor_v2r9": dict(predecessor_evidence)',
            '"authenticated_predecessor_v2r10": dict(',
            '"authenticated_predecessor_v2r11": dict(',
            'manifest["authenticated_predecessor_v2r10"]',
            'manifest["authenticated_predecessor_v2r11"]',
            '"predecessor_observed_terminal_boundary": dict(',
            '"temporal_feasibility_amendment": dict(',
            '"scientific_validation_authorized"',
            '"validation_or_test_images_or_labels_read"',
            'deployment["package_installation_performed"] is not False',
            'deployment["candidate_builder_invoked"] is not False',
            'deployment["candidate_environment_modified"] is not False',
            'deployment["scientific_contract_changed"] is not False',
            'deployment["candidate_requirements_changed"] is not False',
            'deployment_source_compatibility["runtime_smoke_input_schema"]',
            '"semuav.model-runtime-smoke-input.v3"',
            'deployment_source_compatibility["trusted_code_cache_receipt_schema"]',
            '"semuav.visdrone-trusted-code-cache.v2"',
            'deployment_source_compatibility["measurement_schema_version"]',
            '"semuav.repeated-inference-measurement.v2"',
            'deployment_source_compatibility["runner_version"]',
            '"semuav.visdrone-paper-runner.v7"',
            'candidate_guards["inherited_loader_environment_cleared"] is not True',
            'source["release_id"] != candidate_source_release_id',
            'Path(str(source["path"])).resolve() != candidate_source_root.resolve()',
            'requirements_bytes != candidate_requirements_bytes',
            'manifest["empirical_results_included"] is not False',
            'or ".." in PurePosixPath(relative_path).parts',
            'current_pip_freeze = subprocess.run(',
            'observed["run"] != observed["passed"]',
            'expected_kernel_name = "semuav-vlm-20260921-v2r13"',
            '"candidate_environment_reused_without_rebuild": True',
            '"scientific_contract_changed": False',
            '"candidate_requirements_changed": False',
            '"active_and_candidate_source_requirements_identical": True',
            '"candidate_source_manifest_sha256": candidate_manifest_sha256',
            '"schema_version": "semuav.visdrone-source-candidate-binding.v1"',
            "if require_inference_lock or INFERENCE_PIP_FREEZE.exists()",
        ):
            self.assertIn(token, candidate_source)
        predecessor_required_segment = candidate_source[
            candidate_source.index("predecessor_evidence_required = {"):
            candidate_source.index("predecessor_evidence = require_mapping(")
        ]
        for terminal_field in (
            "validation_input_gate_entered",
            "validation_data_identity_and_frame_availability_read",
            "validation_temporal_sample_written",
            "validation_caches_built",
            "scientific_experiments_run",
            "protocol_frozen",
            "untouched_test_run",
        ):
            self.assertNotIn(terminal_field, predecessor_required_segment)
        for digest in (
            "73b94ea64ac21fc61c7a3e0cec1038397c6dfc327c58d962ac96dd3052e48cf7",
            "c63bddc8676d05d5ee9a3f196daa407681177fb980385c4750fb60536baf7efd",
            "2df86df69fa36ae975f4f380b851dcd6fb28a59ff7868ff45f73f6dc806bd7dc",
            "3f6b863284313418bf21097bfa0e548f19b22fef11db1175560e09f9b3b48f28",
            "b5962f818a3ffa616e24bbf3ce188c150eeed0c0401ca101e9c483e2b13aef27",
            "ab90dc31f7bf9ae7957316f3e36058991272d78a8db1b729d68346bca46dab78",
            "424c57628cfbab7b15ce5546af997e6773db75ab8ea72274d0b76ccd827d5c10",
            "5b6aacbcd2b891c1bf968511d17b3385aff2233e99d0bae440f81bc7f977d5d2",
            "a3c9d33c87d2b620acd41be55d5b5494cdb75bcbe6a0abb82a5f625cce4674e1",
            "7642b89cd36a87faafc62c44e88baec5bb45d3ea8d7c3438e2ad0293aab585b5",
            "340dd59ca55a9be491ac806b50f9778a12d2b0b4a447485627aaf422dda05353",
            "6f83118bb62cd68c98e9ee2afd947022c04088b9329a0eeef832eba17c2fb08c",
            "947c5e1ad828eed9beaf6538f227fd871b406f3fab11bf6ce7202dabf23749b0",
            "ad45cd62384697243de0391bc929b05356ba4252d80fd93854f5d1a90d4a55ed",
            "01f18450cac8bc5d1edf4dd45cf7302e0b2903d53609e0850cddf723bbf69ec3",
        ):
            self.assertIn(digest, candidate_source)
        self.assertNotIn('"fresh_candidate_environment": True', candidate_source)
        for stale in (
            'deployment["authenticated_predecessor_v2r5"]',
            '"semuav.runtime-smoke-subprocess.v1"',
            '"moondream2": "FAIL_SCHEMA"',
            '"v2r5 predecessor archive"',
            '"v2r5 predecessor failed runtime-smoke report"',
        ):
            self.assertNotIn(stale, candidate_source)
        self.assertNotIn(
            'Path(str(source["path"])).resolve() != release_root',
            candidate_source,
        )
        self.assertNotIn(
            "source_compatibility = _verify_v2r2p1_source_compatibility(",
            candidate_source,
        )
        identity_gate = candidate_source.index("configured_receipt = Path(")
        deployment_read = candidate_source.index(
            "deployment_bytes, deployment_sha256 = read_and_hash(",
        )
        candidate_receipt_read = candidate_source.index(
            "candidate_receipt_bytes, candidate_receipt_sha256 = read_and_hash(",
        )
        self.assertLess(identity_gate, deployment_read)
        self.assertLess(deployment_read, candidate_receipt_read)

        runtime_guard_function = next(
            node for node in helper_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verified_runtime_guards"
        )
        runtime_guard_source = ast.get_source_segment(
            helper_cell, runtime_guard_function,
        ) or ""
        for token in (
            'required_absent = (\n        "BASH_ENV", "ENV", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH",',
            'expected_revision_guards = {',
            'authenticated_model_loader_environment.get(name)',
            'locked_revision_environment.get(name)',
            'observed_revision_guards != expected_revision_guards',
            '"authenticated_model_revisions": observed_revision_guards',
        ):
            self.assertIn(token, runtime_guard_source)

        prepare_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "_verify_candidate_ready_receipt(require_inference_lock=False)"
            in str(cell.get("source", ""))
        )
        early_receipt_gate = prepare_source.index(
            "_verify_candidate_ready_receipt(require_inference_lock=False)"
        )
        first_inference_lock = prepare_source.index(
            "_capture_or_verify_inference_runtime("
        )
        self.assertLess(early_receipt_gate, first_inference_lock)

        self.assertNotIn(
            "_retired_v2r3_candidate_receipt_acceptance_fixture", helper_cell,
        )
        self.assertNotIn(
            "_retired_v2r2p1_source_compatibility_acceptance_fixture", helper_cell,
        )

        seal_function = next(
            node for node in helper_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_inference_environment_seal_payload"
        )
        seal_source = ast.get_source_segment(helper_cell, seal_function) or ""
        self.assertIn(
            '"source_compatibility": candidate_ready["source_compatibility"]',
            seal_source,
        )

        capture_function = next(
            node for node in helper_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_capture_or_verify_inference_runtime"
        )
        capture_source = ast.get_source_segment(helper_cell, capture_function) or ""
        self.assertLess(
            capture_source.index("_require_exclusive_scientific_gpu("),
            capture_source.index("environment_report("),
        )
        self.assertLess(
            capture_source.index("lock_path.is_symlink()"),
            capture_source.index("environment_report("),
        )
        self.assertNotIn("INFERENCE_ENVIRONMENT_FINGERPRINT.read_text", capture_source)
        self.assertNotIn("INFERENCE_PIP_FREEZE.read_text", capture_source)

        smoke_loader = next(
            node for node in helper_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_load_frozen_runtime_smoke_input_receipt"
        )
        smoke_loader_source = ast.get_source_segment(helper_cell, smoke_loader) or ""
        self.assertIn("_read_regular_file_bytes(", smoke_loader_source)
        self.assertNotIn("MODEL_SMOKE_INPUT_RECEIPT.read_text", smoke_loader_source)
        prepare_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "existing_smoke_input = _load_frozen_runtime_smoke_input_receipt()"
            in str(cell.get("source", ""))
        )
        self.assertIn("MODEL_SMOKE_INPUT_RECEIPT.is_symlink()", prepare_source)
        self.assertIn(
            "existing_smoke_input = _load_frozen_runtime_smoke_input_receipt()",
            prepare_source,
        )

        read_function = next(
            node for node in helper_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_read_regular_file_bytes"
        )
        isolated_reader = ast.Module(body=[read_function], type_ignores=[])
        ast.fix_missing_locations(isolated_reader)
        reader_namespace = {"Path": Path, "StudyGateError": RuntimeError}
        exec(compile(isolated_reader, "regular-file-reader", "exec"), reader_namespace)
        read_regular = reader_namespace["_read_regular_file_bytes"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_target = root / "validation-derived.json"
            validation_target.write_text("{}", encoding="utf-8")
            for leaf_name in (
                "MODEL_RUNTIME_SMOKE_INPUT.json",
                "INFERENCE_ENVIRONMENT_FINGERPRINT.json",
                "INFERENCE_PIP_FREEZE.txt",
            ):
                leaf = root / leaf_name
                leaf.symlink_to(validation_target)
                with self.assertRaises(RuntimeError):
                    read_regular(leaf, label=leaf_name)

    def test_v2r10_predecessor_notebook_is_archive_bound_not_live_rehashed(
        self,
    ) -> None:
        helper_cell = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "def _verify_candidate_ready_receipt" in str(
                cell.get("source", "")
            )
        )
        helper_module = ast.parse(helper_cell)
        candidate_function = next(
            node for node in helper_module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_candidate_ready_receipt"
        )
        candidate_source = ast.get_source_segment(
            helper_cell, candidate_function,
        ) or ""
        self.assertIn(
            'source_predecessor_inventory_by_path.get(\n'
            '            "SemUAV_VLM_VisDrone_Paper_Study.ipynb"',
            candidate_source,
        )
        self.assertIn(
            'source_predecessor_notebook_record.get("sha256")\n'
            '        != source_predecessor_evidence["notebook_sha256"]',
            candidate_source,
        )
        self.assertIn(
            '"f35bde4c03536b97bf82ab719ad5a516225bcc321d5a878971410b4e86a16108"',
            candidate_source,
        )
        self.assertNotIn("v2r10 source predecessor notebook\"", candidate_source)
        self.assertNotIn(
            'source_predecessor_release_root\n'
            '                / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"',
            candidate_source,
        )

    def test_validation_requires_reuse_receipt_and_bound_three_model_smoke(self) -> None:
        validation_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "validation-only" in cell.get("metadata", {}).get("tags", [])
        )
        seal = validation_sources.index("_verify_inference_environment_seal()")
        reuse = validation_sources.index(
            "_verify_or_create_reused_v1_artifacts_receipt("
        )
        full_development = validation_sources.index(
            "_verify_full_reused_v1_development_bundle("
        )
        inference = validation_sources.index("run_validation_perception_and_frontier(")
        self.assertLess(seal, reuse)
        self.assertLess(reuse, full_development)
        self.assertLess(full_development, inference)
        self.assertIn("allow_create=False", validation_sources)
        self.assertIn("VALIDATION_3_OF_3_RUNTIME_SMOKE_GATE_PASS", validation_sources)
        self.assertIn(
            "POST_SEAL_FULL_DEVELOPMENT_BUNDLE_VERIFIED", validation_sources,
        )
        self.assertIn('"reused_v1_frozen_artifacts_receipt"', validation_sources)
        self.assertIn('"model_runtime_smoke_input"', validation_sources)
        self.assertIn('"runtime_smoke_canary"', validation_sources)
        self.assertIn('"detector_training_receipt": configured_detector_receipt', validation_sources)
        self.assertIn('"detector_training_receipt_sha256"', validation_sources)
        self.assertIn('"inference_environment_seal"', validation_sources)
        self.assertIn('"inference_environment_seal_sha256"', validation_sources)

    def test_prepare_seals_candidate_only_after_bound_three_model_smoke(self) -> None:
        prepare_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "prepare-only" in cell.get("metadata", {}).get("tags", [])
        )
        smoke_gate = prepare_sources.index(
            "verified_smoke = _verify_runtime_smoke_report(smoke_input)"
        )
        post_smoke_runtime = prepare_sources.index(
            "post_smoke_runtime = _capture_or_verify_inference_runtime(",
            smoke_gate,
        )
        post_smoke_reuse = prepare_sources.index(
            "post_smoke_reused_inputs = (", post_smoke_runtime,
        )
        same_receipt_gate = prepare_sources.index(
            'post_smoke_reused_inputs["receipt_sha256"]', post_smoke_reuse,
        )
        seal = prepare_sources.index(
            "scientific_environment_seal = _create_inference_environment_seal(",
            same_receipt_gate,
        )
        self.assertLess(smoke_gate, post_smoke_runtime)
        self.assertLess(post_smoke_runtime, post_smoke_reuse)
        self.assertLess(post_smoke_reuse, same_receipt_gate)
        self.assertLess(same_receipt_gate, seal)
        self.assertIn(
            'pre_smoke_reuse_receipt_sha256 = reused_inputs["receipt_sha256"]',
            prepare_sources[:prepare_sources.index("if RUN_MODEL_SMOKE:")],
        )
        self.assertIn(
            "reused_inputs=post_smoke_reused_inputs,",
            prepare_sources[seal:],
        )
        self.assertIn(
            "INFERENCE_ENVIRONMENT_SEALED_FOR_SCIENTIFIC_USE",
            prepare_sources,
        )
        before_smoke_branch = prepare_sources[:prepare_sources.index("if RUN_MODEL_SMOKE:")]
        self.assertNotIn("_create_inference_environment_seal(", before_smoke_branch)

        helper_source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "def _inference_environment_seal_payload" in str(cell.get("source", ""))
        )
        self.assertIn("def _read_verified_reused_artifacts_receipt", helper_source)
        self.assertIn("REUSED_ARTIFACTS_RECEIPT.is_symlink()", helper_source)
        self.assertIn("not REUSED_ARTIFACTS_RECEIPT.is_file()", helper_source)
        self.assertIn("expected_sha256=reused_inputs[\"receipt_sha256\"]", helper_source)
        self.assertIn('"sha256": str(reused_receipt_sha256)', helper_source)
        self.assertNotIn("hash_file(REUSED_ARTIFACTS_RECEIPT)", helper_source)

    def test_matching_three_model_smoke_report_passes_exact_provenance_gate(self) -> None:
        from semuav.perception import (
            VISDRONE_GENERATION_POLICY_SHA256,
            VISDRONE_GENERATION_POLICY_VERSION,
            VISDRONE_GENERATION_STRATEGY_BY_MODEL,
            VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256,
            prompt_for_mission,
            visdrone_assistant_prefill_for_model,
        )
        from semuav.registry import hash_file, hash_object
        from semuav.visdrone_models import (
            OFFICIAL_RUNTIME_MODEL_IDS,
            RUNTIME_SMOKE_ISOLATION_SCHEMA,
            RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
        )

        helper_cell = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "def _verify_runtime_smoke_report" in str(cell.get("source", ""))
        )
        module = ast.parse(helper_cell)
        helper = next(
            node for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_verify_runtime_smoke_report"
        )
        isolated = ast.Module(body=[helper], type_ignores=[])
        ast.fix_missing_locations(isolated)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_lock = root / "MODEL_LOCK.json"
            model_lock.write_text("{}\n", encoding="utf-8")
            smoke_path = root / "MODEL_RUNTIME_SMOKE.json"
            input_payload = {
                "source_image_sha256": "1" * 64,
                "request_sha256": "2" * 64,
                "detector_proposals_sha256": "3" * 64,
                "detector_proposal_count": 32,
                "input_contract": "semuav.visdrone-proposal-decision.v2",
                "request": {"observation_id": "registered-train-canary"},
            }
            model_specs = {
                "qwen3_vl_2b": ("repo/qwen-2b", "a" * 40),
                "smolvlm2_2b": ("repo/smolvlm", "b" * 40),
                "qwen3_vl_8b_upper_bound": ("repo/qwen-8b", "d" * 40),
            }
            self.assertEqual(tuple(model_specs), OFFICIAL_RUNTIME_MODEL_IDS)
            verified_lock = {
                "model_set_sha256": "4" * 64,
                "models": [
                    {
                        "model_id": model_id,
                        "repo": repo,
                        "resolved_revision": revision,
                    }
                    for model_id, (repo, revision) in model_specs.items()
                ],
            }
            prompt_sha256 = hash_object(
                prompt_for_mission("visdrone_urban").strip()
            )
            rows = []
            for index, (model_id, (repo, revision)) in enumerate(
                model_specs.items(), start=1,
            ):
                strategy = VISDRONE_GENERATION_STRATEGY_BY_MODEL[model_id]
                prefill = visdrone_assistant_prefill_for_model(model_id)
                raw_text = '{"e":true,"a":[],"l":[]}'
                self.assertTrue(raw_text.startswith(prefill))
                continuation = raw_text[len(prefill):]
                row = {
                    "model_id": model_id,
                    "repo": repo,
                    "revision": revision,
                    "status": "PASS",
                    "record_model_id": model_id,
                    "record_revision": revision,
                    "record_observation_id": input_payload["request"]["observation_id"],
                    "record_prompt_hash": prompt_sha256,
                    "input_contract": input_payload["input_contract"],
                    "request_sha256": input_payload["request_sha256"],
                    "detector_proposals_sha256": input_payload[
                        "detector_proposals_sha256"
                    ],
                    "detector_proposal_count": input_payload[
                        "detector_proposal_count"
                    ],
                    "image_sha256": input_payload["source_image_sha256"],
                    "decision_contract_version": input_payload["input_contract"],
                    "decision_core_valid": True,
                    "decision_core_json": json.dumps({
                        "e": True,
                        "a": [],
                        "l": [],
                    }),
                    "decision_context_sha256": "5" * 64,
                    "decision_proposal_count": input_payload[
                        "detector_proposal_count"
                    ],
                    "decision_composer_policy_sha256": (
                        VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                    ),
                    "raw_text": raw_text,
                    "raw_text_sha256": hashlib.sha256(
                        raw_text.encode("utf-8")
                    ).hexdigest(),
                    "parsed_packet_sha256": f"{index + 5:x}" * 64,
                    "input_tokens": 100,
                    "output_tokens": 12,
                    "terminated_by_eos": True,
                    "reached_token_cap": False,
                    "generation_policy_version": (
                        VISDRONE_GENERATION_POLICY_VERSION
                    ),
                    "generation_policy_sha256": (
                        VISDRONE_GENERATION_POLICY_SHA256
                    ),
                    "generation_strategy": strategy,
                    "generation_max_new_tokens": 256,
                    "assistant_prefill_characters": len(prefill),
                    "assistant_prefill_sha256": hashlib.sha256(
                        prefill.encode("utf-8")
                    ).hexdigest(),
                    "generated_continuation_sha256": hashlib.sha256(
                        continuation.encode("utf-8")
                    ).hexdigest(),
                    "worker_isolation_schema": RUNTIME_SMOKE_ISOLATION_SCHEMA,
                    "worker_result_valid": True,
                    "worker_returncode": 0,
                    "worker_signal": None,
                    "worker_timed_out": False,
                }
                rows.append(row)
            isolation_artifacts = [
                {"model_id": model_id, "index": index}
                for index, model_id in enumerate(model_specs)
            ]
            report = {
                "schema_version": RUNTIME_SMOKE_REPORT_SCHEMA_VERSION,
                "all_passed": True,
                "model_lock": str(model_lock.resolve()),
                "model_lock_sha256": hash_file(model_lock),
                "model_set_sha256": verified_lock["model_set_sha256"],
                "code_sha256": "7" * 64,
                "composer_policy_sha256": (
                    VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                ),
                "generation_policy_version": VISDRONE_GENERATION_POLICY_VERSION,
                "generation_policy_sha256": VISDRONE_GENERATION_POLICY_SHA256,
                "expected_decision_context_sha256": "5" * 64,
                "prompt_sha256": prompt_sha256,
                "image_sha256": input_payload["source_image_sha256"],
                "request_sha256": input_payload["request_sha256"],
                "detector_proposals_sha256": input_payload[
                    "detector_proposals_sha256"
                ],
                "detector_proposal_count": 32,
                "input_contract": input_payload["input_contract"],
                "execution_isolation": {
                    "schema_version": RUNTIME_SMOKE_ISOLATION_SCHEMA,
                    "strategy": "fresh_exec_per_locked_model",
                    "worker_environment": {},
                    "artifacts": isolation_artifacts,
                    "artifacts_sha256": hash_object(isolation_artifacts),
                },
                "models": rows,
            }
            smoke_path.write_text(json.dumps(report), encoding="utf-8")
            namespace = {
                "Path": Path,
                "json": json,
                "MODEL_SMOKE": smoke_path,
                "MODEL_LOCK": model_lock,
                "StudyGateError": RuntimeError,
                "require_runtime_smoke_pass": lambda path: None,
                "verify_huggingface_model_lock": lambda path, *, models: verified_lock,
                "cfg": SimpleNamespace(models={"max_new_tokens": 256}),
                "ctx": SimpleNamespace(code_hash="7" * 64),
                "hash_file": hash_file,
                "hash_object": hash_object,
                "hashlib": hashlib,
                "prompt_for_mission": prompt_for_mission,
                "VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256": (
                    VISDRONE_PROPOSAL_COMPOSER_POLICY_SHA256
                ),
                "VISDRONE_GENERATION_STRATEGY_BY_MODEL": (
                    VISDRONE_GENERATION_STRATEGY_BY_MODEL
                ),
                "visdrone_assistant_prefill_for_model": (
                    visdrone_assistant_prefill_for_model
                ),
                "RUNTIME_SMOKE_REPORT_SCHEMA_VERSION": (
                    RUNTIME_SMOKE_REPORT_SCHEMA_VERSION
                ),
                "RUNTIME_SMOKE_ISOLATION_SCHEMA": (
                    RUNTIME_SMOKE_ISOLATION_SCHEMA
                ),
                "REGISTERED_RUNTIME_SMOKE_MODEL_IDS": (
                    OFFICIAL_RUNTIME_MODEL_IDS
                ),
                "_verified_generation_policy": lambda: {
                    "version": VISDRONE_GENERATION_POLICY_VERSION,
                    "sha256": VISDRONE_GENERATION_POLICY_SHA256,
                },
            }
            exec(compile(isolated, "smoke-report-helper", "exec"), namespace)
            verified = namespace["_verify_runtime_smoke_report"](input_payload)
            self.assertEqual(verified, report)
            self.assertNotIn("moondream", json.dumps(report).lower())

            report["models"][0], report["models"][1] = (
                report["models"][1], report["models"][0]
            )
            smoke_path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact 3/3 PASS"):
                namespace["_verify_runtime_smoke_report"](input_payload)
            report["models"][0], report["models"][1] = (
                report["models"][1], report["models"][0]
            )
            smoke_path.write_text(json.dumps(report), encoding="utf-8")

            input_payload["input_contract"] = "changed-contract"
            with self.assertRaisesRegex(RuntimeError, "exact 3/3 PASS"):
                namespace["_verify_runtime_smoke_report"](input_payload)

    def test_validation_exports_complete_development_model_selection_evidence(self) -> None:
        validation_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "validation-only" in cell.get("metadata", {}).get("tags", [])
        )
        for token in (
            'selection.get("selection_summary", [])',
            'selection.get("calibration_gate", [])',
            'how="left"', 'validate="one_to_one"',
            '"calibration_eligible"',
            '"INELIGIBLE_NOT_AVAILABLE"',
            '"calibration_blocking_for_confirmatory"',
            '"DEVELOPMENT_VALIDATION_MODEL_SELECTION_ONLY"',
            '"validation_exp1_perception_summary"',
            '"validation_model_selection_gate"',
        ):
            self.assertIn(token, validation_sources)
        self.assertNotIn(
            'validation_manifest["visual_group_id"].astype(str).nunique()',
            validation_sources,
        )

    def test_validation_controller_cell_consumes_only_compact_runner_outputs(self) -> None:
        validation_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "validation-only" in cell.get("metadata", {}).get("tags", [])
        )
        self.assertIn('phase345["resource_summary"]', validation_sources)
        self.assertIn("display(validation_resource_summary)", validation_sources)
        self.assertNotIn('phase345["exp4_policy"]', validation_sources)
        self.assertNotIn('phase345["exp5_detail"]', validation_sources)
        self.assertNotIn("resource_summary(exp4_policy)", validation_sources)
        self.assertNotIn("exp4_policy is not None", validation_sources)
        self.assertIn("exp4_summary is not None and not exp4_summary.empty", validation_sources)

    def test_h2_resource_mismatch_is_nonblocking_but_audit_domain_is_exact(self) -> None:
        combined = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
        )
        self.assertNotIn(
            'if not validation_h2_matched_budget["passed"].astype(bool).all()',
            combined,
        )
        self.assertNotIn(
            'raise StudyGateError("H2 pairwise matched-budget validation gate failed")',
            combined,
        )
        for token in (
            'selection["policy_spec"]["required_h2_baselines"]',
            '"h2_comparator_and_resource_audits_registered"',
            'selection.get("h2_validation_unmatched_baselines", ())',
            '"NOT_ESTIMABLE" if audited_h2_unmatched else "MATCHED"',
            '"H2 validation family: NOT_ESTIMABLE; unmatched registered baselines:"',
        ):
            self.assertIn(token, combined)

        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("seven-sequence validation amendment", readme)
        self.assertIn("15-sequence untouched-test cohort", readme)
        self.assertIn("450 group-by-trace cells", readme)
        self.assertNotIn("bounded 30-group general cohort", readme)
        guide = (ROOT / "VISDRONE_PAPER_RUN_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn(
            "Do not change a model, prompt, sample, threshold, score weight, "
            "tie-break,\ncampaign cell, or hypothesis gate after seeing "
            "validation outcomes.",
            guide,
        )
        self.assertIn(
            "A null,\nnegative, failed preregistered hypothesis, or "
            "`NOT_ESTIMABLE` outcome remains a\nvalid scientific result",
            guide,
        )
        self.assertNotIn("still a mandatory fairness gate", guide)

    def test_freeze_binds_selection_and_test_exceptions_close_the_ledger(self) -> None:
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        for token in (
            '"selection": selection', '"candidate_profiles"',
            '"detector_native_receiver_cache"', '"registered_feature_cache"',
            'except StudyGateError as exc:', 'status="BLOCKED"',
            'except Exception as exc:', 'status="FAILED"',
            'restricted_artifacts=test_restricted_artifacts',
            'test_sample_gate_table.csv.gz', '"required_for": "study_completion"',
            'test_sample_cluster_manifest_PRE_RUN.csv.gz',
            '"required_for": "H3_estimability"', '"NOT_ESTIMABLE"',
            'cfg.evaluation["fully_paired_scene_trace_cells"]',
            '"NOT_APPLICABLE_NO_TARGET_EVIDENCE"',
        ):
            self.assertIn(token, combined)
        self.assertNotIn(
            'RESULTS_DIR / "test_sample_cluster_manifest.csv.gz"', combined,
        )

    def test_report_claim_receipt_is_two_level_and_not_estimable_is_structured(self) -> None:
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        provisional = combined.index("atomic_write_frame(PROVISIONAL_CLAIM_LEDGER")
        inner = combined.index("paper_artifact_manifest = write_artifact_hash_manifest", provisional)
        final = combined.index("atomic_write_frame(FINAL_CLAIM_LEDGER")
        outer = combined.index("paper_manifest = write_artifact_hash_manifest", final)
        receipt = combined.index("completion_receipt = atomic_write_json", outer)
        self.assertLess(provisional, inner)
        self.assertLess(inner, final)
        self.assertLess(final, outer)
        self.assertLess(outer, receipt)
        for token in (
            'execution_status == "NOT_ESTIMABLE"', '"count_gates": count_gates',
            '"temporal_group_trace_cells"',
            '"positive_visual_groups"',
            '"matched_coverage_selected_rows"',
            '"matched_coverage_group_trace_cells"',
            '"high_confidence_group_trace_cells"',
            'paper_artifact_manifest_sha256', 'paper_bundle_status',
            'ELIGIBLE_PENDING_COMPLETION_RECEIPT',
            'FINAL_COMPLETION_RECEIPT',
            'verify_paper_completion_receipt(',
            '"h1_fixed_grid_aggregates": report_frames[',
            '"test_h1_fixed_grid_group_aggregates"',
            '"h1_fixed_grid_audit": report_frames[',
            '"test_h1_fixed_grid_coverage_audit"',
            '"coverage_gate_pass", "high_confidence_gate_pass"',
            '"h4_count_gate_pass"',
            'cfg.evaluation["h4_high_confidence_threshold"]',
            'handoff.protocol["decisions"].get(\n                "inference_environment_fingerprint_sha256"',
        ):
            self.assertIn(token, combined)

    def test_claim_scope_and_paper_artifacts_are_explicit(self) -> None:
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        for token in (
            "accident detection", "emergency dispatch", "road obstruction",
            "physical proximity", "wildfire", "cross-dataset",
            "H1–H4", "H5", "PAPER_RESULTS_MANIFEST.json",
            "export_paper_tables", "generate_publication_figures",
            "registered_selected_vlm_claim_gate(selection)",
            '"selected_vlm_inference": real_vlm_inference',
            "768×768 aspect-preserving letterboxed RGB",
            "model-specific normalization",
            "H2 and H3 are therefore structurally `NOT_ESTIMABLE`",
            '"h3_structural_status": "NOT_ESTIMABLE"',
            '"blocking_protocol_gate"',
        ):
            self.assertIn(token, combined)

    def test_complete_generic_fallback_keeps_h3_underpower_fail_closed(self) -> None:
        from semuav.visdrone_workflow import build_readiness_claim_ledger

        ledger = build_readiness_claim_ledger(
            phase="report",
            evidence={
                "official_visdrone_receipt": True,
                "publisher_group_splits_passed": True,
                "real_detector_inference": True,
                "selected_vlm_inference": False,
                "validation_gates_passed": True,
                "protocol_frozen": True,
                "one_time_test_complete": True,
                "paper_artifacts_hashed": True,
            },
            selected_vlm_used=False,
            hypotheses={
                "H1": "PASS", "H2": "PASS",
                "H3": {
                    "status": "NOT_ESTIMABLE",
                    "reason": "registered independent-group count gate failed",
                    "count_gates": [
                        {"gate": "positive_groups", "observed": 100, "required": 140},
                        {"gate": "negative_groups", "observed": 73, "required": 73},
                    ],
                },
                "H4": "FAIL",
            },
        ).set_index("item_id")
        self.assertEqual(ledger.loc["visdrone_sensor_space_results", "status"], "ELIGIBLE")
        self.assertEqual(ledger.loc["selected_vlm_contribution", "status"], "BLOCKED")
        self.assertEqual(ledger.loc["H1", "status"], "PASS")
        self.assertEqual(ledger.loc["H3", "status"], "NOT_ESTIMABLE")
        self.assertFalse(bool(ledger.loc["H3", "eligible"]))

    def test_pre_open_recomputes_exact_environment_and_distribution_lock(self) -> None:
        combined = "\n".join(str(cell.get("source", "")) for cell in self.notebook["cells"])
        test_start = combined.index('if PHASE == "test" and EXECUTE_PHASE:')
        test_source = combined[test_start:]
        authorization = test_source.index("authorize_test_once(ctx, PROTOCOL)")
        before_open = test_source[:authorization]
        for token in (
            "_verify_inference_environment_seal()",
            "_verify_or_create_reused_v1_artifacts_receipt(",
            "preopen_runtime = preopen_scientific_environment[\"runtime\"]",
            '"scientific_seal_sha256="',
        ):
            self.assertIn(token, before_open)
        self.assertLess(
            before_open.index("_verify_inference_environment_seal()"),
            before_open.index("_verify_or_create_reused_v1_artifacts_receipt("),
        )
        self.assertNotIn("DETECTOR_TRAINING_PIP_FREEZE.read_text", before_open)

    def test_temporal_samples_are_phase_isolated_persisted_and_runner_bound(self) -> None:
        validation_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "validation-only" in cell.get("metadata", {}).get("tags", [])
        )
        for token in (
            "select_registered_temporal_sample_from_config(",
            'split="validation"',
            "write_registered_temporal_sample(",
            'temporal_manifest=validation_inputs["temporal_manifest"]',
            '"registered_validation_temporal_sample"',
            '"temporal_sampling": dict(cfg.benchmark["temporal_sampling"])',
            'int(validation_temporal_spec["sequences_per_split"]) != 7',
            'int(test_temporal_spec["sequences_per_split"]) != 15',
            'test_trace_design_realizations != int(',
            "derived_validation_paired_cells = (",
            "registered_validation_paired_cells != 84",
            "derived_validation_paired_cells != registered_validation_paired_cells",
            "validation_temporal_groups * validation_trace_realizations",
            "V2R13_TEMPORAL_CARTESIAN_DESIGN_VERIFIED",
            "V2R13_TEMPORAL_ELIGIBLE_CAPACITY_VERIFIED",
            '"eligible_sequence_capacity"',
            "pd.testing.assert_frame_equal(",
            "validation_temporal_selected.reset_index(drop=True)",
            "validation_temporal_audit.reset_index(drop=True)",
            "check_exact=True",
            '"temporal_feasibility_amendment": dict(',
            '"temporal_feasibility_amendment_record": (',
        ):
            self.assertIn(token, validation_sources)

        test_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "test-only" in cell.get("metadata", {}).get("tags", [])
        )
        for token in (
            'preview_decisions.get("temporal_feasibility_amendment")',
            '!= temporal_feasibility_amendment',
            'preview_decisions.get("temporal_sampling")',
            '!= dict(cfg.benchmark["temporal_sampling"])',
            '"validation_fully_paired_scene_trace_cells"',
            '"validation_trace_realizations"',
            '"planned_test_temporal_visual_groups"',
            '"planned_test_temporal_group_trace_cells"',
        ):
            self.assertIn(token, test_sources)

        test_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "test-only" in cell.get("metadata", {}).get("tags", [])
        )
        opened = test_sources.index("authorize_test_once(ctx, PROTOCOL)")
        permit = test_sources.index("permit = load_untouched_test_permit", opened)
        selected = test_sources.index("test_temporal_selected", permit)
        persisted = test_sources.index("write_registered_temporal_sample(", selected)
        runner = test_sources.index("run_confirmatory_test_once(", persisted)
        self.assertLess(opened, permit)
        self.assertLess(permit, selected)
        self.assertLess(selected, persisted)
        self.assertLess(persisted, runner)
        self.assertIn("test_restricted_artifacts.extend(test_temporal_paths.values())", test_sources)
        self.assertIn("temporal_manifest=test_temporal_manifest", test_sources)
        self.assertIn("temporal_priority_positive_units", test_sources)
        self.assertIn("temporal_priority_negative_units", test_sources)
        self.assertIn('test_temporal_manifest["critical"]', test_sources)
        self.assertIn('test_outputs["restricted_raw_partition_artifacts"]', test_sources)
        self.assertIn("test_restricted_artifacts.extend(restricted_raw_partitions)", test_sources)

    def test_validation_runner_receives_both_registered_trace_splits(self) -> None:
        validation_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "validation-only" in cell.get("metadata", {}).get("tags", [])
        )
        self.assertIn(
            'validation_traces = all_traces[all_traces["split"].eq("validation")].copy()',
            validation_sources,
        )
        self.assertIn(
            'test_trace_design = all_traces[all_traces["split"].eq("test")].copy()',
            validation_sources,
        )
        self.assertIn('"traces": all_traces', validation_sources)
        self.assertNotIn('"traces": validation_traces', validation_sources)

    def test_report_loads_only_exact_bounded_compact_allowlist(self) -> None:
        report_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "report-only" in cell.get("metadata", {}).get("tags", [])
        )
        for token in (
            "report_csv_contract = {",
            '"policy_pair_cells.csv.gz"',
            '"test_temporal_cluster_manifest.csv.gz"',
            '"policy_reduction": policy_reduction_frames',
            '"temporal_manifest": report_frames["test_temporal_cluster_manifest"]',
            "max_report_rows = 250_000",
            "max_report_compressed_bytes = 64 * 1024 * 1024",
            "max_report_metadata_bytes = 8 * 1024 * 1024",
            "nrows=max_report_rows + 1",
            "forbidden_report_artifacts",
            'handoff.artifact("policy_reduction_receipt.json")',
            "receipt_partition_bindings != index_partition_bindings",
            '"h4_campaign_selected_trial_rows_materialized"',
            'resolve_frozen_artifact(handoff.protocol, artifact_key)',
            '"validation_exp1_perception_summary"',
            '"validation_model_selection_gate"',
        ):
            self.assertIn(token, report_sources)
        self.assertNotIn('report_frames["test_policy_trials"]', report_sources)
        self.assertNotIn('"test_calibration_report.csv.gz"', report_sources)
        self.assertNotIn("report_prefixes", report_sources)

    def test_report_rechecks_and_binds_exact_runtime(self) -> None:
        report_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "report-only" in cell.get("metadata", {}).get("tags", [])
        )
        for token in (
            '_require_exclusive_scientific_gpu("report_runtime_preflight")',
            "report_preflight = environment_report(",
            "report_environment = environment_fingerprint(report_preflight)",
            '"pip", "freeze", "--all"',
            "Report runtime differs from the frozen validation/test environment",
            "REPORT_ENVIRONMENT_RECEIPT",
            '"report_environment_receipt_sha256"',
            "expected_report_environment_receipt=report_environment_receipt",
        ):
            self.assertIn(token, report_sources)
        self.assertLess(
            report_sources.index(
                '_require_exclusive_scientific_gpu("report_runtime_preflight")'
            ),
            report_sources.index("report_preflight = environment_report("),
        )

    def test_generic_runtime_preflight_never_touches_cuda_while_paused(self) -> None:
        source = next(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if "source_receipt = VISDRONE_ROOT" in str(cell.get("source", ""))
        )
        execute_gate = source.index(
            'if EXECUTE_PHASE and PHASE in {"prepare", "validation", "test"}:'
        )
        exclusive_gate = source.index(
            '_require_exclusive_scientific_gpu(f"{PHASE}_initial_runtime_preflight")',
            execute_gate,
        )
        cuda_preflight = source.index("preflight = environment_report(", execute_gate)
        deferred = source.index(
            'elif PHASE in {"prepare", "validation", "test"}:', cuda_preflight,
        )
        self.assertLess(execute_gate, exclusive_gate)
        self.assertLess(exclusive_gate, cuda_preflight)
        self.assertLess(cuda_preflight, deferred)

    def test_report_rejects_post_freeze_notebook_source_changes(self) -> None:
        report_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "report-only" in cell.get("metadata", {}).get("tags", [])
        )
        handoff = report_sources.index("load_report_safe_finalized_test_handoff(")
        source_gate = report_sources.index(
            "current_notebook_source_hash = notebook_source_hash(NOTEBOOK_PATH)",
            handoff,
        )
        recompute = report_sources.index(
            "recompute_hypotheses_from_frozen_results(", source_gate,
        )
        self.assertLess(handoff, source_gate)
        self.assertLess(source_gate, recompute)
        self.assertIn(
            "Notebook cell sources changed after protocol freeze; report is blocked",
            report_sources,
        )

    def test_existing_completion_receipt_is_verified_then_blocks_report_mutation(self) -> None:
        report_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "report-only" in cell.get("metadata", {}).get("tags", [])
        )
        stale = report_sources.index("if FINAL_COMPLETION_RECEIPT.exists():")
        load = report_sources.index("load_report_safe_finalized_test_handoff", stale)
        self.assertLess(stale, load)
        self.assertIn("Paper report is already finalized and immutable", report_sources)

    def test_completion_counts_only_estimable_hypotheses_as_evaluated(self) -> None:
        report_sources = "\n".join(
            str(cell.get("source", "")) for cell in self.notebook["cells"]
            if "report-only" in cell.get("metadata", {}).get("tags", [])
        )
        for token in (
            '"hypotheses_registered", "hypotheses_evaluated"',
            '"hypotheses_not_estimable"',
            'str.upper().eq("COMPLETE").sum()',
            "expected_not_estimable = int(",
            "Completion hypothesis counts disagree with frozen hypothesis status",
        ):
            self.assertIn(token, report_sources)

    def test_full_proposal_notebook_is_not_replaced(self) -> None:
        locked = ROOT / "SemUAV_VLM_Full_Study.ipynb"
        locked_builder = ROOT / "build_full_notebook.py"
        self.assertTrue(locked.is_file())
        self.assertTrue(locked_builder.is_file())
        self.assertNotEqual(NOTEBOOK.name, locked.name)
        self.assertNotEqual(BUILDER.name, locked_builder.name)

    def test_run_guide_has_concrete_v2r13_release_and_immediate_smoke_gate(self) -> None:
        guide = RUN_GUIDE.read_text(encoding="utf-8")
        self.assertNotIn("REPLACE_WITH", guide)
        self.assertIn("release_20260921_visdrone_paper_v2r13", guide)
        self.assertIn("semuav_visdrone_v2r13", guide)
        self.assertIn("release_20260912_visdrone_inference_candidate_v2r3", guide)
        self.assertIn("semuav-vlm-20260921-v2r13", guide)
        self.assertIn("modules-v2r13", guide)
        self.assertIn(
            'JUPYTER_DATA_DIR="$STUDY_ROOT/.cache/jupyter/data"', guide,
        )
        self.assertNotIn(".cache/jupyter/data-v2", guide)
        self.assertNotIn(".cache/jupyter/data-v2r1", guide)
        self.assertIn("MODEL_RUNTIME_SMOKE_3_OF_3_PASS", guide)
        self.assertIn(
            "The deployer is source-only.",
            guide,
        )
        exact_ids = (
            "qwen3_vl_2b",
            "smolvlm2_2b",
            "qwen3_vl_8b_upper_bound",
        )
        ordered_positions = [
            guide.index(f"{index}. `{model_id}`", guide.index("The v2r13 model population"))
            for index, model_id in enumerate(exact_ids, start=1)
        ]
        self.assertEqual(ordered_positions, sorted(ordered_positions))
        self.assertIn("must run all three retained models again", guide)
        self.assertIn("A 2/3 result or skipped upper bound keeps validation and test closed", guide)
        self.assertIn("Do not rerun v2r5 or copy any of its PASS rows", guide)
        self.assertIn("v2r5 Starmie tokenizer cache", guide)
        self.assertIn("historical failed-run evidence only", guide)
        self.assertIn("v2r13 does not acquire, export, bind, or\nverify a Moondream tokenizer", guide)
        correction_path = ROOT / "PRE_VALIDATION_RUNTIME_CORRECTION_20260912_V2R4.md"
        self.assertTrue(correction_path.is_file())
        correction = correction_path.read_text(encoding="utf-8")
        for token in (
            "prospective development-only correction",
            "No validation or test image",
            "V2R3_RUNTIME_SMOKE=FAILED_INCOMPLETE",
            "semuav.visdrone-generation-policy.v4",
            "qwen_open_brace_assistant_continuation",
            "smol_native_double_quote_assistant_prefix",
            "release_20260912_visdrone_paper_v2r4",
            "semuav_visdrone_v2r4",
            "release_20260912_visdrone_inference_candidate_v2r3",
            "semuav-vlm-20260912-v2r4",
            "scientific_contract_changed=true",
            "NEXT_GATE=FRESH_V2R4_FOUR_MODEL_DEVELOPMENT_RUNTIME_SMOKE",
        ):
            self.assertIn(token, correction)


if __name__ == "__main__":
    unittest.main()

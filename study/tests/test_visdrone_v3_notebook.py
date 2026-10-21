"""V3 notebook orchestration contracts, without a GPU or scientific data."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"


def code_cells():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def source(cell):
    value = cell["source"]
    return "".join(value) if isinstance(value, list) else value


def notebook_function(name, namespace=None):
    for cell in code_cells():
        tree = ast.parse(source(cell))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == name:
                module = ast.Module(body=[node], type_ignores=[])
                scope = dict(namespace or {})
                exec(compile(module, str(NOTEBOOK), "exec"), scope)
                return scope[name], scope
    raise AssertionError(f"Notebook helper absent: {name}")


def notebook_assignments(names):
    scope = {}
    for cell in code_cells():
        for node in ast.parse(source(cell)).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names:
                scope[node.targets[0].id] = ast.literal_eval(node.value)
    return scope


class V3NotebookTests(unittest.TestCase):
    def test_every_code_cell_is_valid_python_and_current_kernel(self):
        for cell in code_cells():
            ast.parse(source(cell))
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "semuav-vlm-20261013-v3r2")

    def test_long_runners_use_synchronous_experiment_callback(self):
        required = {
            "run_validation_perception_and_frontier",
            "run_validation_network_and_controller",
            "run_confirmatory_test_once",
        }
        observed = set()
        for cell in code_cells():
            for node in ast.walk(ast.parse(source(cell))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in required:
                    callbacks = [item.value for item in node.keywords if item.arg == "on_experiment_complete"]
                    self.assertEqual(len(callbacks), 1)
                    self.assertIsInstance(callbacks[0], ast.Name)
                    self.assertEqual(callbacks[0].id, "_show_experiment_results")
                    observed.add(node.func.id)
        self.assertEqual(observed, required)

    def test_validation_six_is_presented_before_seven_runs(self):
        cells = code_cells()
        six_index = next(i for i, cell in enumerate(cells) if "experiment-6" in cell["metadata"].get("tags", []) and "validation-only" in cell["metadata"].get("tags", []))
        seven_index = next(i for i, cell in enumerate(cells) if "experiment-7" in cell["metadata"].get("tags", []) and "validation-only" in cell["metadata"].get("tags", []))
        self.assertLess(six_index, seven_index)
        self.assertIn('_show_experiment_results(6, "validation"', source(cells[six_index]))
        self.assertNotIn("experiment7_readiness(", source(cells[six_index]))
        self.assertIn("experiment7_readiness(", source(cells[seven_index]))

    def test_report_seven_result_then_figure_cells_in_order(self):
        observed = []
        for cell in code_cells():
            tags = cell["metadata"].get("tags", [])
            if "report-only" not in tags:
                continue
            experiment = [int(tag.split("-")[1]) for tag in tags if tag.startswith("experiment-")]
            if not experiment:
                continue
            self.assertEqual(len(experiment), 1)
            observed.extend(experiment)
            text = source(cell)
            self.assertLess(text.index("display(report_frames["), text.index("generate_publication_figures("))
            self.assertIn(f"experiment={experiment[0]}, strict=True", text)
            self.assertIn(f"_require_experiment_figures({experiment[0]}, experiment_figures)", text)
        self.assertEqual(observed, list(range(1, 8)))

    def _preview_scope(self, generated):
        events = []
        issues = []
        constants = notebook_assignments({"EXPERIMENT_TITLES", "REQUIRED_EXPERIMENT_FIGURES"})
        def generate(**kwargs):
            events.append(("generate", kwargs))
            if isinstance(generated, Exception):
                raise generated
            return generated
        env = {
            **constants, "pd": pd, "Markdown": str,
            "display": lambda item: events.append(("display", item)),
            "print": lambda *items: events.append(("print", items)),
            "generate_publication_figures": generate,
            "plt": SimpleNamespace(close=lambda figure: events.append(("close", figure))),
            "ctx": SimpleNamespace(is_scientific=True),
            "FIGURE_DIR": Path("/test/figures"), "RESULTS_DIR": Path("/test/results"),
            "StudyGateError": RuntimeError, "live_figure_errors": issues,
            "atomic_write_json": lambda path, payload: events.append(("receipt", payload)),
        }
        require, _ = notebook_function("_require_experiment_figures", env)
        env["_require_experiment_figures"] = require
        callback, _ = notebook_function("_show_experiment_results", env)
        return callback, events, issues

    def test_callback_displays_results_then_figures_and_closes(self):
        callback, events, issues = self._preview_scope({"exp2_pareto_frontier": "figure-object"})
        original = pd.DataFrame({"mode": ["L5", "L5"], "variant": ["jpeg", "jpeg"], "payload_bytes": [20, 40]})
        before = original.copy(deep=True)
        callback(2, "validation", {"exp2_frontier": original})
        pd.testing.assert_frame_equal(original, before)
        table_position = next(i for i, (kind, value) in enumerate(events) if kind == "display" and isinstance(value, pd.DataFrame))
        generate_position = next(i for i, (kind, _) in enumerate(events) if kind == "generate")
        self.assertLess(table_position, generate_position)
        self.assertIn(("display", "figure-object"), [(k,v) for k,v in events if not isinstance(v, pd.DataFrame)])
        self.assertIn(("close", "figure-object"), events)
        kwargs = events[generate_position][1]
        self.assertEqual(kwargs["output_dir"], Path("/test/experiment_previews/validation/experiment_02"))
        self.assertEqual(kwargs["experiment"], 2)
        self.assertTrue(kwargs["strict"])
        self.assertIn("DEVELOPMENT", kwargs["evidence_label"])
        self.assertFalse(issues)

    def test_plot_failure_is_explicit_and_does_not_consume_test(self):
        callback, events, issues = self._preview_scope(RuntimeError("PNG write failed"))
        callback(3, "test", {"exp3_summary": pd.DataFrame({"measured": [1]})})
        self.assertEqual(issues, [{"experiment": 3, "phase": "test", "error_type": "RuntimeError", "message": "PNG write failed"}])
        self.assertTrue(any(kind == "receipt" for kind, _ in events))
        self.assertTrue(any(kind == "print" and "FIGURE_PREVIEW_FAILED" in str(value) for kind, value in events))

    def test_disconnected_header_display_does_not_fail_finalized_test(self):
        callback, events, issues = self._preview_scope({"exp7_readiness_hardware": "figure-object"})
        def disconnected(_):
            raise RuntimeError("frontend disconnected")
        callback.__globals__["display"] = disconnected
        callback(7, "test", {"exp7_readiness": pd.DataFrame({"status": ["PASS"]})})
        self.assertEqual(issues[0]["message"], "frontend disconnected")
        self.assertFalse(any(kind == "generate" for kind, _ in events))
        self.assertTrue(any(kind == "receipt" for kind, _ in events))

    def test_missing_expected_figure_is_explicit(self):
        callback, _, issues = self._preview_scope({})
        callback(4, "validation", {})
        self.assertEqual(len(issues), 1)
        self.assertIn("exp4_utility_resources", issues[0]["message"])
        self.assertIn("exp4_mode_mix", issues[0]["message"])

    def test_deferred_tc_does_not_disable_any_active_runtime_check(self):
        failure_check, _ = notebook_function("_required_environment_failures", {"StudyGateError": RuntimeError})
        frame = pd.DataFrame([
            {"item": "tc_netem", "status": "FAIL", "required_for": "experiment_7b"},
            {"item": "cuda", "status": "FAIL", "required_for": "real_profiles"},
            {"item": "ffmpeg_encoder:libx264", "status": "FAIL", "required_for": "experiments_2_7"},
            {"item": "package:numpy", "status": "PASS", "required_for": "all"},
        ])
        self.assertEqual(failure_check(frame), ["cuda", "ffmpeg_encoder:libx264"])
        frame.loc[0, "required_for"] = "full"
        self.assertEqual(failure_check(frame), ["tc_netem", "cuda", "ffmpeg_encoder:libx264"])
        with self.assertRaises(RuntimeError):
            failure_check(pd.DataFrame())

    def test_parameter_cell_authenticates_before_scientific_imports(self):
        parameter = next(source(c) for c in code_cells() if "parameters" in c["metadata"].get("tags", []))
        self.assertIn("SEMUAV_V3_DEPLOYMENT_RECEIPT_SHA256", parameter)
        self.assertLess(parameter.index("hashlib.sha256(bootstrap_bytes)"), parameter.index("exec(compile(bootstrap_bytes"))
        self.assertIn('V3_DEPLOYMENT_API["verify_source"]', parameter)
        self.assertNotIn("from semuav", parameter)
        self.assertNotIn("import semuav", parameter)
        assigned = {
            node.targets[0].id for node in ast.parse(parameter).body
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }
        self.assertTrue({"PHASE", "EXECUTE_PHASE", "FREEZE_PROTOCOL", "AUTHORIZE_TEST_ONCE"}.issubset(assigned))

    def test_saved_outputs_and_safe_controls_remain_testable(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "saved runtime output"}]
            cell["execution_count"] = 1
            if "parameters" in cell["metadata"].get("tags", []):
                lines = source(cell).splitlines(keepends=True)
                replacements = []
                for node in ast.parse(source(cell)).body:
                    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        name = node.targets[0].id
                        if name in {"PHASE", "EXECUTE_PHASE", "FREEZE_PROTOCOL"}:
                            replacements.append((node.lineno - 1, node.end_lineno, f'{name} = ' + ('"report"' if name == "PHASE" else "True") + "\n"))
                for start, end, replacement in sorted(replacements, reverse=True):
                    lines[start:end] = [replacement]
                cell["source"] = "".join(lines)
        with tempfile.TemporaryDirectory() as temporary:
            edited = Path(temporary) / "executed.ipynb"
            edited.write_text(json.dumps(notebook), encoding="utf-8")
            with mock.patch.object(sys.modules[__name__], "NOTEBOOK", edited):
                self.test_every_code_cell_is_valid_python_and_current_kernel()
                self.test_parameter_cell_authenticates_before_scientific_imports()
                self.test_raw_recovery_is_validation_only_after_actual_seal()
                self.test_report_seven_result_then_figure_cells_in_order()

    def test_raw_recovery_is_validation_only_after_actual_seal(self):
        occurrences = []
        for cell in code_cells():
            text = source(cell)
            if '["configure_recovery"]' not in text:
                continue
            occurrences.append(cell)
            self.assertIn("validation-only", cell["metadata"].get("tags", []))
            self.assertIn('if PHASE == "validation" and EXECUTE_PHASE:', text)
            self.assertIn('phase="validation"', text)
            self.assertLess(text.index("_verify_inference_environment_seal()"), text.index('["configure_recovery"]'))
            self.assertLess(text.index("_verify_full_reused_v1_development_bundle()"), text.index('["configure_recovery"]'))
        self.assertEqual(len(occurrences), 1)


if __name__ == "__main__":
    unittest.main()

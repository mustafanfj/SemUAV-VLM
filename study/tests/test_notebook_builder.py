from __future__ import annotations

import copy
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from semuav.registry import StudyGateError


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "build_full_notebook.py"


_VOLATILE_CELL_METADATA = frozenset({
    "collapsed",
    "execution",
    "jupyter",
    "scrolled",
    "trusted",
})
_LIVE_SELF_TEST_MARKER_LINE = (
    '    env={**os.environ, "SEMUAV_LIVE_NOTEBOOK_SELFTEST": "1"},\n'
)


def _normalized_source(value: object) -> object:
    if isinstance(value, list):
        value = "".join(str(part) for part in value)
    if isinstance(value, str):
        # Backward compatibility for the already-open hotfix1 notebook.  The
        # marker only selects live-safe test behavior; it has no study meaning.
        # Exact pre-delivery comparison still requires it in the new artifact.
        return value.replace(_LIVE_SELF_TEST_MARKER_LINE, "")
    return value


def _python_major_minor(value: object) -> object:
    if not isinstance(value, str):
        return value
    parts = value.split(".")
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return ".".join(parts[:2])
    return value


def _notebook_contract(notebook: dict[str, object]) -> dict[str, object]:
    """Return the authored notebook contract, excluding live runtime state.

    Jupyter legitimately adds cell IDs, execution counters, outputs, timing/trust
    metadata, widget state, and extra language metadata while a notebook is open.
    Those fields must not make the notebook's own acceptance suite fail.  Cell
    sources and order, authored cell metadata (especially parameter tags), the
    declared kernel, language, and notebook-format versions remain enforced.
    """

    notebook_metadata = notebook.get("metadata", {})
    if not isinstance(notebook_metadata, dict):
        notebook_metadata = {}
    kernelspec = notebook_metadata.get("kernelspec", {})
    if not isinstance(kernelspec, dict):
        kernelspec = {}
    language_info = notebook_metadata.get("language_info", {})
    if not isinstance(language_info, dict):
        language_info = {}

    cells = []
    raw_cells = notebook.get("cells", [])
    if not isinstance(raw_cells, list):
        raw_cells = []
    for cell in raw_cells:
        if not isinstance(cell, dict):
            cells.append({"invalid_cell": cell})
            continue
        raw_metadata = cell.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raw_metadata = {"invalid_metadata": raw_metadata}
        authored_metadata = {
            key: value
            for key, value in raw_metadata.items()
            if key not in _VOLATILE_CELL_METADATA and key != "tags"
        }
        authored_metadata["tags"] = list(raw_metadata.get("tags", []))
        cells.append({
            "cell_type": cell.get("cell_type"),
            "source": _normalized_source(cell.get("source", "")),
            "metadata": authored_metadata,
        })

    return {
        "nbformat": notebook.get("nbformat"),
        "nbformat_minor": notebook.get("nbformat_minor"),
        "metadata": {
            "kernelspec": {
                "display_name": kernelspec.get("display_name"),
                "language": kernelspec.get("language"),
                "name": kernelspec.get("name"),
            },
            "language_info": {
                "name": language_info.get("name"),
                "version": _python_major_minor(language_info.get("version")),
            },
        },
        "cells": cells,
    }


def _delivery_cleanliness_errors(notebook: dict[str, object]) -> list[str]:
    """Describe runtime state that must not ship in a generated notebook."""

    errors: list[str] = []
    for index, cell in enumerate(notebook.get("cells", [])):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        if cell.get("execution_count") is not None:
            errors.append(f"cell {index} has an execution count")
        if cell.get("outputs", []) != []:
            errors.append(f"cell {index} has stored outputs")
        metadata = cell.get("metadata", {})
        if isinstance(metadata, dict):
            volatile = sorted(set(metadata).intersection(_VOLATILE_CELL_METADATA))
            if volatile:
                errors.append(f"cell {index} has runtime metadata: {volatile}")
    return errors


def _cell_id_errors(
    notebook: dict[str, object],
    *,
    require_all: bool,
) -> list[str]:
    """Validate notebook-format cell IDs without binding live IDs to builder IDs."""

    errors: list[str] = []
    seen: set[str] = set()
    for index, cell in enumerate(notebook.get("cells", [])):
        if not isinstance(cell, dict):
            continue
        cell_id = cell.get("id")
        if cell_id is None:
            if require_all:
                errors.append(f"cell {index} has no id")
            continue
        if not isinstance(cell_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cell_id):
            errors.append(f"cell {index} has invalid id {cell_id!r}")
            continue
        if cell_id in seen:
            errors.append(f"cell {index} duplicates id {cell_id!r}")
        seen.add(cell_id)
    return errors


def _requires_exact_delivery_match(
    notebook: dict[str, object],
    *,
    live_self_test_marker: str,
) -> bool:
    """Use byte equality only for a clean artifact outside a live self-test.

    The runtime-state fallback is required for an already-open notebook created
    before the explicit self-test marker was added to its acceptance cell.
    Authored-contract comparison is unconditional and remains the integrity gate.
    """

    return (
        live_self_test_marker != "1"
        and not _delivery_cleanliness_errors(notebook)
    )


class NotebookBuilderOperationalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary_builder = Path(directory) / BUILDER.name
            namespace = {
                "__file__": str(temporary_builder),
                "__name__": "notebook_builder_test",
                "print": lambda *args, **kwargs: None,
            }
            source = BUILDER.read_text(encoding="utf-8")
            exec(compile(source, str(BUILDER), "exec"), namespace)
            cls.generated_notebook_text = (
                Path(directory) / "SemUAV_VLM_Full_Study.ipynb"
            ).read_text(encoding="utf-8")
            cls.notebook = json.loads(cls.generated_notebook_text)

    @classmethod
    def _cell(cls, needle: str) -> str:
        matches = [
            str(cell.get("source", ""))
            for cell in cls.notebook["cells"]
            if needle in str(cell.get("source", ""))
        ]
        if len(matches) != 1:
            raise AssertionError(f"Expected one notebook cell containing {needle!r}, got {len(matches)}")
        return matches[0]

    def _run_acquisition_cell(
        self,
        *,
        action: str,
        profile: str,
        acquisition_root: Path,
        explicit_root: bool,
        prepare_result: pd.DataFrame | None = None,
    ) -> dict[str, object]:
        calls: dict[str, list[object]] = {
            "status": [], "download": [], "prepare": [], "display": [],
        }
        successful_prepare = pd.DataFrame([
            {"dataset": "visdrone", "status": "native_layout", "detail": "ready"},
        ])

        def status(root: Path) -> pd.DataFrame:
            calls["status"].append(Path(root))
            return pd.DataFrame([{"dataset": "visdrone", "present": False}])

        def download(root: Path, datasets: tuple[str, ...], *, accept_terms: bool) -> pd.DataFrame:
            calls["download"].append((Path(root), datasets, accept_terms))
            return pd.DataFrame([{"dataset": "visdrone", "status": "downloaded"}])

        def prepare(root: Path, datasets: tuple[str, ...]) -> pd.DataFrame:
            calls["prepare"].append((Path(root), datasets))
            return (prepare_result if prepare_result is not None else successful_prepare).copy()

        namespace = {
            "DATASETS": ("visdrone",),
            "SUPPORTED_DATASETS": ("visdrone", "uavid", "uavreason", "flame3"),
            "DATASET_ACTION": action,
            "RUN_PROFILE": profile,
            "ACQUISITION_ROOT": acquisition_root.resolve(),
            "ACQUISITION_ROOT_EXPLICIT": explicit_root,
            "ACCEPT_DATASET_TERMS": True,
            "StudyGateError": StudyGateError,
            "dataset_source_table": lambda: pd.DataFrame([{"dataset": "visdrone"}]),
            "disk_preflight": lambda root, datasets: {
                "estimated_download_gb": 1.0,
                "recommended_free_gb": 20.0,
                "available_free_gb": 100.0,
            },
            "dataset_status": status,
            "acquire_datasets": download,
            "prepare_dataset_manifests": prepare,
            "display": lambda value: calls["display"].append(value),
            "print": lambda *args, **kwargs: None,
            "pd": pd,
            "json": json,
        }
        exec(compile(self._cell("acquisition_mutates ="), "acquisition-cell", "exec"), namespace)
        return calls

    def test_dry_run_mutation_cannot_target_fixtures(self) -> None:
        fixture_root = ROOT / ".fixtures"
        for action in ("download", "prepare"):
            with self.subTest(action=action):
                with self.assertRaisesRegex(StudyGateError, r"dry-run \.fixtures"):
                    self._run_acquisition_cell(
                        action=action,
                        profile="dry_run",
                        acquisition_root=fixture_root,
                        explicit_root=False,
                    )

        with self.assertRaisesRegex(StudyGateError, r"dry-run \.fixtures"):
            self._run_acquisition_cell(
                action="prepare",
                profile="dry_run",
                acquisition_root=fixture_root,
                explicit_root=True,
            )

    def test_dry_run_mutation_allows_explicit_non_fixture_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "publisher-data"
            calls = self._run_acquisition_cell(
                action="prepare",
                profile="dry_run",
                acquisition_root=root,
                explicit_root=True,
            )
        self.assertEqual(calls["prepare"], [(root.resolve(), ("visdrone",))])
        self.assertEqual(len(calls["status"]), 2)

    def test_prepare_fails_fast_on_any_blocked_row(self) -> None:
        blocked = pd.DataFrame([
            {
                "dataset": "uavreason",
                "status": "blocked",
                "detail": "missing extracted tree",
            },
            {"dataset": "visdrone", "status": "native_layout", "detail": "ready"},
        ])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(StudyGateError, "uavreason"):
                self._run_acquisition_cell(
                    action="prepare",
                    profile="full",
                    acquisition_root=Path(directory) / "data",
                    explicit_root=False,
                    prepare_result=blocked,
                )

    def _run_environment_cell(self, *, phase: str, device: str) -> list[tuple[str, ...]]:
        checked_scopes: list[tuple[str, ...]] = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = SimpleNamespace(
                enabled_datasets=(),
                prerequisite_errors=lambda stage: [],
            )
            namespace = {
                "PROJECT_ROOT": root,
                "CACHE_ROOT": root / "cache",
                "DEVICE": device,
                "RUN_PROFILE": "full",
                "FULL_PHASE": phase,
                "EXECUTE_FULL_STAGES": True,
                "ctx": SimpleNamespace(run_root=root / "run"),
                "cfg": cfg,
                "environment_report": lambda *args: pd.DataFrame(),
                "environment_fingerprint": lambda report: {},
                "atomic_write_json": lambda path, value: None,
                "assert_environment_ready": lambda report, scopes: checked_scopes.append(tuple(scopes)),
                "display": lambda value: None,
                "print": lambda *args, **kwargs: None,
                "pd": pd,
                "StudyGateError": StudyGateError,
            }
            exec(compile(self._cell("environment_scopes = {\"all\", \"full\"}"), "environment-cell", "exec"), namespace)
        return checked_scopes

    def test_full_prepare_uses_cpu_safe_environment_scopes(self) -> None:
        self.assertEqual(
            self._run_environment_cell(phase="prepare", device="cpu"),
            [("all", "full")],
        )

    def test_later_full_phases_require_a_cuda_device_selection(self) -> None:
        for phase in ("validation", "test", "embedded"):
            with self.subTest(phase=phase):
                with self.assertRaisesRegex(StudyGateError, "requires SEMUAV_DEVICE"):
                    self._run_environment_cell(phase=phase, device="cpu")

        validation_scopes = self._run_environment_cell(phase="validation", device="cuda:0")
        self.assertIn("real_profiles", validation_scopes[0])
        self.assertIn("experiments_2_7", validation_scopes[0])
        embedded_scopes = self._run_environment_cell(phase="embedded", device="cuda:0")
        self.assertIn("real_profiles", embedded_scopes[0])
        self.assertNotIn("experiments_2_7", embedded_scopes[0])

    def test_disk_preflight_row_has_no_mapping_expansion(self) -> None:
        acquisition_cell = self._cell("acquisition_mutates =")
        self.assertNotIn("**acquisition_disk", acquisition_cell)
        self.assertEqual(acquisition_cell.count("disk_preflight("), 1)

    def test_generated_cells_compile_and_document_current_routes(self) -> None:
        for index, cell in enumerate(self.notebook["cells"]):
            if cell.get("cell_type") == "code":
                compile(str(cell.get("source", "")), f"cell-{index}", "exec")
        freeze_cell = self._cell("artifact_paths = {")
        self.assertIn("**cfg.dataset_provenance_paths()", freeze_cell)
        self.assertIn("required_dataset_provenance_artifacts", freeze_cell)
        markdown = "\n".join(
            str(cell.get("source", ""))
            for cell in self.notebook["cells"]
            if cell.get("cell_type") == "markdown"
        )
        self.assertIn("releases/release_20260830/SemUAV-VLM-Full-Study", markdown)
        self.assertIn("already extracted UAVReason/FLAME trees", markdown)
        self.assertIn("Manual UAVid/FLAME ZIPs use", markdown)

    def test_checked_in_notebook_matches_builder_contract(self) -> None:
        checked_in_text = (ROOT / "SemUAV_VLM_Full_Study.ipynb").read_text(encoding="utf-8")
        checked_in = json.loads(checked_in_text)
        self.assertEqual(_notebook_contract(checked_in), _notebook_contract(self.notebook))
        self.assertEqual(_cell_id_errors(checked_in, require_all=False), [])

        # Outside a running notebook this is also the pre-delivery guard: the
        # checked-in artifact must be the exact clean builder output.  During
        # notebook self-tests Jupyter may already have autosaved legitimate live
        # outputs, counters, cell IDs, and runtime metadata to this same file.
        if _requires_exact_delivery_match(
            checked_in,
            live_self_test_marker=os.environ.get("SEMUAV_LIVE_NOTEBOOK_SELFTEST", ""),
        ):
            self.assertEqual(checked_in_text, self.generated_notebook_text)

    def test_builder_output_is_clean_for_delivery(self) -> None:
        self.assertEqual(_delivery_cleanliness_errors(self.notebook), [])
        self.assertEqual(_cell_id_errors(self.notebook, require_all=True), [])
        expected_ids = [f"semuav-{index:02d}" for index in range(len(self.notebook["cells"]))]
        self.assertEqual([cell["id"] for cell in self.notebook["cells"]], expected_ids)

    def test_live_jupyter_state_does_not_change_notebook_contract(self) -> None:
        live = copy.deepcopy(self.notebook)
        live["metadata"]["widgets"] = {"application/vnd.jupyter.widget-state+json": {}}
        live["metadata"]["language_info"]["mimetype"] = "text/x-python"
        live["metadata"]["language_info"]["version"] = "3.10.12"
        cell = next(item for item in live["cells"] if item["cell_type"] == "code")
        source = cell["source"]
        cell["source"] = source.splitlines(keepends=True)
        cell["id"] = "live-cell-id"
        cell["execution_count"] = 4
        cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": "ok\n"}]
        cell["metadata"]["execution"] = {"iopub.status.busy": "2026-08-30T22:00:00Z"}
        cell["metadata"]["trusted"] = True
        self.assertEqual(_notebook_contract(live), _notebook_contract(self.notebook))
        self.assertNotEqual(_delivery_cleanliness_errors(live), [])
        self.assertEqual(_cell_id_errors(live, require_all=True), [])
        self.assertFalse(
            _requires_exact_delivery_match(live, live_self_test_marker=""),
            "autosaved runtime state must activate the legacy live-notebook fallback",
        )
        self.assertTrue(
            _requires_exact_delivery_match(self.notebook, live_self_test_marker=""),
            "a clean artifact outside Jupyter must retain the exact-byte release gate",
        )

        legacy_live = copy.deepcopy(live)
        acceptance_cell = next(
            item for item in legacy_live["cells"]
            if "test_process = subprocess.run" in _normalized_source(item["source"])
        )
        legacy_source = _normalized_source(acceptance_cell["source"])
        self.assertNotIn("SEMUAV_LIVE_NOTEBOOK_SELFTEST", legacy_source)
        acceptance_cell["source"] = legacy_source
        for index, legacy_cell in enumerate(legacy_live["cells"]):
            legacy_cell["id"] = f"legacy-{index:02d}"
        self.assertEqual(_notebook_contract(legacy_live), _notebook_contract(self.notebook))
        self.assertFalse(_requires_exact_delivery_match(legacy_live, live_self_test_marker=""))

    def test_notebook_contract_detects_authored_changes(self) -> None:
        mutations = []

        changed_source = copy.deepcopy(self.notebook)
        changed_source["cells"][0]["source"] += "changed\n"
        mutations.append(("source", changed_source))

        reordered = copy.deepcopy(self.notebook)
        reordered["cells"][0], reordered["cells"][1] = reordered["cells"][1], reordered["cells"][0]
        mutations.append(("order", reordered))

        changed_tags = copy.deepcopy(self.notebook)
        parameter_cell = next(
            item for item in changed_tags["cells"]
            if "parameters" in item.get("metadata", {}).get("tags", [])
        )
        parameter_cell["metadata"]["tags"] = ["not-parameters"]
        mutations.append(("cell metadata", changed_tags))

        changed_kernel = copy.deepcopy(self.notebook)
        changed_kernel["metadata"]["kernelspec"]["name"] = "wrong-kernel"
        mutations.append(("kernel metadata", changed_kernel))

        expected = _notebook_contract(self.notebook)
        for label, mutation in mutations:
            with self.subTest(label=label):
                self.assertNotEqual(_notebook_contract(mutation), expected)

    def test_cell_ids_are_validated_without_binding_live_values(self) -> None:
        live = copy.deepcopy(self.notebook)
        for index, cell in enumerate(live["cells"]):
            cell["id"] = f"jupyter-{index:02d}"
        self.assertEqual(_cell_id_errors(live, require_all=True), [])
        self.assertEqual(_notebook_contract(live), _notebook_contract(self.notebook))

        duplicate = copy.deepcopy(live)
        duplicate["cells"][1]["id"] = duplicate["cells"][0]["id"]
        self.assertTrue(_cell_id_errors(duplicate, require_all=True))

        invalid = copy.deepcopy(live)
        invalid["cells"][0]["id"] = "contains spaces"
        self.assertTrue(_cell_id_errors(invalid, require_all=True))

        missing = copy.deepcopy(live)
        missing["cells"][0].pop("id")
        self.assertEqual(_cell_id_errors(missing, require_all=False), [])
        self.assertTrue(_cell_id_errors(missing, require_all=True))


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Run one phase of the sealed v3r2 study notebook headlessly (survives a closed laptop).

1. Sets the ten operator controls for the phase in the deployed notebook. Only
   the right-hand-side values change, and the edit is checked with the
   deployment's own notebook contract against the sealed archive copy.
2. Executes every cell with nbclient in the registered v3r2 kernel, exactly as
   "Restart Kernel and Run All" would.
3. After every cell, saves the executed notebook and appends a short log to
   ~/semuav_v3r2_outputs/. A failure stops the run and keeps everything executed so far.

Usage (from run_semuav_v3r2.sh, candidate Python):  run_notebook_v3r2.py <phase>
"""
from __future__ import annotations

import ast
import datetime as dt
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

STUDY = Path("/data/Lane Detection Datasets/musta/semuav_vlm")
PACKAGE = "SemUAV-VLM-VisDrone-Paper-Study"
RELEASE = STUDY / "releases" / "release_20261013_visdrone_paper_v3r2" / PACKAGE
ARCHIVE = STUDY / "incoming" / "SemUAV-VLM-VisDrone-Paper-Study-20261013-v3r2.zip"
NOTEBOOK = RELEASE / "SemUAV_VLM_VisDrone_Paper_Study.ipynb"
KERNEL = "semuav-vlm-20261013-v3r2"
OUTPUTS = Path.home() / "semuav_v3r2_outputs"

BASE = {"RUN_NAME": "semuav_visdrone_v3r2", "DEVICE": "cuda:0"}
PHASES = {
    "prepare": dict(PHASE="prepare", EXECUTE_PHASE=True, FREEZE_PROTOCOL=False, AUTHORIZE_TEST_ONCE=False,
                    RESUME=True, RUN_MODEL_SMOKE=True, BUILD_NATIVE_RECEIVER_CACHE=False, BUILD_FEATURE_CACHE=False),
    "validation": dict(PHASE="validation", EXECUTE_PHASE=True, FREEZE_PROTOCOL=False, AUTHORIZE_TEST_ONCE=False,
                       RESUME=True, RUN_MODEL_SMOKE=False, BUILD_NATIVE_RECEIVER_CACHE=True, BUILD_FEATURE_CACHE=True),
    "freeze": dict(PHASE="validation", EXECUTE_PHASE=True, FREEZE_PROTOCOL=True, AUTHORIZE_TEST_ONCE=False,
                   RESUME=True, RUN_MODEL_SMOKE=False, BUILD_NATIVE_RECEIVER_CACHE=False, BUILD_FEATURE_CACHE=False),
    "test": dict(PHASE="test", EXECUTE_PHASE=True, FREEZE_PROTOCOL=False, AUTHORIZE_TEST_ONCE=True,
                 RESUME=False, RUN_MODEL_SMOKE=False, BUILD_NATIVE_RECEIVER_CACHE=False, BUILD_FEATURE_CACHE=False),
    "report": dict(PHASE="report", EXECUTE_PHASE=True, FREEZE_PROTOCOL=False, AUTHORIZE_TEST_ONCE=False,
                   RESUME=True, RUN_MODEL_SMOKE=False, BUILD_NATIVE_RECEIVER_CACHE=False, BUILD_FEATURE_CACHE=False),
}
CONTROLS = ("PHASE", "EXECUTE_PHASE", "FREEZE_PROTOCOL", "AUTHORIZE_TEST_ONCE", "RUN_NAME", "RESUME",
            "DEVICE", "RUN_MODEL_SMOKE", "BUILD_NATIVE_RECEIVER_CACHE", "BUILD_FEATURE_CACHE")


def set_controls(source: str, values: dict) -> str:
    """Replace only the right-hand sides of the ten control assignments."""
    tree = ast.parse(source)
    lines = source.encode("utf-8").splitlines(keepends=True)
    spans = []
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in CONTROLS):
            value = node.value
            start = sum(map(len, lines[:value.lineno - 1])) + value.col_offset
            end = sum(map(len, lines[:value.end_lineno - 1])) + value.end_col_offset
            spans.append((start, end, repr(values[node.targets[0].id])))
    if sorted(name for name in CONTROLS) != sorted(
        n.targets[0].id for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name) and n.targets[0].id in CONTROLS
    ):
        raise SystemExit("STOP: the parameter cell does not contain exactly the ten controls")
    data = source.encode("utf-8")
    for start, end, text in sorted(spans, reverse=True):
        data = data[:start] + text.encode("utf-8") + data[end:]
    return data.decode("utf-8")


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in PHASES:
        raise SystemExit(f"usage: run_notebook_v3r2.py {{{'|'.join(PHASES)}}}")
    phase = sys.argv[1]
    values = {**BASE, **PHASES[phase]}

    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError

    spec = importlib.util.spec_from_file_location("semuav_v3_deployment_runner", RELEASE / "v3_deployment.py")
    deployment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deployment)
    with zipfile.ZipFile(ARCHIVE) as bundle:
        sealed = bundle.read(f"{PACKAGE}/SemUAV_VLM_VisDrone_Paper_Study.ipynb")

    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    parameter_cells = [cell for cell in payload["cells"]
                       if "parameters" in cell.get("metadata", {}).get("tags", [])]
    if len(parameter_cells) != 1:
        raise SystemExit("STOP: expected exactly one parameter cell")
    source = parameter_cells[0]["source"]
    source = "".join(source) if isinstance(source, list) else source
    parameter_cells[0]["source"] = set_controls(source, values)
    candidate = (json.dumps(payload, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    deployment.verify_notebook(candidate, sealed)  # only the ten RHS values may differ
    NOTEBOOK.write_bytes(candidate)
    notebook = nbformat.reads(candidate.decode("utf-8"), as_version=4)
    print("controls set for", phase, json.dumps(values), flush=True)

    OUTPUTS.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    executed_path = OUTPUTS / f"{phase}_{stamp}.ipynb"
    log_path = OUTPUTS / f"{phase}_{stamp}.log"

    def log(text: str) -> None:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(text.rstrip("\n") + "\n")
        print(text, flush=True)

    client = NotebookClient(
        notebook, timeout=None, kernel_name=KERNEL, allow_errors=False, record_timing=True,
        resources={"metadata": {"path": str(RELEASE)}},
    )

    def on_cell_executed(cell=None, cell_index=None, execute_reply=None, **_):
        nbformat.write(client.nb, executed_path)
        now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        log(f"[{now}] cell {cell_index} done")
        for output in (cell or {}).get("outputs", []):
            if output.get("output_type") == "stream":
                text = "".join(output.get("text", ""))
                log("    " + text[-4000:].replace("\n", "\n    "))
            elif output.get("output_type") == "error":
                log("    ERROR " + output.get("ename", "") + ": " + output.get("evalue", ""))

    client.on_cell_executed = on_cell_executed
    log(f"START {phase} -> {executed_path}")
    status = 0
    try:
        client.execute()
        log(f"FINISHED {phase} OK")
    except CellExecutionError as error:
        log(f"FAILED {phase}: {str(error)[-6000:]}")
        status = 1
    except Exception as error:  # kernel death or interruption
        log(f"FAILED {phase}: {type(error).__name__}: {error}")
        status = 1
    finally:
        nbformat.write(client.nb, executed_path)
        log(f"executed notebook saved: {executed_path}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

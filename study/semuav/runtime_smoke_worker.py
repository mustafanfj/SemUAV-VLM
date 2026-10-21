"""Disposable one-model runtime-smoke worker.

This file is executed by absolute path with CPython ``-I -B``.  It inserts
only its authenticated project root before importing the study package.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: runtime_smoke_worker.py INPUT_JSON RESULT_JSON")
    project_root = Path(__file__).resolve().parents[1]
    if project_root.is_symlink() or not project_root.is_dir():
        raise SystemExit("runtime-smoke project root is missing or unsafe")
    sys.path.insert(0, str(project_root))
    from semuav.visdrone_models import execute_runtime_smoke_worker

    return execute_runtime_smoke_worker(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except BaseException:  # parent records the nonzero exit and hashed stderr
        traceback.print_exc()
    finally:
        # Avoid CUDA-backed object finalizers in a context that may have seen a
        # device-side assertion.  The result writer fsyncs and atomically
        # replaces its destination before returning here.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            os._exit(int(exit_code))

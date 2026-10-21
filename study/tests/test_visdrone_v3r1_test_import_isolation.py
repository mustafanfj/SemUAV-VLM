"""The deployment discovery gate must not import an installed `tests` package."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


class DeploymentTestImportIsolationTests(unittest.TestCase):
    def test_affected_tests_pass_with_an_installed_shadow_tests_package(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="semuav-shadow-package-") as directory:
            shadow = Path(directory) / "tests"
            shadow.mkdir()
            (shadow / "__init__.py").write_text(
                "SHADOW_INSTALLED_PACKAGE = True\n", encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = directory + os.pathsep + str(root)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["MPLBACKEND"] = "Agg"
            script = textwrap.dedent("""
                import json
                import unittest
                import tests

                assert tests.SHADOW_INSTALLED_PACKAGE is True
                suite = unittest.TestSuite()
                for pattern in (
                    "test_visdrone_v3_experiment_flow.py",
                    "test_visdrone_v3_receiver_checkpoints.py",
                ):
                    suite.addTests(unittest.TestLoader().discover("tests", pattern=pattern))
                discovered = suite.countTestCases()
                result = unittest.TextTestRunner(verbosity=0).run(suite)
                print(json.dumps({
                    "shadow_package_loaded": True,
                    "discovered": discovered,
                    "run": result.testsRun,
                    "failures": len(result.failures),
                    "errors": len(result.errors),
                }, sort_keys=True))
                raise SystemExit(0 if result.wasSuccessful() else 1)
            """)
            completed = subprocess.run(
                [sys.executable, "-c", script], cwd=root, env=environment,
                capture_output=True, text=True, timeout=90,
            )
            self.assertEqual(
                completed.returncode, 0,
                "Deployment-style discovery failed with an installed shadow package:\n"
                + completed.stdout + completed.stderr,
            )
            report = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertEqual(report, {
                "shadow_package_loaded": True,
                "discovered": 15,
                "run": 15,
                "failures": 0,
                "errors": 0,
            })


if __name__ == "__main__":
    unittest.main()

"""End-to-end tests for the shipped demos.

Runs gat.demo.run_pipeline (the README §17 five-act demo) twice into temp
directories and asserts identical final state digests (determinism), and
runs the geometry demo (python -m gat.demo.geometry) as a subprocess,
asserting a zero exit code and a valid splat PLY artifact.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest

import gat
from gat.demo.__main__ import run_pipeline

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(gat.__file__)))


class TestRunPipelineDeterminism(unittest.TestCase):
    def test_two_runs_identical_digest(self):
        with tempfile.TemporaryDirectory() as tmp1, tempfile.TemporaryDirectory() as tmp2:
            digest_1 = run_pipeline(tmp1, quiet=True)
            digest_2 = run_pipeline(tmp2, quiet=True)
        self.assertIsInstance(digest_1, str)
        self.assertEqual(len(digest_1), 64)  # sha256 hex
        self.assertEqual(digest_1, digest_2)

    def test_pipeline_writes_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_pipeline(tmp, quiet=True)
            self.assertTrue(os.path.exists(os.path.join(tmp, "model_transformed.ifc")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "state.json")))


class TestGeometryDemoSubprocess(unittest.TestCase):
    def test_geometry_demo_runs_and_exports_splats(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                [sys.executable, "-m", "gat.demo.geometry", tmp],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=120,
            )
            self.assertEqual(
                proc.returncode,
                0,
                f"geometry demo failed:\nstdout:\n{proc.stdout[-3000:]}\n"
                f"stderr:\n{proc.stderr[-3000:]}",
            )
            ply = os.path.join(tmp, "building_splats.ply")
            self.assertTrue(os.path.exists(ply))
            with open(ply, "rb") as fh:
                head = fh.read(3)
            self.assertEqual(head, b"ply")


if __name__ == "__main__":
    unittest.main()

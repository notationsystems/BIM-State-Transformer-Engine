"""The ``gat`` command line: exit codes, JSON output, artifacts."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

from gat.cli import main

MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "gat", "demo", "model.ifc",
)


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTests(unittest.TestCase):
    def test_verify_passes_on_demo(self) -> None:
        code, out, _ = run_cli("verify", MODEL)
        self.assertEqual(code, 0)
        self.assertIn("compliance:", out)

    def test_verify_json_is_parseable(self) -> None:
        code, out, _ = run_cli("verify", MODEL, "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertTrue(data["passed"])
        self.assertEqual(data["invariants"]["fail"], 0)

    def test_check_clean_model_exits_zero(self) -> None:
        code, out, _ = run_cli("check", MODEL)
        self.assertEqual(code, 0)
        self.assertIn("clash report:", out)

    def test_check_crossing_duct_exits_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = os.path.join(tmp, "duct.json")
            with open(spec, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "origin": [4.0, 1.8, 2.6],
                        "extents": [3.0, 0.4, 0.4],
                        "angle_deg": 0.0,
                        "position_sigma": 0.02,
                    },
                    fh,
                )
            code, out, _ = run_cli("check", MODEL, "--proposed", spec, "--json")
            self.assertEqual(code, 1)
            data = json.loads(out)
            self.assertGreater(data["worst_p_clash"], 0.999)
            self.assertEqual(data["proposed"][0]["a"], "Wall-Party")

    def test_check_rerouted_duct_exits_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            spec = os.path.join(tmp, "duct.json")
            with open(spec, "w", encoding="utf-8") as fh:
                json.dump(
                    {"origin": [4.0, 1.8, 3.55], "extents": [3.0, 0.4, 0.4]}, fh
                )
            code, out, _ = run_cli("check", MODEL, "--proposed", spec, "--json")
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["proposed"], [])

    def test_inspect_summary_and_variable(self) -> None:
        code, out, _ = run_cli("inspect", MODEL, "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["n_raw"], 24)

        code, out, _ = run_cli("inspect", MODEL, "--var", "Level 1.TotalWallCost")
        self.assertEqual(code, 0)
        self.assertIn("8503.2", out)
        self.assertIn("Wall-South.Width", out)  # pretty-printed sensitivity

    def test_inspect_bad_var_syntax_exits_two(self) -> None:
        code, _, err = run_cli("inspect", MODEL, "--var", "NoDotHere")
        self.assertEqual(code, 2)
        self.assertIn("Entity-Name.Quantity", err)

    def test_splats_with_variations_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code, out, _ = run_cli(
                "splats", MODEL, tmp, "--variations", "3", "--seed", "5"
            )
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(os.path.join(tmp, "building.ply")))
            with open(os.path.join(tmp, "building.ply"), "rb") as fh:
                self.assertTrue(fh.read(3) == b"ply")
            manifest_path = os.path.join(tmp, "manifest.json")
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
            self.assertEqual(manifest["n"], 3)
            for sample in manifest["samples"]:
                self.assertTrue(os.path.exists(os.path.join(tmp, sample["file"])))

    def test_variations_are_seed_deterministic(self) -> None:
        import filecmp

        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            run_cli("splats", MODEL, a, "--variations", "2", "--seed", "9")
            run_cli("splats", MODEL, b, "--variations", "2", "--seed", "9")
            for name in ("variation_000.ply", "variation_001.ply", "manifest.json"):
                self.assertTrue(
                    filecmp.cmp(
                        os.path.join(a, name), os.path.join(b, name), shallow=False
                    ),
                    name,
                )

    def test_sample_json(self) -> None:
        code, out, _ = run_cli("sample", MODEL, "--n", "100", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["n"], 100)
        self.assertGreaterEqual(data["pass_rate"], 0.9)

    def test_missing_model_exits_two(self) -> None:
        code, _, err = run_cli("verify", "/nonexistent/model.ifc")
        self.assertEqual(code, 2)
        self.assertIn("gat:", err)

    def test_bad_usage_exits_two(self) -> None:
        code, _, _ = run_cli("frobnicate")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()

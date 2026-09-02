"""The offline 3D viewer: deterministic payload, self-contained instrument."""

from __future__ import annotations

import math
import os
import tempfile
import unittest

import gat.demo
from gat.cli import main as cli_main
from gat.engine.verify import run_invariants
from gat.geometry.viewer import (
    VIEWER_SCENE_FORMAT,
    export_viewer_html,
    viewer_payload,
)
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class ViewerPayloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.world = GatSession.load_ifc(MODEL).world
        cls.payload = viewer_payload(cls.world, n=3, seed=7, model_name="model.ifc")

    def test_payload_carries_nominal_plus_n_samples(self) -> None:
        self.assertEqual(self.payload["format"], VIEWER_SCENE_FORMAT)
        samples = self.payload["samples"]
        self.assertEqual(len(samples), 4)
        self.assertEqual(samples[0]["label"], "nominal")
        self.assertEqual(samples[0]["world_digest"], self.world.digest())
        self.assertEqual(samples[0]["passed"], run_invariants(self.world).passed)

    def test_arrays_are_consistent_and_finite(self) -> None:
        classes = self.payload["classes"]
        for element in self.payload["elements"]:
            self.assertLess(element["class"], len(classes))
        for sample in self.payload["samples"]:
            count = len(sample["element"])
            self.assertEqual(len(sample["centers"]), 3 * count)
            self.assertEqual(len(sample["axes"]), 9 * count)
            self.assertTrue(all(math.isfinite(v) for v in sample["centers"]))
            self.assertTrue(all(math.isfinite(v) for v in sample["axes"]))
            for index in sample["element"]:
                self.assertLess(index, len(self.payload["elements"]))

    def test_payload_is_deterministic(self) -> None:
        again = viewer_payload(self.world, n=3, seed=7, model_name="model.ifc")
        self.assertEqual(self.payload, again)

    def test_different_seed_changes_samples_not_nominal(self) -> None:
        other = viewer_payload(self.world, n=3, seed=8, model_name="model.ifc")
        self.assertEqual(
            self.payload["samples"][0]["centers"], other["samples"][0]["centers"]
        )
        self.assertNotEqual(
            self.payload["samples"][1]["centers"], other["samples"][1]["centers"]
        )


class ViewerHtmlTests(unittest.TestCase):
    def test_html_is_self_contained_and_offline(self) -> None:
        world = GatSession.load_ifc(MODEL).world
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "viewer.html")
            count = export_viewer_html(world, path, n=2, model_name="model.ifc")
            self.assertEqual(count, 3)
            with open(path, encoding="utf-8") as handle:
                html = handle.read()
        self.assertTrue(html.startswith("<!doctype html>"))
        self.assertNotIn("__GAT_SCENE_JSON__", html)
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn(VIEWER_SCENE_FORMAT, html)
        self.assertIn("Read-only: no BIM state was changed.", html)

    def test_cli_view_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "viewer.html")
            self.assertEqual(
                cli_main(["view", MODEL, "-o", path, "--variations", "2"]), 0
            )
            self.assertTrue(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()

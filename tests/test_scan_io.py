"""Tests for the external PLY scan-artifact boundary.

The loader supports both common standard PLY encodings and only admits
unambiguous x/y/z vertex data.  A final integration test loads the binary
3DGS PLY emitted by GAT itself, proving that the same adapter can consume
standard external Gaussian-splat mesh/point artifacts without a producer
runtime dependency.
"""

from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

import gat.demo
from gat.errors import ScanArtifactError
from gat.geometry.scan_io import load_ply_points
from gat.geometry.splat_io import export_splat_ply
from gat.geometry.stateio import derive_scene
from gat.session import GatSession


MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")


class PlyScanIoTests(unittest.TestCase):
    def test_loads_ascii_vertices_in_declared_property_order(self) -> None:
        source = (
            "ply\n"
            "format ascii 1.0\n"
            "comment mesh faces follow vertices and are ignored\n"
            "element vertex 2\n"
            "property float z\n"
            "property uchar red\n"
            "property double x\n"
            "property float y\n"
            "element face 1\n"
            "property list uchar int vertex_indices\n"
            "end_header\n"
            "3.0 255 1.5 -2.0\n"
            "6.0 10 -4.0 8.5\n"
            "3 0 1 0\n"
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh_ascii.ply"
            path.write_bytes(source)
            points = load_ply_points(path)

        np.testing.assert_allclose(points, [[1.5, -2.0, 3.0], [-4.0, 8.5, 6.0]])
        self.assertFalse(points.flags.writeable)

    def test_loads_binary_little_endian_vertices(self) -> None:
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 2\n"
            "property float y\n"
            "property double x\n"
            "property uchar confidence\n"
            "property float z\n"
            "end_header\n"
        ).encode("ascii")
        body = struct.pack("<fdBf", -2.0, 1.5, 255, 3.0) + struct.pack(
            "<fdBf", 8.5, -4.0, 10, 6.0
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mesh_binary.ply"
            path.write_bytes(header + body)
            points = load_ply_points(path)

        np.testing.assert_allclose(points, [[1.5, -2.0, 3.0], [-4.0, 8.5, 6.0]])

    def test_rejects_missing_coordinate_property(self) -> None:
        source = (
            "ply\nformat ascii 1.0\nelement vertex 1\n"
            "property float x\nproperty float y\nend_header\n0 0\n"
        ).encode("ascii")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.ply"
            path.write_bytes(source)
            with self.assertRaises(ScanArtifactError):
                load_ply_points(path)

    def test_reads_gat_standard_binary_splat_artifact(self) -> None:
        # GAT's 3DGS exporter has x/y/z plus fourteen non-coordinate scalar
        # vertex fields.  A scan adapter must ignore those fields without
        # depending on the original splat exporter or viewer.
        scene = derive_scene(GatSession.load_ifc(MODEL).world)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "building_splats.ply"
            export_splat_ply(scene.cloud, str(path))
            points = load_ply_points(path)

        self.assertEqual(points.shape, scene.cloud.means.shape)
        np.testing.assert_allclose(points, scene.cloud.means, rtol=0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

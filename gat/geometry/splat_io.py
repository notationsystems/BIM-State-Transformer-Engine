"""Export the Gaussian scene as a 3D Gaussian Splatting PLY file.

Writes the binary little-endian PLY layout used by the reference 3DGS
implementation (graphdeco-inria/gaussian-splatting) and its derivatives:

    x, y, z, nx, ny, nz, f_dc_0..2, opacity, scale_0..2, rot_0..3

with log-scales, pre-sigmoid opacity, and SH DC color coefficients — so a
Gaussianized building drops straight into any standard splat viewer.
Colors encode semantic class (walls grey, spaces sky-blue, doors amber),
which makes the export a legible inspection artifact, not just a blob.
"""

from __future__ import annotations

import math
import struct

import numpy as np

from gat.geometry.primitives import CLASS_CHANNEL, GaussianCloud

#: RGB in [0,1] per semantic class channel.
CLASS_COLORS: dict[int, tuple[float, float, float]] = {
    CLASS_CHANNEL["IfcWall"]: (0.72, 0.70, 0.66),
    CLASS_CHANNEL["IfcSpace"]: (0.45, 0.70, 0.95),
    CLASS_CHANNEL["IfcDoor"]: (0.90, 0.62, 0.20),
    CLASS_CHANNEL["IfcOpeningElement"]: (0.85, 0.30, 0.30),
    CLASS_CHANNEL["IfcBuildingStorey"]: (0.55, 0.55, 0.55),
}

_SH_C0 = 0.28209479177387814  # Y_0^0; color = 0.5 + C0 * f_dc


def _inverse_sigmoid(x: float) -> float:
    x = min(max(x, 1e-6), 1.0 - 1e-6)  # domain guard: opacity 0/1 saturates
    return math.log(x / (1.0 - x))


def export_splat_ply(
    cloud: GaussianCloud,
    path: str,
    opacity: float = 0.85,
    space_opacity: float = 0.15,
) -> int:
    """Write the cloud as a 3DGS-format PLY; returns the primitive count."""
    n = len(cloud)
    scales, quats = cloud.to_scaling_rotation()

    class_idx = np.argmax(cloud.features[:, : len(CLASS_CHANNEL)], axis=1)
    colors = np.array(
        [CLASS_COLORS.get(int(c), (0.8, 0.8, 0.8)) for c in class_idx]
    )
    f_dc = (colors - 0.5) / _SH_C0

    space_channel = CLASS_CHANNEL["IfcSpace"]
    opacities = np.where(
        class_idx == space_channel,
        _inverse_sigmoid(space_opacity),
        _inverse_sigmoid(opacity),
    )

    fields = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {name}\n" for name in fields)
        + "end_header\n"
    )

    data = np.zeros((n, len(fields)), dtype=np.float32)
    data[:, 0:3] = cloud.means
    data[:, 6:9] = f_dc
    data[:, 9] = opacities
    data[:, 10:13] = np.log(np.clip(scales, 1e-9, None))
    data[:, 13:17] = quats

    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(data.astype("<f4").tobytes())
    return n

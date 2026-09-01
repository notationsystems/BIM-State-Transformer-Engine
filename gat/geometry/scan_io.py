"""Adapter boundary for external PLY scan and reconstructed-mesh artifacts.

The geometry layer treats external geometry as *evidence*, not as a second
canonical BIM model.  This module reads only the ``x, y, z`` vertex positions
from a standard PLY artifact and returns an immutable point cloud suitable for
the existing :class:`gat.geometry.registration.ScanRegistrar`.

It deliberately has no dependency on a reconstruction runtime.  In
particular, Geometry-Grounded-Gaussian-Splatting writes ``recon_post.ply``
after extracting and post-processing a triangle mesh; its vertices can be
passed through this adapter without importing its CUDA/PyTorch/Open3D stack.
Normals, colors, faces, and any extra vertex properties are evidence metadata
owned by the producer and are ignored by GAT's current point-to-Gaussian
registration likelihood.

Supported PLY layouts are ASCII 1.0 and binary little-endian 1.0 with scalar
vertex properties.  PLYs whose vertices are not the first data element, or
which put a list property on a vertex, fail loudly instead of being partially
misread.
"""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import numpy as np

from gat.errors import ScanArtifactError


_SCALAR_DTYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "<i2",
    "int16": "<i2",
    "ushort": "<u2",
    "uint16": "<u2",
    "int": "<i4",
    "int32": "<i4",
    "uint": "<u4",
    "uint32": "<u4",
    "float": "<f4",
    "float32": "<f4",
    "double": "<f8",
    "float64": "<f8",
}


def load_ply_points(path: str | Path) -> np.ndarray:
    """Load immutable ``(n, 3)`` float64 points from a PLY vertex element.

    The point order is preserved exactly.  A triangle mesh is therefore a
    valid scan artifact: its face element is left unread once all vertices
    have been consumed.
    """
    path = Path(path)
    try:
        with path.open("rb") as fh:
            fmt, count, props = _read_header(fh, path)
            if fmt == "ascii":
                points = _read_ascii_vertices(fh, count, props, path)
            else:
                points = _read_binary_vertices(fh, count, props, path)
    except OSError as exc:
        raise ScanArtifactError(f"could not read PLY artifact {path}: {exc}") from exc

    if not np.isfinite(points).all():
        raise ScanArtifactError(f"PLY artifact {path} contains non-finite vertex coordinates")
    points = np.ascontiguousarray(points, dtype=np.float64)
    points.setflags(write=False)
    return points


def _read_header(
    fh: BinaryIO, path: Path
) -> tuple[str, int, tuple[tuple[str, str], ...]]:
    first = _header_line(fh, path)
    if first != "ply":
        raise ScanArtifactError(f"{path}: missing PLY magic header")

    fmt: str | None = None
    vertex_count: int | None = None
    vertex_props: list[tuple[str, str]] = []
    current_element: str | None = None
    saw_data_element = False

    while True:
        line = _header_line(fh, path)
        if line == "end_header":
            break
        parts = line.split()
        if not parts or parts[0] in {"comment", "obj_info"}:
            continue
        if parts[0] == "format":
            if len(parts) != 3 or parts[2] != "1.0":
                raise ScanArtifactError(f"{path}: unsupported PLY format declaration {line!r}")
            if parts[1] == "ascii":
                fmt = "ascii"
            elif parts[1] == "binary_little_endian":
                fmt = "binary_little_endian"
            else:
                raise ScanArtifactError(f"{path}: unsupported PLY format {parts[1]!r}")
        elif parts[0] == "element":
            if len(parts) != 3:
                raise ScanArtifactError(f"{path}: malformed element declaration {line!r}")
            try:
                element_count = int(parts[2])
            except ValueError as exc:
                raise ScanArtifactError(f"{path}: invalid element count in {line!r}") from exc
            if element_count < 0:
                raise ScanArtifactError(f"{path}: negative element count in {line!r}")
            current_element = parts[1]
            if current_element == "vertex":
                if saw_data_element:
                    raise ScanArtifactError(
                        f"{path}: vertex element must be first for streaming point import"
                    )
                vertex_count = element_count
            saw_data_element = True
        elif parts[0] == "property" and current_element == "vertex":
            if len(parts) == 3:
                ptype, name = parts[1:]
                if ptype not in _SCALAR_DTYPES:
                    raise ScanArtifactError(f"{path}: unsupported vertex property type {ptype!r}")
                if name in {existing for existing, _ in vertex_props}:
                    raise ScanArtifactError(f"{path}: duplicate vertex property {name!r}")
                vertex_props.append((name, ptype))
            elif len(parts) >= 2 and parts[1] == "list":
                raise ScanArtifactError(f"{path}: list-valued vertex properties are unsupported")
            else:
                raise ScanArtifactError(f"{path}: malformed vertex property {line!r}")

    if fmt is None:
        raise ScanArtifactError(f"{path}: PLY header has no format declaration")
    if vertex_count is None:
        raise ScanArtifactError(f"{path}: PLY header has no vertex element")
    names = {name for name, _ in vertex_props}
    missing = {"x", "y", "z"} - names
    if missing:
        raise ScanArtifactError(f"{path}: vertex element lacks coordinate properties {sorted(missing)}")
    return fmt, vertex_count, tuple(vertex_props)


def _header_line(fh: BinaryIO, path: Path) -> str:
    raw = fh.readline()
    if not raw:
        raise ScanArtifactError(f"{path}: unexpected EOF in PLY header")
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise ScanArtifactError(f"{path}: PLY header is not ASCII") from exc


def _read_ascii_vertices(
    fh: BinaryIO,
    count: int,
    props: tuple[tuple[str, str], ...],
    path: Path,
) -> np.ndarray:
    coord_columns = tuple(next(i for i, p in enumerate(props) if p[0] == axis) for axis in "xyz")
    out = np.empty((count, 3), dtype=np.float64)
    for row in range(count):
        raw = fh.readline()
        if not raw:
            raise ScanArtifactError(f"{path}: EOF after {row} of {count} vertex rows")
        try:
            fields = raw.decode("ascii").split()
            if len(fields) != len(props):
                raise ValueError(f"expected {len(props)} fields, found {len(fields)}")
            out[row] = [float(fields[i]) for i in coord_columns]
        except (UnicodeDecodeError, ValueError) as exc:
            raise ScanArtifactError(f"{path}: invalid ASCII vertex row {row}: {exc}") from exc
    return out


def _read_binary_vertices(
    fh: BinaryIO,
    count: int,
    props: tuple[tuple[str, str], ...],
    path: Path,
) -> np.ndarray:
    dtype = np.dtype([(name, _SCALAR_DTYPES[ptype]) for name, ptype in props])
    values = np.fromfile(fh, dtype=dtype, count=count)
    if len(values) != count:
        raise ScanArtifactError(f"{path}: EOF after {len(values)} of {count} binary vertices")
    return np.column_stack((values["x"], values["y"], values["z"])).astype(np.float64)

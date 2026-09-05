"""Explicit right-handed frames and first-order joint pose propagation.

This is a mathematical contract, not an IFC lowering extension or a source
of calibrated evidence. All rigid-transform translations are in metres.
Frame units describe point coordinates at the API boundary only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import hashlib
import json
from types import MappingProxyType
from typing import Iterable

import numpy as np

FRAME_CONTRACT = "gat-coordinate-frame-v1"
POSE_CONVENTION = "right-local-tx-ty-tz-rx-ry-rz"


def _array(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    array.setflags(write=False)
    return array


def _covariance(value, dimension: int) -> np.ndarray:
    matrix = _array(value, (dimension, dimension), "covariance")
    scale = max(float(np.linalg.norm(matrix, ord=2)), np.finfo(float).tiny)
    tolerance = 64 * np.finfo(float).eps * dimension * scale
    if not np.allclose(matrix, matrix.T, atol=tolerance, rtol=0):
        raise ValueError("covariance must be symmetric")
    matrix = (matrix + matrix.T) / 2
    if float(np.linalg.eigvalsh(matrix).min()) < -tolerance:
        raise ValueError("covariance must be positive semidefinite")
    matrix.setflags(write=False)
    return matrix


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


class LengthUnit(StrEnum):
    METRE = "m"
    MILLIMETRE = "mm"

    @property
    def metres(self) -> float:
        return 1.0 if self is LengthUnit.METRE else 0.001


@dataclass(frozen=True, eq=False)
class RigidTransform:
    """Parent-from-child: p_parent_m = rotation @ p_child_m + translation_m.

    Rotation columns are child axes expressed in the parent frame. Only
    proper orthonormal rotations are accepted; scale/shear/reflection fail.
    """

    rotation: np.ndarray
    translation_m: np.ndarray

    def __post_init__(self):
        rotation = _array(self.rotation, (3, 3), "rotation")
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10, rtol=0)
        ):
            raise ValueError("rotation must be proper orthonormal (right-handed)")
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation_m", _array(self.translation_m, (3,), "translation_m"))

    @classmethod
    def identity(cls) -> "RigidTransform":
        return cls(np.eye(3), np.zeros(3))

    def compose(self, child: "RigidTransform") -> "RigidTransform":
        """Return self @ child; composition order is not commutative."""
        return RigidTransform(
            self.rotation @ child.rotation,
            self.rotation @ child.translation_m + self.translation_m,
        )

    def inverse(self) -> "RigidTransform":
        return RigidTransform(self.rotation.T, -self.rotation.T @ self.translation_m)

    def point(self, point_m) -> np.ndarray:
        return self.rotation @ _array(point_m, (3,), "point_m") + self.translation_m

    def propagate_points(self, points_m, joint_covariance) -> tuple[np.ndarray, np.ndarray]:
        """Propagate a joint point/pose Gaussian through one shared frame.

        For N points, covariance order is [p1_xyz, ..., pN_xyz, tx,ty,tz,rx,ry,rz].
        Point and translational errors use metres; angular errors use radians.
        Pose errors are a right-local SE(3) tangent: T_actual = T @ Exp(delta).
        All point/pose and cross-point correlations must be supplied in the
        (3N+6)^2 matrix. Zero cross-blocks explicitly assume independence.
        Returns transformed nominal points and their full 3N joint covariance.
        This first-order result is not a calibration or a large-angle model.
        """
        points = np.asarray(points_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            raise ValueError("points_m must be a nonempty Nx3 array")
        points = _array(points, points.shape, "points_m")
        count = len(points)
        covariance = _covariance(joint_covariance, 3 * count + 6)
        point_jacobian = np.kron(np.eye(count), self.rotation)
        pose_jacobian = np.vstack([
            self.rotation @ np.hstack((np.eye(3), -_skew(point)))
            for point in points
        ])
        jacobian = np.hstack((point_jacobian, pose_jacobian))
        transformed = points @ self.rotation.T + self.translation_m
        propagated = jacobian @ covariance @ jacobian.T
        return transformed, (propagated + propagated.T) / 2


@dataclass(frozen=True)
class CoordinateFrame:
    frame_id: str
    parent_id: str | None
    to_parent: RigidTransform
    unit: LengthUnit = LengthUnit.METRE
    handedness: str = "right"

    def __post_init__(self):
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ValueError("frame_id must be nonempty")
        if self.parent_id is not None and (
            not isinstance(self.parent_id, str) or not self.parent_id.strip()
            or self.parent_id == self.frame_id
        ):
            raise ValueError("parent_id must identify a different frame")
        if not isinstance(self.to_parent, RigidTransform):
            raise ValueError("to_parent must be a RigidTransform")
        if self.handedness != "right":
            raise ValueError("only right-handed frames are supported")
        object.__setattr__(self, "unit", LengthUnit(self.unit))
        if self.parent_id is None and (
            not np.array_equal(self.to_parent.rotation, np.eye(3))
            or not np.array_equal(self.to_parent.translation_m, np.zeros(3))
        ):
            raise ValueError("the root frame must have an identity transform")


class FrameGraph:
    """A validated, single-root tree; identifiers are checked before composing."""

    def __init__(self, frames: Iterable[CoordinateFrame]):
        entries = tuple(frames)
        if not entries or any(not isinstance(frame, CoordinateFrame) for frame in entries):
            raise ValueError("frames must contain CoordinateFrame records")
        records = {frame.frame_id: frame for frame in entries}
        if len(records) != len(entries):
            raise ValueError("duplicate frame identity")
        if sum(frame.parent_id is None for frame in entries) != 1:
            raise ValueError("frames must have exactly one root")
        if any(frame.parent_id is not None and frame.parent_id not in records for frame in entries):
            raise ValueError("unknown parent frame")
        self.frames = MappingProxyType(records)
        for frame in entries:
            self.to_root(frame.frame_id)  # reject cycles at construction

    def _frame(self, frame_id: str) -> CoordinateFrame:
        try:
            return self.frames[frame_id]
        except KeyError as exc:
            raise ValueError(f"unknown frame {frame_id!r}") from exc

    def to_root(self, frame_id: str) -> RigidTransform:
        frame = self._frame(frame_id)
        visited = set()
        result = RigidTransform.identity()
        while frame.parent_id is not None:
            if frame.frame_id in visited:
                raise ValueError("coordinate frame cycle")
            visited.add(frame.frame_id)
            result = frame.to_parent.compose(result)
            frame = self._frame(frame.parent_id)
        return result

    def between(self, source_id: str, target_id: str) -> RigidTransform:
        """Return target-from-source, operating on metre coordinates."""
        return self.to_root(target_id).inverse().compose(self.to_root(source_id))

    def point(self, point, source_id: str, target_id: str) -> np.ndarray:
        source, target = self._frame(source_id), self._frame(target_id)
        point_m = _array(point, (3,), "point") * source.unit.metres
        return self.between(source_id, target_id).point(point_m) / target.unit.metres

    def covariance(self, covariance, source_id: str, target_id: str) -> np.ndarray:
        """Change coordinates of a point covariance with exact frame transforms.

        This does not add frame uncertainty; use propagate_points with the
        explicit joint pose covariance when the frame itself is uncertain.
        """
        source, target = self._frame(source_id), self._frame(target_id)
        linear = self.between(source_id, target_id).rotation * (
            source.unit.metres / target.unit.metres
        )
        result = linear @ _covariance(covariance, 3) @ linear.T
        return (result + result.T) / 2

    def representation_digest(self) -> str:
        """Hash this frame representation; not physical or evidence identity."""
        record = {
            "contract": FRAME_CONTRACT,
            "frames": [
                {
                    "id": frame.frame_id,
                    "parent": frame.parent_id,
                    "unit": frame.unit.value,
                    "handedness": frame.handedness,
                    "rotation": frame.to_parent.rotation.tolist(),
                    "translation_m": frame.to_parent.translation_m.tolist(),
                }
                for _, frame in sorted(self.frames.items())
            ],
        }
        encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

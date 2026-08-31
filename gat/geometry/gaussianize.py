"""Gaussianization: oriented boxes → moment-matched Gaussian primitive sets.

Every architectural element enters the geometric layer as an oriented box
(placement origin, yaw about +Z, extents along local axes).  The box is
split into a regular grid of sub-boxes; each sub-box becomes one Gaussian
primitive that is the *exact second-moment match* of the uniform measure
on that sub-box:

    Var(Uniform[-h, h]) = h^2 / 3      (per axis, half-extent h)

so a sub-box with half-extents ``h_i / n_i`` maps to
``Sigma_k = R diag((h_i/n_i)^2 / 3) R^T`` and weight = its volume.
Because the sub-boxes partition the box, the mixture's total mean and
covariance are exactly the box moments (law of total covariance) — a
provable invariant the tests pin at 1e-12, instead of an arbitrary
"sigma = extent/k" heuristic.

The map ships its analytic Jacobians with respect to the extents — the
chain-rule bridge from primitive means/covariances back into the canonical
state parameters (via the extents' own raw-space Jacobian rows).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class OrientedBox:
    """An axis-oriented box: corner origin, yaw, full extents along local axes."""

    origin: tuple[float, float, float]
    angle: float
    extents: tuple[float, float, float]  # full extents (E_x, E_y, E_z), local frame

    @property
    def volume(self) -> float:
        return self.extents[0] * self.extents[1] * self.extents[2]

    def center(self) -> np.ndarray:
        R = rot_z(self.angle)
        local = 0.5 * np.asarray(self.extents)
        return np.asarray(self.origin) + R @ local


@dataclass(frozen=True)
class Tiling:
    """Grid resolution of one Gaussianized box."""

    n: tuple[int, int, int]

    @property
    def count(self) -> int:
        return self.n[0] * self.n[1] * self.n[2]


def tiling_for(box: OrientedBox, spacing: float) -> Tiling:
    n = tuple(max(1, math.ceil(box.extents[i] / spacing)) for i in range(3))
    return Tiling(n)  # type: ignore[arg-type]


def gaussianize_box(
    box: OrientedBox, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split the box into a grid of moment-matched Gaussians.

    Returns ``(means (K,3), covs (K,3,3), weights (K,), fractions (K,3))``
    where ``fractions[k, i] = (2 a_i + 1) / (2 n_i)`` is the sub-box
    center's fractional position along local axis ``i`` — the quantity the
    extent Jacobian needs.
    """
    tiling = tiling_for(box, spacing)
    nx, ny, nz = tiling.n
    R = rot_z(box.angle)
    E = np.asarray(box.extents, dtype=np.float64)

    fracs = []
    for ax in range(nx):
        for ay in range(ny):
            for az in range(nz):
                fracs.append(
                    (
                        (2 * ax + 1) / (2 * nx),
                        (2 * ay + 1) / (2 * ny),
                        (2 * az + 1) / (2 * nz),
                    )
                )
    fractions = np.asarray(fracs, dtype=np.float64)  # (K, 3)

    local_centers = fractions * E  # (K, 3)
    means = np.asarray(box.origin) + local_centers @ R.T

    n = np.array([nx, ny, nz], dtype=np.float64)
    cell_half = 0.5 * E / n
    cell_cov_local = np.diag(cell_half**2 / 3.0)
    cov = R @ cell_cov_local @ R.T
    covs = np.broadcast_to(cov, (fractions.shape[0], 3, 3)).copy()

    cell_volume = box.volume / tiling.count
    weights = np.full(fractions.shape[0], cell_volume, dtype=np.float64)
    return means, covs, weights, fractions


def mean_jacobian_wrt_extents(box: OrientedBox, fractions: np.ndarray) -> np.ndarray:
    """d mu_k / d E — (K, 3, 3): row axis is the world coordinate, column
    the extent.  mu_k = origin + R (f_k ∘ E), so d mu_k / d E_i = R[:, i] f_ki.
    """
    R = rot_z(box.angle)
    K = fractions.shape[0]
    J = np.zeros((K, 3, 3), dtype=np.float64)
    for i in range(3):
        J[:, :, i] = np.outer(fractions[:, i], R[:, i])
    return J


def cov_jacobian_wrt_extents(box: OrientedBox, spacing: float) -> np.ndarray:
    """d Sigma_k / d E — (3, 3, 3): last axis is the extent index.

    Sigma_k = R diag(E_i^2 / (12 n_i^2)) R^T, so
    d Sigma / d E_i = (E_i / (6 n_i^2)) R e_i e_i^T R^T.
    Identical for every primitive of the box (uniform grid).
    """
    tiling = tiling_for(box, spacing)
    R = rot_z(box.angle)
    E = np.asarray(box.extents, dtype=np.float64)
    n = np.asarray(tiling.n, dtype=np.float64)
    out = np.zeros((3, 3, 3), dtype=np.float64)
    for i in range(3):
        col = R[:, i]
        out[:, :, i] = (E[i] / (6.0 * n[i] ** 2)) * np.outer(col, col)
    return out


def mean_jacobian_wrt_yaw(box: OrientedBox, fractions: np.ndarray) -> np.ndarray:
    """d mu_k / d angle — (K, 3).  dR/dtheta rotates the local offset."""
    c, s = math.cos(box.angle), math.sin(box.angle)
    dR = np.array([[-s, -c, 0.0], [c, -s, 0.0], [0.0, 0.0, 0.0]])
    local_centers = fractions * np.asarray(box.extents)
    return local_centers @ dR.T

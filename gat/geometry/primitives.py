"""The Gaussian primitive cloud — GAT's continuous geometric representation.

A :class:`GaussianCloud` is a struct-of-arrays container of 3D Gaussian
primitives: means, covariances, weights (absolute volumes, so the cloud is
a *measure*, not a pdf), semantic feature vectors, and provenance links to
the elements they were derived from.  It replaces boundary representations
with a continuous, differentiable field — the representational shift the
project proposes — while remaining a *derived* view of the canonical
architectural state, never a second source of truth.

Covariances follow the 3D Gaussian Splatting factorization convention
``Sigma = R S S^T R^T`` (rotation + per-axis scales); ``to_scaling_rotation``
recovers that factorization for interchange with splatting toolchains
(graphdeco-inria/gaussian-splatting and derivatives).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gat.errors import NumericalError


#: Fixed feature schema: column -> meaning.  Class one-hots are identity
#: channels and are never mutated by propagation (README §14 principle 3).
FEATURE_NAMES: tuple[str, ...] = (
    "is_wall",
    "is_space",
    "is_door",
    "is_opening",
    "is_storey",
    "log_volume",
    "u_value",        # thermal transmittance [W/m2K]
    "load_bearing",   # 0/1
    "external",       # 0/1
    "storey_index",
)

N_FEATURES = len(FEATURE_NAMES)

#: Channels attention is allowed to rewrite (payload); the rest are frozen.
PAYLOAD_CHANNELS: tuple[int, ...] = (
    FEATURE_NAMES.index("u_value"),
    FEATURE_NAMES.index("load_bearing"),
    FEATURE_NAMES.index("external"),
)

CLASS_CHANNEL: dict[str, int] = {
    "IfcWall": 0,
    "IfcSpace": 1,
    "IfcDoor": 2,
    "IfcOpeningElement": 3,
    "IfcBuildingStorey": 4,
}


@dataclass
class GaussianCloud:
    """Struct-of-arrays batch of N Gaussian primitives."""

    means: np.ndarray          # (N, 3) float64
    covs: np.ndarray           # (N, 3, 3) float64, symmetric positive definite
    weights: np.ndarray        # (N,) float64, absolute volumes
    features: np.ndarray       # (N, F) float64
    element_index: np.ndarray  # (N,) intp — row into the scene's element table
    version: str = ""          # digest of the world this cloud was derived from
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.means.shape[0]
        if self.covs.shape != (n, 3, 3) or self.weights.shape != (n,):
            raise ValueError("inconsistent cloud array shapes")
        if self.features.shape[0] != n or self.element_index.shape != (n,):
            raise ValueError("inconsistent cloud array shapes")

    def __len__(self) -> int:
        return int(self.means.shape[0])

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        if not np.isfinite(self.means).all() or not np.isfinite(self.covs).all():
            raise NumericalError("cloud contains non-finite entries")
        if (self.weights <= 0).any():
            raise NumericalError("cloud weights must be strictly positive")
        # SPD check via Cholesky, batched.
        try:
            np.linalg.cholesky(self.covs)
        except np.linalg.LinAlgError as exc:
            raise NumericalError(f"non-SPD primitive covariance: {exc}") from exc

    # -- selection ---------------------------------------------------------

    def select(self, mask: np.ndarray) -> "GaussianCloud":
        return GaussianCloud(
            self.means[mask],
            self.covs[mask],
            self.weights[mask],
            self.features[mask],
            self.element_index[mask],
            self.version,
        )

    def of_element(self, element_row: int) -> "GaussianCloud":
        return self.select(self.element_index == element_row)

    # -- moments -----------------------------------------------------------

    def total_weight(self) -> float:
        return float(self.weights.sum())

    def mixture_moments(self) -> tuple[np.ndarray, np.ndarray]:
        """Weight-normalized mixture mean and covariance (moment match)."""
        w = self.weights / self.weights.sum()
        mean = w @ self.means
        centered = self.means - mean
        cov = np.einsum("k,kij->ij", w, self.covs) + np.einsum(
            "k,ki,kj->ij", w, centered, centered
        )
        return mean, cov

    # -- 3DGS interchange --------------------------------------------------

    def to_scaling_rotation(self) -> tuple[np.ndarray, np.ndarray]:
        """Factor each covariance as ``Sigma = R S S^T R^T``.

        Returns ``(scales (N,3), quaternions (N,4) in (w,x,y,z))`` — the
        parameterization used by 3D Gaussian Splatting codebases.  Uses a
        batched symmetric eigendecomposition (interchange path only; never
        called by the execution engine).
        """
        eigvals, eigvecs = np.linalg.eigh(self.covs)
        eigvals = np.clip(eigvals, 1e-12, None)
        # eigh returns ascending eigenvalues; keep that order and fix
        # handedness so R is a proper rotation.
        det = np.linalg.det(eigvecs)
        eigvecs[det < 0, :, 2] *= -1.0
        scales = np.sqrt(eigvals)
        quats = _rotmat_to_quat(eigvecs)
        return scales, quats


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Batched rotation-matrix -> unit quaternion (w, x, y, z)."""
    n = R.shape[0]
    q = np.zeros((n, 4), dtype=np.float64)
    trace = np.einsum("nii->n", R)

    # Shepperd's method, branch per element for numerical safety.
    for k in range(n):
        m = R[k]
        t = trace[k]
        if t > 0:
            s = np.sqrt(t + 1.0) * 2.0
            q[k] = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
        elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            q[k] = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
        elif m[1, 1] >= m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            q[k] = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            q[k] = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
        if q[k, 0] < 0:
            q[k] = -q[k]
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    return q / norms


def concat(clouds: list[GaussianCloud], version: str = "") -> GaussianCloud:
    if not clouds:
        raise ValueError("cannot concatenate zero clouds")
    return GaussianCloud(
        np.concatenate([c.means for c in clouds]),
        np.concatenate([c.covs for c in clouds]),
        np.concatenate([c.weights for c in clouds]),
        np.concatenate([c.features for c in clouds]),
        np.concatenate([c.element_index for c in clouds]),
        version or clouds[0].version,
    )

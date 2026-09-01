"""Multi-scale fusion: moment-matched level-of-detail merging.

Bridges high-resolution architectural detail and macro-scale (GIS)
context with exact mixture moment matching: merging a group of weighted
Gaussians into one preserves total weight, mean, and covariance exactly
(law of total covariance):

    w   = sum w_k
    mu  = sum (w_k / w) mu_k
    S   = sum (w_k / w) [S_k + (mu_k - mu)(mu_k - mu)^T]

Levels:
  L0 — primitives (as Gaussianized)
  L1 — one Gaussian per element        (exactly the element box moments)
  L2 — one Gaussian per storey
  L3 — one Gaussian per building

The merge error of a group is reported as the KL divergence from the
moment-matched Gaussian to each member — the information lost by the
coarsening, so LoD selection is a measured tradeoff, not a guess.
``FrameTransform`` re-expresses a cloud in a different affine world frame
(e.g. a geo-referenced GIS frame): Gaussians are closed under affine maps,
so the transported cloud is exact, not resampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gat.geometry.primitives import GaussianCloud
from gat.geometry.stateio import GeometryScene


def moment_match(
    weights: np.ndarray, means: np.ndarray, covs: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Merge a weighted Gaussian set into one moment-matched Gaussian."""
    w_total = float(weights.sum())
    frac = weights / w_total
    mean = frac @ means
    centered = means - mean
    cov = np.einsum("k,kij->ij", frac, covs) + np.einsum(
        "k,ki,kj->ij", frac, centered, centered
    )
    return w_total, mean, cov


def kl_gauss(mu0, S0, mu1, S1) -> float:
    """KL(N0 || N1) for 3D Gaussians."""
    S1_inv = np.linalg.inv(S1)
    d = np.asarray(mu1) - np.asarray(mu0)
    trace = float(np.trace(S1_inv @ S0))
    m2 = float(d @ S1_inv @ d)
    _, logdet0 = np.linalg.slogdet(S0)
    _, logdet1 = np.linalg.slogdet(S1)
    return 0.5 * (trace + m2 - 3.0 + logdet1 - logdet0)


@dataclass(frozen=True)
class LodNode:
    label: str
    row: int               # element row for leaves; -1 for merged levels
    weight: float          # volume carried by this node [m3]
    mean: np.ndarray
    cov: np.ndarray
    n_members: int
    merge_error: float     # max KL(member || merged), 0 for leaves


def _element_node(scene: GeometryScene, element) -> LodNode:
    prims = scene.cloud.of_element(element.row)
    w, mu, S = moment_match(prims.weights, prims.means, prims.covs)
    err = max(
        (kl_gauss(prims.means[k], prims.covs[k], mu, S) for k in range(len(prims))),
        default=0.0,
    )
    return LodNode(element.name, element.row, w, mu, S, len(prims), err)


def element_level(scene: GeometryScene) -> tuple[LodNode, ...]:
    """L1: one moment-matched Gaussian per element (identified by row,
    so duplicate element names are harmless)."""
    return tuple(_element_node(scene, element) for element in scene.elements)


def merged_level(
    scene: GeometryScene, label: str, member_rows: list[int]
) -> LodNode:
    """Merge a set of elements into one macro Gaussian (L2/L3)."""
    member_set = set(member_rows)
    mask = np.isin(scene.cloud.element_index, member_rows)
    prims = scene.cloud.select(mask)
    w, mu, S = moment_match(prims.weights, prims.means, prims.covs)
    element_nodes = [
        _element_node(scene, e) for e in scene.elements if e.row in member_set
    ]
    err = max(
        (kl_gauss(n.mean, n.cov, mu, S) for n in element_nodes), default=0.0
    )
    return LodNode(label, -1, w, mu, S, len(prims), err)


def building_level(scene: GeometryScene) -> LodNode:
    """L3: the whole scene as a single geo-scale Gaussian."""
    return merged_level(
        scene, "building", [e.row for e in scene.elements if e.is_solid]
    )


@dataclass(frozen=True)
class FrameTransform:
    """Affine world-frame change ``x -> A x + b`` (e.g. into a GIS CRS)."""

    A: np.ndarray  # (3, 3)
    b: np.ndarray  # (3,)

    def apply_cloud(self, cloud: GaussianCloud) -> GaussianCloud:
        from gat.geometry.primitives import FEATURE_NAMES

        means = cloud.means @ self.A.T + self.b
        covs = np.einsum("ij,kjl,ml->kim", self.A, cloud.covs, self.A)
        scale = abs(float(np.linalg.det(self.A)))
        features = cloud.features.copy()
        # Keep the log_volume channel equal to log(weight) after scaling.
        features[:, FEATURE_NAMES.index("log_volume")] += np.log(scale)
        out = GaussianCloud(
            means,
            covs,
            cloud.weights * scale,
            features,
            cloud.element_index.copy(),
            cloud.version,
        )
        out.extras.update(cloud.extras)
        return out

"""Structural attention: semantic propagation over the Gaussian scene.

An analytic attention operator over primitive tokens.  Weights are
computed from geometry and semantics — scaled dot-product over feature
encodings, modulated by the Gaussian overlap kernel and the relationship
graph — with **no learned parameters anywhere**.  This is deterministic
kernel message passing wearing the attention API: the query/key/value
structure, row-stochastic weights, and stacked rounds are real, but
nothing here was trained, and this module says so rather than pretending
otherwise.  (When learned weights ever arrive, they slot into ``QK_SCALE``
and the affinity table without changing the operator's shape.)

Update rule (diffusion form, one round):

    payload_i' = payload_i + lam * sum_j alpha_ij (payload_j - payload_i)

with ``alpha`` row-stochastic.  For ``0 < lam <= 1`` this is a convex
combination, so payload values obey the maximum principle: propagation
smooths, never overshoots — asserted in tests.  Semantic identity channels
(class one-hots) are frozen; only payload channels move (README §14
principle 3: semantics are not Gaussianized — and not attention-ized
either).

``laplacian_baseline`` runs the same rounds with *uniform* adjacency
weights.  The demo compares the two: attention's content-dependence
(feature similarity + overlap kernel) is demonstrated against the
structure-only baseline, not asserted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gat.geometry.overlap import log_product_integral
from gat.geometry.primitives import CLASS_CHANNEL, PAYLOAD_CHANNELS, GaussianCloud
from gat.geometry.stateio import GeometryScene

#: Scale of the feature dot-product logits (1/sqrt(d_k) with d_k = payload dim).
QK_SCALE = 1.0 / np.sqrt(float(len(PAYLOAD_CHANNELS)))

#: Additive floor on adjacency so no softmax row is ever empty.
ADJACENCY_FLOOR = 1e-4

#: Class-affinity table (symmetric): how strongly classes listen to each
#: other.  Analytic prior, not learned weights.
_C = len(CLASS_CHANNEL)
CLASS_AFFINITY = np.ones((_C, _C), dtype=np.float64)
CLASS_AFFINITY[CLASS_CHANNEL["IfcWall"], CLASS_CHANNEL["IfcWall"]] = 2.0
CLASS_AFFINITY[CLASS_CHANNEL["IfcWall"], CLASS_CHANNEL["IfcDoor"]] = 1.5
CLASS_AFFINITY[CLASS_CHANNEL["IfcDoor"], CLASS_CHANNEL["IfcWall"]] = 1.5


@dataclass(frozen=True)
class AttentionConfig:
    lam: float = 0.5           # diffusion step size, in (0, 1]
    rounds: int = 3
    length_scale: float = 1.5  # metres; geometric kernel bandwidth

    def __post_init__(self) -> None:
        if not 0.0 < self.lam <= 1.0:
            raise ValueError("lam must be in (0, 1] for the maximum principle to hold")


def attention_weights(scene: GeometryScene, config: AttentionConfig) -> np.ndarray:
    """Row-stochastic (N, N) attention matrix over primitive tokens.

    logits_ij = QK_SCALE * (payload_i . payload_j)          [content]
              + log overlap kernel (bandwidth-widened)      [geometry]
              + log class affinity                          [semantics]
              + log(ADJACENCY_FLOOR + same-element bonus)   [structure]
    """
    cloud = scene.cloud
    n = len(cloud)
    payload = cloud.features[:, list(PAYLOAD_CHANNELS)]

    content = QK_SCALE * (payload @ payload.T)

    widen = np.eye(3) * config.length_scale**2
    geo = log_product_integral(
        cloud.means[:, None, :], cloud.covs[:, None, :, :] + widen,
        cloud.means[None, :, :], cloud.covs[None, :, :, :] + widen,
    )

    class_idx = np.argmax(cloud.features[:, : len(CLASS_CHANNEL)], axis=1)
    affinity = np.log(CLASS_AFFINITY[class_idx[:, None], class_idx[None, :]])

    same_element = (
        cloud.element_index[:, None] == cloud.element_index[None, :]
    ).astype(np.float64)
    # Graph-derived affinity between the primitives' parent elements: the
    # relationship graph (bounds, fills, shared-space adjacency) is the
    # structural prior for cross-element listening.
    related = scene.element_affinity[
        cloud.element_index[:, None], cloud.element_index[None, :]
    ]
    structure = np.log(ADJACENCY_FLOOR + same_element + related)

    logits = content + geo + affinity + structure
    np.fill_diagonal(logits, -np.inf)  # no self-attention; the residual term keeps self

    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    weights /= weights.sum(axis=1, keepdims=True)
    assert weights.shape == (n, n)
    return weights


def propagate(
    scene: GeometryScene,
    config: AttentionConfig = AttentionConfig(),
) -> tuple[GaussianCloud, np.ndarray]:
    """Run attention rounds; returns (new cloud, final weight matrix)."""
    scene.check_fresh(scene.world)
    cloud = scene.cloud
    features = cloud.features.copy()
    payload_cols = list(PAYLOAD_CHANNELS)

    alpha = attention_weights(scene, config)
    for _ in range(config.rounds):
        payload = features[:, payload_cols]
        delta = alpha @ payload - payload
        features[:, payload_cols] = payload + config.lam * delta

    out = GaussianCloud(
        cloud.means.copy(),
        cloud.covs.copy(),
        cloud.weights.copy(),
        features,
        cloud.element_index.copy(),
        cloud.version,
    )
    out.extras.update(cloud.extras)
    return out, alpha


def laplacian_baseline(
    scene: GeometryScene,
    config: AttentionConfig = AttentionConfig(),
    radius: float = 2.0,
) -> GaussianCloud:
    """Ablation: identical rounds with uniform weights over a distance graph.

    Content-blind — every neighbour within ``radius`` counts equally.  The
    demo contrasts this with :func:`propagate` to show that the attention
    weights actually condition on features and overlap, not just topology.
    """
    cloud = scene.cloud
    features = cloud.features.copy()
    payload_cols = list(PAYLOAD_CHANNELS)

    dist = np.linalg.norm(
        cloud.means[:, None, :] - cloud.means[None, :, :], axis=-1
    )
    adjacency = (dist < radius).astype(np.float64)
    np.fill_diagonal(adjacency, 0.0)
    adjacency += ADJACENCY_FLOOR
    weights = adjacency / adjacency.sum(axis=1, keepdims=True)

    for _ in range(config.rounds):
        payload = features[:, payload_cols]
        features[:, payload_cols] = payload + config.lam * (weights @ payload - payload)

    out = GaussianCloud(
        cloud.means.copy(),
        cloud.covs.copy(),
        cloud.weights.copy(),
        features,
        cloud.element_index.copy(),
        cloud.version,
    )
    out.extras.update(cloud.extras)
    return out


def element_payload_means(scene: GeometryScene, cloud: GaussianCloud) -> dict[str, np.ndarray]:
    """Weight-averaged payload per element — the element-level readout."""
    out: dict[str, np.ndarray] = {}
    payload_cols = list(PAYLOAD_CHANNELS)
    for element in scene.elements:
        prims = cloud.select(cloud.element_index == element.row)
        w = prims.weights / prims.weights.sum()
        out[element.name] = w @ prims.features[:, payload_cols]
    return out

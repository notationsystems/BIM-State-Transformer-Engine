"""Tests for gat/geometry/attention.py — structural attention propagation.

Covers: row-stochastic attention weights, the maximum principle on payload
channels, frozen identity (class one-hot) channels, content-dependence
versus the content-blind Laplacian baseline (door u-value drift), bitwise
determinism, and AttentionConfig validation.

Demo model facts used below (gat/demo/model.ifc): 5 walls, 2 spaces,
1 door with distinct default u-values (external wall 0.25, internal wall
0.80, door 1.80); payload channels are (u_value, load_bearing, external)
= feature columns (6, 7, 8); class one-hots occupy columns 0..4.
"""

from __future__ import annotations

import os
import unittest

import numpy as np

import gat.demo
from gat.geometry.attention import (
    AttentionConfig,
    attention_weights,
    element_payload_means,
    laplacian_baseline,
    propagate,
)
from gat.geometry.primitives import CLASS_CHANNEL, PAYLOAD_CHANNELS
from gat.geometry.stateio import derive_scene
from gat.session import GatSession

MODEL = os.path.join(os.path.dirname(gat.demo.__file__), "model.ifc")

_CACHE: dict = {}


def _scene():
    if "scene" not in _CACHE:
        session = GatSession.load_ifc(MODEL)
        _CACHE["scene"] = derive_scene(session.world)
    return _CACHE["scene"]


class AttentionTestBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = _scene()
        cls.config = AttentionConfig()


class TestAttentionWeights(AttentionTestBase):
    def test_rows_sum_to_one(self):
        alpha = attention_weights(self.scene, self.config)
        n = len(self.scene.cloud)
        self.assertEqual(alpha.shape, (n, n))
        row_sums = alpha.sum(axis=1)
        # Row-stochastic to 1e-12: softmax rows are normalized explicitly.
        self.assertLess(np.abs(row_sums - 1.0).max(), 1e-12)

    def test_weights_nonnegative_and_no_self_attention(self):
        alpha = attention_weights(self.scene, self.config)
        self.assertGreaterEqual(alpha.min(), 0.0)
        # Diagonal logits are -inf, so self-weight is exactly zero.
        self.assertEqual(np.abs(np.diag(alpha)).max(), 0.0)


class TestPropagate(AttentionTestBase):
    def test_maximum_principle_on_payload(self):
        out, _ = propagate(self.scene, self.config)
        before = self.scene.cloud.features
        for col in PAYLOAD_CHANNELS:
            lo, hi = before[:, col].min(), before[:, col].max()
            self.assertGreaterEqual(out.features[:, col].min(), lo - 1e-12)
            self.assertLessEqual(out.features[:, col].max(), hi + 1e-12)

    def test_identity_channels_bit_identical(self):
        out, _ = propagate(self.scene, self.config)
        onehot_cols = sorted(CLASS_CHANNEL.values())
        self.assertTrue(
            np.array_equal(
                out.features[:, onehot_cols], self.scene.cloud.features[:, onehot_cols]
            )
        )
        # Every non-payload channel is frozen, not just the one-hots.
        frozen = [
            c for c in range(out.features.shape[1]) if c not in PAYLOAD_CHANNELS
        ]
        self.assertTrue(
            np.array_equal(
                out.features[:, frozen], self.scene.cloud.features[:, frozen]
            )
        )

    def test_geometry_untouched(self):
        out, _ = propagate(self.scene, self.config)
        self.assertTrue(np.array_equal(out.means, self.scene.cloud.means))
        self.assertTrue(np.array_equal(out.covs, self.scene.cloud.covs))
        self.assertTrue(np.array_equal(out.weights, self.scene.cloud.weights))

    def test_deterministic_across_runs(self):
        out1, alpha1 = propagate(self.scene, self.config)
        out2, alpha2 = propagate(self.scene, self.config)
        self.assertTrue(np.array_equal(out1.features, out2.features))
        self.assertTrue(np.array_equal(alpha1, alpha2))

    def test_content_dependence_vs_laplacian_baseline(self):
        # The door's u-value (1.80) is a semantic outlier; attention should
        # preserve it better than the content-blind uniform baseline.
        att_cloud, _ = propagate(self.scene, self.config)
        lap_cloud = laplacian_baseline(self.scene, self.config)
        base = element_payload_means(self.scene, self.scene.cloud)
        att = element_payload_means(self.scene, att_cloud)
        lap = element_payload_means(self.scene, lap_cloud)
        # Payload index 0 is u_value (PAYLOAD_CHANNELS[0] == feature col 6).
        drift_att = abs(att["Door-1"][0] - base["Door-1"][0])
        drift_lap = abs(lap["Door-1"][0] - base["Door-1"][0])
        self.assertLess(drift_att, drift_lap)


class TestAttentionConfig(unittest.TestCase):
    def test_rejects_lam_zero(self):
        with self.assertRaises(ValueError):
            AttentionConfig(lam=0.0)

    def test_rejects_lam_above_one(self):
        with self.assertRaises(ValueError):
            AttentionConfig(lam=1.0 + 1e-9)

    def test_accepts_lam_one(self):
        self.assertEqual(AttentionConfig(lam=1.0).lam, 1.0)


if __name__ == "__main__":
    unittest.main()

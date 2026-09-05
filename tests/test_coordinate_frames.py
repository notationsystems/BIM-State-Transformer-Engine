"""Analytic, metamorphic and finite-difference checks, not field calibration."""
import math
from statistics import NormalDist
import unittest

import numpy as np

from gat.geometry.frames import CoordinateFrame, FrameGraph, RigidTransform


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


class CoordinateFrameTests(unittest.TestCase):
    def setUp(self):
        self.building_from_storey = RigidTransform(rotation([0, 0, 1], 0.7), [3, -2, 4])
        self.storey_from_element = RigidTransform(rotation([1, 0, 0], 0.4), [1, 2, 0.5])
        self.frames = FrameGraph([
            CoordinateFrame("building", None, RigidTransform.identity()),
            CoordinateFrame("storey", "building", self.building_from_storey),
            CoordinateFrame("element", "storey", self.storey_from_element, "mm"),
        ])

    def test_nested_placement_agrees_with_direct_composition(self):
        point_m = np.array([0.25, 0.5, 1.0])
        expected = self.building_from_storey.point(self.storey_from_element.point(point_m))
        np.testing.assert_allclose(
            self.frames.point(point_m * 1000, "element", "building"), expected,
            atol=1e-12, rtol=0,
        )

    def test_point_and_covariance_roundtrip_between_metres_and_millimetres(self):
        point = np.array([250, 500, 1000])
        covariance = np.array([[4, 1, 0], [1, 9, 2], [0, 2, 16]], dtype=float)
        mapped = self.frames.point(point, "element", "building")
        mapped_cov = self.frames.covariance(covariance, "element", "building")
        np.testing.assert_allclose(self.frames.point(mapped, "building", "element"), point, atol=1e-9, rtol=0)
        np.testing.assert_allclose(self.frames.covariance(mapped_cov, "building", "element"), covariance, atol=1e-12, rtol=0)
        self.assertAlmostEqual(np.trace(mapped_cov), np.trace(covariance) * 1e-6)

    def test_composition_order_and_inverse(self):
        first = RigidTransform(rotation([0, 0, 1], math.pi / 2), [1, 0, 0])
        second = RigidTransform(np.eye(3), [2, 0, 0])
        np.testing.assert_allclose(first.compose(second).point([0, 0, 0]), [1, 2, 0], atol=1e-12)
        np.testing.assert_allclose(second.compose(first).point([0, 0, 0]), [3, 0, 0], atol=1e-12)
        np.testing.assert_allclose(first.inverse().point(first.point([1, 2, 3])), [1, 2, 3], atol=1e-12)

    def test_invalid_rotations_units_and_handedness_are_rejected(self):
        for matrix in (np.diag([-1, 1, 1]), np.diag([2, 1, 1]), np.ones((3, 3)), np.full((3, 3), np.nan)):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    RigidTransform(matrix, [0, 0, 0])
        with self.assertRaises(ValueError):
            CoordinateFrame("root", None, RigidTransform.identity(), "feet")
        with self.assertRaises(ValueError):
            CoordinateFrame("root", None, RigidTransform.identity(), handedness="left")

    def test_invalid_frame_topology_is_rejected(self):
        root = CoordinateFrame("root", None, RigidTransform.identity())
        a = CoordinateFrame("a", "b", RigidTransform.identity())
        b = CoordinateFrame("b", "a", RigidTransform.identity())
        for records in ([], [root, root], [root, a], [root, a, b]):
            with self.subTest(records=records):
                with self.assertRaises(ValueError):
                    FrameGraph(records)
        with self.assertRaises(ValueError):
            self.frames.point([0, 0, 0], "missing", "building")

    def test_constructor_copies_rotation_and_origin(self):
        matrix, origin = np.eye(3), np.zeros(3)
        transform = RigidTransform(matrix, origin)
        matrix[:] = 0
        origin[:] = 42
        np.testing.assert_array_equal(transform.rotation, np.eye(3))
        np.testing.assert_array_equal(transform.translation_m, np.zeros(3))

    def test_representation_identity_is_separate_from_physical_units_equivalence(self):
        metre = FrameGraph([CoordinateFrame("root", None, RigidTransform.identity(), "m")])
        millimetre = FrameGraph([CoordinateFrame("root", None, RigidTransform.identity(), "mm")])
        self.assertNotEqual(metre.representation_digest(), millimetre.representation_digest())
        self.assertEqual(
            self.frames.representation_digest(),
            FrameGraph(reversed(tuple(self.frames.frames.values()))).representation_digest(),
        )


class SharedPoseTests(unittest.TestCase):
    def test_shared_translation_cancels_from_relative_position(self):
        points = np.array([[0, 0, 0], [2, 0, 0]])
        covariance = np.zeros((12, 12))
        covariance[6:9, 6:9] = np.eye(3) * 0.25
        _, result = RigidTransform.identity().propagate_points(points, covariance)
        difference = np.hstack((-np.eye(3), np.eye(3)))
        np.testing.assert_allclose(difference @ result @ difference.T, np.zeros((3, 3)), atol=1e-14)
        # Independent marginal treatment would incorrectly invent relative error.
        np.testing.assert_allclose(result[:3, :3] + result[3:, 3:], np.eye(3) * 0.5)

    def test_shared_rotation_produces_correlated_relative_uncertainty(self):
        points = np.array([[1, 0, 0], [3, 0, 0]])
        covariance = np.zeros((12, 12))
        covariance[-1, -1] = 0.01 ** 2  # local z rotation
        _, result = RigidTransform.identity().propagate_points(points, covariance)
        difference = np.hstack((-np.eye(3), np.eye(3)))
        relative = difference @ result @ difference.T
        self.assertAlmostEqual(relative[1, 1], 4e-4)
        self.assertAlmostEqual(result[1, 4], 3e-4)
        self.assertAlmostEqual(relative[0, 0], 0)

    def test_full_joint_propagation_matches_finite_difference(self):
        points = np.array([[0.2, -0.4, 0.6], [1.1, 0.3, -0.2]])
        transform = RigidTransform(rotation([1, 2, 3], 0.5), [4, -3, 2])
        rng = np.random.default_rng(120)
        factor = rng.normal(size=(12, 12)) * 0.001
        covariance = factor @ factor.T  # includes point/pose cross-correlations
        _, propagated = transform.propagate_points(points, covariance)

        def evaluate(delta):
            updated = points + delta[:6].reshape(2, 3)
            omega = delta[-3:]
            angle = np.linalg.norm(omega)
            perturbed = rotation(omega, angle) if angle else np.eye(3)
            updated = updated @ perturbed.T + delta[6:9]
            return (updated @ transform.rotation.T + transform.translation_m).ravel()

        epsilon = 1e-6
        jacobian = np.column_stack([
            (evaluate(delta) - evaluate(-delta)) / (2 * epsilon)
            for delta in np.eye(12) * epsilon
        ])
        np.testing.assert_allclose(propagated, jacobian @ covariance @ jacobian.T, atol=1e-12, rtol=1e-8)

    def test_opening_fit_probability_is_invariant_under_global_frame_change(self):
        # Synthetic 2 m opening and 1.99 m assembly, measured on a shared axis.
        points = np.array([[-1, 0, 0], [1, 0, 0], [-0.995, 0, 0], [0.995, 0, 0]])
        covariance = np.diag([2e-5] * 12 + [0.04] * 3 + [1e-4] * 3)
        transforms = [
            RigidTransform.identity(),
            RigidTransform(np.eye(3), [100, -20, 3]),
            RigidTransform(rotation([1, 2, -1], 1.2), [-7, 11, 5]),
        ]
        outcomes = []
        for transform in transforms:
            moved, joint = transform.propagate_points(points, covariance)
            axis = transform.rotation @ np.array([1, 0, 0])
            weights = np.concatenate([-axis, axis, axis, -axis])
            margin = float(weights @ moved.ravel())
            variance = float(weights @ joint @ weights)
            outcomes.append((margin, variance, NormalDist().cdf(margin / math.sqrt(variance))))
        for outcome in outcomes:
            np.testing.assert_allclose(outcome, outcomes[0], atol=1e-10, rtol=0)
        self.assertAlmostEqual(outcomes[0][0], 0.01)
        self.assertAlmostEqual(outcomes[0][1], 8e-5)

    def test_invalid_joint_covariances_are_rejected(self):
        transform = RigidTransform.identity()
        negative = np.eye(9)
        negative[0, 0] = -1
        asymmetric = np.eye(9)
        asymmetric[0, 1] = 0.5
        for covariance in (np.eye(6), negative, asymmetric, np.full((9, 9), np.nan)):
            with self.subTest(covariance=covariance):
                with self.assertRaises(ValueError):
                    transform.propagate_points([[0, 0, 0]], covariance)

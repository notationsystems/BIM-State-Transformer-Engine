"""Calibrated scan-to-clearance likelihood adapter.

Registration answers whether scan points can be associated with the
Gaussian BIM field.  It must not also provide the pose used to claim that
the BIM dimensions are wrong: doing so would feed a model-derived alignment
back as apparently independent evidence about that same model.

This adapter therefore requires an independently calibrated model-from-scan
pose (survey control or a separately calibrated SLAM trajectory).  An
accepted registration is retained only as a fit/association gate.  For the
element and separating direction selected by clearance assurance it:

* extracts responsibility-weighted points near the controlling support face,
* rejects weak, ambiguous, clustered, or non-planar support,
* estimates the support coordinate and decomposes sampling, pose, and
  systematic calibration variance,
* checks pose agreement and the normalized measurement innovation, and
* emits a provenance-bound :class:`~gat.engine.transform.ObserveLinearized`.

The returned transformation still enters the ordinary
condition -> propagate -> verify -> commit/rollback pipeline.  No function
in this module mutates canonical BIM state directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

from gat.engine.transform import ObserveLinearized
from gat.errors import LikelihoodCalibrationError
from gat.geometry.assurance import (
    ClearanceEvidencePlan,
    InspectionAction,
)
from gat.geometry.registration import (
    RegistrationResult,
    RigidTransformZ,
    ScanEvidenceReport,
    ScanRegistrar,
)
from gat.geometry.stateio import GeometryScene, rot_z, support_radius


@dataclass(frozen=True)
class IndependentPoseCalibration:
    """Externally sourced model-from-scan pose and covariance.

    ``covariance`` is ordered ``(yaw, tx, ty, tz)``.  ``source_id`` must
    identify the survey control, calibrated SLAM trajectory, or equivalent
    provenance; a registration fit against this BIM is not independent.
    """

    transform: RigidTransformZ
    covariance: np.ndarray
    scan_digest: str
    source_id: str

    def __post_init__(self) -> None:
        values = (self.transform.theta, *self.transform.t)
        if not np.isfinite(values).all():
            raise ValueError("independent pose must be finite")
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (4, 4) or not np.isfinite(covariance).all():
            raise ValueError("independent pose covariance must be finite 4x4")
        if not np.allclose(covariance, covariance.T, atol=1e-12, rtol=0.0):
            raise ValueError("independent pose covariance must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) <= 0.0:
            raise ValueError("independent pose covariance must be positive definite")
        if not self.scan_digest or not self.source_id.strip():
            raise ValueError("independent pose provenance must be non-empty")
        copied = covariance.copy()
        copied.setflags(write=False)
        object.__setattr__(self, "covariance", copied)


@dataclass(frozen=True)
class ClearanceLikelihoodCalibration:
    """Declared physical noise model and evidence acceptance gates."""

    sensor_sigma: float = 0.010
    calibration_sigma: float = 0.005
    face_band: float = 0.060
    min_face_effective_points: float = 15.0
    min_tangent_rms: float = 0.100
    min_element_effective_points: float = 25.0
    min_support_diversity: float = 0.50
    min_assignment_confidence: float = 0.80
    min_face_alignment: float = 0.98
    max_pose_disagreement_m2: float = 16.0
    max_innovation_sigma: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "sensor_sigma",
            "calibration_sigma",
            "face_band",
            "min_face_effective_points",
            "min_tangent_rms",
            "min_element_effective_points",
            "max_pose_disagreement_m2",
            "max_innovation_sigma",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "min_support_diversity",
            "min_assignment_confidence",
            "min_face_alignment",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")


@dataclass(frozen=True)
class ScanClearanceLikelihood:
    """Auditable scalar support-plane likelihood ready for conditioning."""

    element_name: str
    direction: tuple[float, float, float]
    predicted_support: float
    observed_support: float
    effective_face_points: float
    face_assignment_confidence: float
    tangent_rms: float
    sampling_sigma: float
    pose_sigma: float
    calibration_sigma: float
    noise_sigma: float
    innovation_sigma: float
    pose_disagreement_m2: float
    scan_digest: str
    scene_version: str
    pose_source_id: str
    evidence_digest: str
    observation: ObserveLinearized

    def render(self) -> str:
        return (
            f"scan likelihood {self.element_name}: support "
            f"{self.observed_support:.4f} m (predicted "
            f"{self.predicted_support:.4f} m); noise={self.noise_sigma:.4f} m "
            f"[sampling {self.sampling_sigma:.4f}, pose {self.pose_sigma:.4f}, "
            f"calibration {self.calibration_sigma:.4f}]; "
            f"face_mass={self.effective_face_points:.1f}; "
            f"assignment={self.face_assignment_confidence:.3f}; "
            f"innovation={self.innovation_sigma:.3f} sigma"
        )


def adapt_clearance_likelihood(
    scene: GeometryScene,
    registrar: ScanRegistrar,
    scan: np.ndarray,
    registration: RegistrationResult,
    evidence: ScanEvidenceReport,
    plan: ClearanceEvidencePlan,
    pose: IndependentPoseCalibration,
    calibration: ClearanceLikelihoodCalibration = ClearanceLikelihoodCalibration(),
) -> ScanClearanceLikelihood:
    """Extract and calibrate the selected clearance support-plane evidence.

    Every input is provenance-bound.  Failure of any declared gate raises
    :class:`LikelihoodCalibrationError`; it never degrades silently to an
    update with inflated but untraceable uncertainty.
    """
    scene.check_fresh(scene.world)
    selected = plan.selected
    if selected is None or selected.action is not InspectionAction.EXTRACT_SCAN_MEASUREMENT:
        raise LikelihoodCalibrationError(
            "clearance plan does not authorize scan measurement extraction"
        )
    if plan.assessment.scene_version != scene.version:
        raise LikelihoodCalibrationError(
            "clearance plan belongs to a different canonical scene"
        )
    if evidence.scene_version != scene.version or registration.scene_version != scene.version:
        raise LikelihoodCalibrationError(
            "scan evidence, registration, and canonical scene versions differ"
        )
    if not (
        plan.scan_digest
        == evidence.scan_digest
        == registration.scan_digest
        == pose.scan_digest
    ):
        raise LikelihoodCalibrationError(
            "scan, evidence plan, registration, and independent pose digests differ"
        )

    points = np.asarray(scan, dtype=np.float64)
    if evidence.point_count != points.shape[0]:
        raise LikelihoodCalibrationError(
            "scan evidence point count differs from the supplied scan"
        )
    element = scene.element_by_name(selected.element_name)
    evidence_rows = [
        row
        for row in evidence.elements
        if row.element_name == element.name and row.element_row == element.row
    ]
    if len(evidence_rows) != 1:
        raise LikelihoodCalibrationError(
            f"expected one evidence row for {element.name}, found {len(evidence_rows)}"
        )
    element_evidence = evidence_rows[0]
    if element_evidence.effective_points < calibration.min_element_effective_points:
        raise LikelihoodCalibrationError("element evidence has insufficient effective points")
    if element_evidence.support_diversity < calibration.min_support_diversity:
        raise LikelihoodCalibrationError("element evidence has insufficient support diversity")
    if element_evidence.assignment_confidence < calibration.min_assignment_confidence:
        raise LikelihoodCalibrationError("element evidence assignment is ambiguous")

    direction = np.asarray(selected.direction, dtype=np.float64)
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction).all() or direction_norm <= 0.0:
        raise LikelihoodCalibrationError("clearance direction must be finite and nonzero")
    direction = direction / direction_norm
    R_element = rot_z(element.box.angle)
    face_alignment = float(np.max(np.abs(direction @ R_element)))
    if face_alignment < calibration.min_face_alignment:
        raise LikelihoodCalibrationError(
            "clearance support is an edge/corner, not a calibrated element face"
        )

    pose_m2 = _pose_disagreement_m2(registration, pose)
    if pose_m2 > calibration.max_pose_disagreement_m2:
        raise LikelihoodCalibrationError(
            f"independent pose disagrees with registration (m2={pose_m2:.3f})"
        )

    posterior = registrar.posterior_at(scan, registration, pose.transform)
    primitive_mask = registrar.element_index == element.row
    if not np.any(primitive_mask):
        raise LikelihoodCalibrationError("selected element has no registration primitives")
    target_gamma = posterior.responsibilities[:, primitive_mask].sum(axis=1)

    center = element.box.center()
    predicted = float(direction @ center + support_radius(element, direction))
    projections = posterior.model_points @ direction
    face_mask = np.abs(projections - predicted) <= calibration.face_band
    weights = target_gamma * face_mask
    face_mass = float(weights.sum())
    if face_mass < calibration.min_face_effective_points:
        raise LikelihoodCalibrationError(
            f"support face has only {face_mass:.3f} effective points"
        )

    point_inlier_mass = posterior.responsibilities.sum(axis=1)
    target_share = np.divide(
        target_gamma,
        point_inlier_mass,
        out=np.zeros_like(target_gamma),
        where=point_inlier_mass > 0.0,
    )
    face_assignment = float(np.sum(weights * target_share) / face_mass)
    if face_assignment < calibration.min_assignment_confidence:
        raise LikelihoodCalibrationError("support-face assignment is ambiguous")

    observed = float(np.sum(weights * projections) / face_mass)
    centered_points = posterior.model_points - np.sum(
        weights[:, None] * posterior.model_points, axis=0
    ) / face_mass
    normal_offset = centered_points @ direction
    tangent = centered_points - normal_offset[:, None] * direction[None, :]
    tangent_rms = float(
        np.sqrt(np.sum(weights * np.square(tangent).sum(axis=1)) / face_mass)
    )
    if tangent_rms < calibration.min_tangent_rms:
        raise LikelihoodCalibrationError("support-face samples are spatially clustered")

    residual_variance = float(
        np.sum(weights * np.square(projections - observed)) / face_mass
    )
    sampling_variance = max(residual_variance, calibration.sensor_sigma**2) / face_mass
    pose_variance = _support_pose_variance(
        points, weights, face_mass, direction, pose
    )
    noise_variance = (
        sampling_variance + pose_variance + calibration.calibration_sigma**2
    )

    proj = np.abs(direction @ R_element)
    row = (
        direction @ scene.center_jacobian_wrt_raw(element)
        + 0.5 * proj @ scene.extent_jacobians[element.row]
    )
    if not np.any(row != 0.0):
        raise LikelihoodCalibrationError(
            "selected support face is not coupled to any uncertain BIM parameter"
        )
    prior_variance = float(row @ scene.world.belief.sigma @ row)
    innovation_std = math.sqrt(max(prior_variance + noise_variance, 0.0))
    innovation_sigma = (
        abs(observed - predicted) / innovation_std
        if innovation_std > 0.0
        else math.inf
    )
    if innovation_sigma > calibration.max_innovation_sigma:
        raise LikelihoodCalibrationError(
            f"support observation failed innovation gate ({innovation_sigma:.3f} sigma)"
        )

    raw_targets = tuple(
        scene.world.binding.raw_index.var(int(index))
        for index in np.flatnonzero(row)
    )
    evidence_digest = _likelihood_digest(
        scene,
        element.name,
        direction,
        observed,
        face_mass,
        face_assignment,
        pose,
        calibration,
    )
    observation = ObserveLinearized(
        row=row,
        predicted=predicted,
        observed=observed,
        noise_sigma=math.sqrt(noise_variance),
        raw_targets=raw_targets,
        expected_raw_order=scene.world.binding.raw_index.vars,
        expected_belief_digest=scene.world.belief.digest(),
        expected_world_digest=scene.world.digest(),
        evidence_digest=evidence_digest,
        label=f"scan support face {element.name}",
    )
    return ScanClearanceLikelihood(
        element_name=element.name,
        direction=tuple(float(value) for value in direction),
        predicted_support=predicted,
        observed_support=observed,
        effective_face_points=face_mass,
        face_assignment_confidence=face_assignment,
        tangent_rms=tangent_rms,
        sampling_sigma=math.sqrt(sampling_variance),
        pose_sigma=math.sqrt(max(pose_variance, 0.0)),
        calibration_sigma=calibration.calibration_sigma,
        noise_sigma=math.sqrt(noise_variance),
        innovation_sigma=innovation_sigma,
        pose_disagreement_m2=pose_m2,
        scan_digest=evidence.scan_digest,
        scene_version=scene.version,
        pose_source_id=pose.source_id,
        evidence_digest=evidence_digest,
        observation=observation,
    )


def _pose_disagreement_m2(
    registration: RegistrationResult, pose: IndependentPoseCalibration
) -> float:
    information = np.asarray(registration.info_matrix, dtype=np.float64)
    if information.shape != (4, 4) or not np.isfinite(information).all():
        raise LikelihoodCalibrationError("registration information matrix is invalid")
    try:
        fit_covariance = np.linalg.inv(information)
        combined = 0.5 * (
            fit_covariance + pose.covariance
            + (fit_covariance + pose.covariance).T
        )
        delta = np.array(
            [
                (pose.transform.theta - registration.transform.theta + math.pi)
                % (2.0 * math.pi)
                - math.pi,
                *(np.asarray(pose.transform.t) - np.asarray(registration.transform.t)),
            ],
            dtype=np.float64,
        )
        return float(delta @ np.linalg.solve(combined, delta))
    except np.linalg.LinAlgError as exc:
        raise LikelihoodCalibrationError(
            "pose agreement covariance is singular"
        ) from exc


def _support_pose_variance(
    scan_points: np.ndarray,
    weights: np.ndarray,
    face_mass: float,
    direction: np.ndarray,
    pose: IndependentPoseCalibration,
) -> float:
    theta = pose.transform.theta
    dR = np.array(
        [
            [-math.sin(theta), -math.cos(theta), 0.0],
            [math.cos(theta), -math.sin(theta), 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    dpoints_dtheta = scan_points @ dR.T
    dtheta = float(np.sum(weights * (dpoints_dtheta @ direction)) / face_mass)
    jacobian = np.array([dtheta, *direction], dtype=np.float64)
    return max(float(jacobian @ pose.covariance @ jacobian), 0.0)


def _likelihood_digest(
    scene: GeometryScene,
    element_name: str,
    direction: np.ndarray,
    observed: float,
    face_mass: float,
    face_assignment: float,
    pose: IndependentPoseCalibration,
    calibration: ClearanceLikelihoodCalibration,
) -> str:
    digest = hashlib.sha256()
    payload = {
        "scene": scene.version,
        "scan": pose.scan_digest,
        "pose_source": pose.source_id,
        "element": element_name,
        "observed": observed,
        "face_mass": face_mass,
        "face_assignment": face_assignment,
        "calibration": asdict(calibration),
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    digest.update(np.asarray(direction, dtype="<f8").tobytes())
    digest.update(
        np.asarray(
            [pose.transform.theta, *pose.transform.t], dtype="<f8"
        ).tobytes()
    )
    digest.update(np.asarray(pose.covariance, dtype="<f8").tobytes())
    return digest.hexdigest()

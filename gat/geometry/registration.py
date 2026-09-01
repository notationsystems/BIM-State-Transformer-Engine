"""Scan-to-BIM registration: aligning point clouds to the Gaussian field.

The Gaussianized building is a mixture model; registering a scan is
maximum-likelihood estimation of the rigid transform that carries scan
points into the model frame:

    NLL(T) = -mean_n log [ (1-pi_out) sum_k w_k N(T x_n; mu_k, S_k) + pi_out / V ]

with a uniform outlier component over the scene bounding box.  v0
estimates the 4-DOF gravity-aligned transform (yaw + translation) —
buildings and scanners agree about *up* — via:

* EM outer loop: responsibilities from the current transform,
* M-step: closed-form generalized-least-squares translation, then a
  Gauss-Newton yaw step, Armijo-guarded on the true NLL so the outer
  iteration is monotone,
* 8 deterministic yaw starts (k * pi/4), each with centroid-matched
  initial translation; best final NLL wins, ties broken by start index.

The reported information matrix is the *complete-data* (responsibility-
weighted) Gauss-Newton Hessian at the optimum.  For a mixture likelihood
this overstates the observed information (the missing-data correction of
Louis' identity is not subtracted), so ``pose_sigma`` is a lower bound on
the pose uncertainty — useful for gating and relative comparisons, and
labeled as the approximation it is.  This module reports it and gates on fit
quality.  Dimensional conditioning is handled separately
by :mod:`gat.geometry.scan_likelihood`, which requires an independent pose
and measures a decision-controlling support face; the naive surface-centroid
observation of a partially visible element remains deliberately unsupported.

Everything is deterministic: scan synthesis takes an explicit seed, and
the optimizer has no stochastic steps.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from gat.errors import RegistrationError
from gat.geometry.stateio import GeometryScene, rot_z


@dataclass(frozen=True)
class RigidTransformZ:
    """x -> R_z(theta) x + t."""

    theta: float
    t: tuple[float, float, float]

    def apply(self, points: np.ndarray) -> np.ndarray:
        return points @ rot_z(self.theta).T + np.asarray(self.t)

    def compose_error(self, other: "RigidTransformZ") -> tuple[float, float]:
        """(yaw error, translation error) between this and another transform."""
        dtheta = abs(
            (self.theta - other.theta + math.pi) % (2.0 * math.pi) - math.pi
        )
        dt = float(np.linalg.norm(np.asarray(self.t) - np.asarray(other.t)))
        return dtheta, dt


@dataclass(frozen=True)
class RegistrationResult:
    transform: RigidTransformZ
    nll: float
    nll_trace: tuple[float, ...]        # fine-stage trace (monotone)
    coarse_trace: tuple[float, ...]     # coarse-stage trace (monotone)
    start_nlls: tuple[float, ...]
    info_matrix: np.ndarray        # (4, 4) over (theta, tx, ty, tz)
    accepted: bool                 # fit-quality gate for any write-back
    scan_digest: str               # binds downstream evidence to exact input bytes
    scene_version: str             # canonical-world digest used to derive the field

    def pose_sigma(self) -> np.ndarray:
        """Marginal standard deviations of the pose estimate.

        Derived from the complete-data Gauss-Newton information matrix, so
        these are LOWER bounds on the true pose uncertainty (see module
        docstring).
        """
        cov = np.linalg.inv(self.info_matrix)
        return np.sqrt(np.clip(np.diag(cov), 0.0, None))


@dataclass(frozen=True)
class ElementScanEvidence:
    """Responsibility-based evidence for one canonical BIM element.

    ``support_diversity`` is the normalized effective number of supported
    primitives (1/K..1), not a claim of physical surface coverage.
    ``assignment_confidence`` is one for evidence assigned exclusively to
    this element and falls when nearby elements share responsibility.
    """

    element_row: int
    element_name: str
    primitive_count: int
    effective_points: float
    responsibility_fraction: float
    mean_mahalanobis2: float
    support_diversity: float
    assignment_confidence: float


@dataclass(frozen=True)
class ScanEvidenceReport:
    """Auditable scan evidence produced only after an accepted fit gate."""

    scan_digest: str
    scene_version: str
    point_count: int
    inlier_effective_points: float
    outlier_fraction: float
    elements: tuple[ElementScanEvidence, ...]

    def render(self) -> str:
        lines = [
            "SCAN EVIDENCE REPORT",
            f"points={self.point_count} "
            f"inlier_mass={self.inlier_effective_points:.3f} "
            f"outlier_fraction={self.outlier_fraction:.3f}",
        ]
        for evidence in self.elements:
            lines.append(
                f"{evidence.element_name}: mass={evidence.effective_points:.3f} "
                f"share={evidence.responsibility_fraction:.3f} "
                f"fit_m2={evidence.mean_mahalanobis2:.3f} "
                f"diversity={evidence.support_diversity:.3f} "
                f"confidence={evidence.assignment_confidence:.3f}"
            )
        return "\n".join(lines)


@dataclass(frozen=True)
class RegisteredScanPosterior:
    """Fine-scale scan/primitive posterior at a declared external pose.

    This is an inspectable likelihood product, not canonical state.  The
    registration result remains the fit and association gate; callers may
    supply an independently calibrated pose so model-derived registration
    error is not recycled as evidence about the model itself.
    """

    model_points: np.ndarray
    responsibilities: np.ndarray
    mahalanobis2: np.ndarray
    scan_digest: str
    scene_version: str

    def __post_init__(self) -> None:
        points = np.asarray(self.model_points, dtype=np.float64)
        gamma = np.asarray(self.responsibilities, dtype=np.float64)
        m2 = np.asarray(self.mahalanobis2, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("model_points must have shape (n, 3)")
        if gamma.ndim != 2 or gamma.shape != m2.shape:
            raise ValueError("responsibilities and mahalanobis2 must have equal 2D shapes")
        if gamma.shape[0] != points.shape[0]:
            raise ValueError("posterior point and responsibility counts differ")
        if not np.isfinite(points).all() or not np.isfinite(gamma).all() or not np.isfinite(m2).all():
            raise ValueError("registered posterior contains non-finite values")
        for name, value in (("model_points", points), ("responsibilities", gamma), ("mahalanobis2", m2)):
            copied = value.copy()
            copied.setflags(write=False)
            object.__setattr__(self, name, copied)


def _validated_scan(scan: np.ndarray) -> np.ndarray:
    points = np.asarray(scan, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise RegistrationError("scan must have shape (n, 3)")
    if not np.isfinite(points).all():
        raise RegistrationError("scan contains non-finite coordinates")
    return np.ascontiguousarray(points)


def _scan_digest(scan: np.ndarray) -> str:
    points = _validated_scan(scan)
    digest = hashlib.sha256()
    digest.update(np.asarray(points.shape, dtype="<i8").tobytes())
    digest.update(np.asarray(points, dtype="<f8").tobytes())
    return digest.hexdigest()


def synthesize_scan(
    scene: GeometryScene,
    n_points: int = 1500,
    noise_sigma: float = 0.01,
    outlier_frac: float = 0.02,
    transform: RigidTransformZ | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Deterministic synthetic laser scan of the scene's solid elements.

    Points sample element box *surfaces* (area-weighted faces) with
    Gaussian sensor noise, plus a uniform outlier fraction.  If
    ``transform`` is given, points are emitted in the scan frame, i.e. the
    INVERSE transform is applied — registration should recover
    ``transform``.
    """
    rng = np.random.default_rng(seed)
    solids = [e for e in scene.elements if e.is_solid]
    areas = []
    for e in solids:
        ex, ey, ez = e.box.extents
        areas.append(2.0 * (ex * ey + ex * ez + ey * ez))
    areas = np.asarray(areas)
    probs = areas / areas.sum()

    n_out = int(round(outlier_frac * n_points))
    n_surf = n_points - n_out
    choices = rng.choice(len(solids), size=n_surf, p=probs)

    points = np.zeros((n_surf, 3), dtype=np.float64)
    for i, idx in enumerate(choices):
        e = solids[idx]
        ex, ey, ez = e.box.extents
        face_areas = np.array([ey * ez, ey * ez, ex * ez, ex * ez, ex * ey, ex * ey])
        face = rng.choice(6, p=face_areas / face_areas.sum())
        u, v = rng.random(), rng.random()
        local = {
            0: (0.0, u * ey, v * ez),
            1: (ex, u * ey, v * ez),
            2: (u * ex, 0.0, v * ez),
            3: (u * ex, ey, v * ez),
            4: (u * ex, v * ey, 0.0),
            5: (u * ex, v * ey, ez),
        }[int(face)]
        R = rot_z(e.box.angle)
        points[i] = np.asarray(e.box.origin) + R @ np.asarray(local)
    points += rng.normal(0.0, noise_sigma, size=points.shape)

    los = np.array([e.aabb()[0] for e in solids]).min(axis=0)
    his = np.array([e.aabb()[1] for e in solids]).max(axis=0)
    outliers = los + rng.random((n_out, 3)) * (his - los)
    scan = np.concatenate([points, outliers])

    if transform is not None:
        # Emit in scan frame: apply the inverse of the model-from-scan map.
        R = rot_z(transform.theta)
        scan = (scan - np.asarray(transform.t)) @ R
    return scan


class ScanRegistrar:
    def __init__(
        self,
        scene: GeometryScene,
        reg_sigma: float = 0.08,
        fine_sigma: float = 0.03,
        outlier_pi: float = 0.05,
        max_iter: int = 30,
        tol: float = 1e-9,
    ):
        scene.check_fresh(scene.world)
        solid_mask = np.isin(
            scene.cloud.element_index,
            [e.row for e in scene.elements if e.is_solid],
        )
        prims = scene.cloud.select(solid_mask)
        self.means = prims.means
        self.base_covs = prims.covs
        self.weights = prims.weights
        self.element_index = prims.element_index
        self.scene_version = scene.version
        self._elements_by_row = {
            element.row: element for element in scene.elements if element.is_solid
        }
        self.outlier_pi = outlier_pi
        self.reg_sigma = reg_sigma
        self.fine_sigma = fine_sigma
        self.max_iter = max_iter
        self.tol = tol
        self._set_sigma(reg_sigma)

    def _set_sigma(self, reg_sigma: float) -> None:
        """(Re)build the smoothed mixture at the given regularization scale."""
        covs = self.base_covs + (reg_sigma**2) * np.eye(3)
        self.inv_covs = np.linalg.inv(covs)
        sign, logdet = np.linalg.slogdet(covs)
        self.log_norm = -0.5 * (3.0 * math.log(2.0 * math.pi) + logdet)
        self.log_w = np.log(self.weights / self.weights.sum())
        los = self.means.min(axis=0) - 1.0
        his = self.means.max(axis=0) + 1.0
        self.log_outlier = math.log(self.outlier_pi) - math.log(
            float(np.prod(his - los))
        )

    # -- likelihood --------------------------------------------------------

    def _log_components(self, model_points: np.ndarray) -> np.ndarray:
        """(M, K) log[(1-pi) w_k N_k(x_m)] — the inlier component logs."""
        d = model_points[:, None, :] - self.means[None, :, :]     # (M, K, 3)
        m2 = np.einsum("mki,kij,mkj->mk", d, self.inv_covs, d)
        return (
            math.log(1.0 - self.outlier_pi)
            + self.log_w[None, :]
            + self.log_norm[None, :]
            - 0.5 * m2
        )

    def nll(self, scan: np.ndarray, T: RigidTransformZ) -> float:
        logs = self._log_components(T.apply(scan))
        top = np.maximum(logs.max(axis=1), self.log_outlier)
        lse = top + np.log(
            np.exp(logs - top[:, None]).sum(axis=1) + np.exp(self.log_outlier - top)
        )
        return float(-lse.mean())

    # -- EM ----------------------------------------------------------------

    def _responsibilities(self, model_points: np.ndarray) -> np.ndarray:
        logs = self._log_components(model_points)
        top = np.maximum(logs.max(axis=1), self.log_outlier)
        denom = np.exp(logs - top[:, None]).sum(axis=1) + np.exp(self.log_outlier - top)
        return np.exp(logs - top[:, None]) / denom[:, None]      # (M, K), inlier only

    def _m_step(
        self, scan: np.ndarray, T: RigidTransformZ, gamma: np.ndarray
    ) -> RigidTransformZ:
        """Closed-form t (GLS) then one Armijo-guarded Gauss-Newton yaw step."""
        # Effective per-point precision and precision-weighted target.
        A = np.einsum("mk,kij->mij", gamma, self.inv_covs)           # (M, 3, 3)
        b = np.einsum("mk,kij,kj->mi", gamma, self.inv_covs, self.means)

        def solve_t(theta: float) -> np.ndarray:
            R = rot_z(theta)
            rx = scan @ R.T                                          # (M, 3)
            lhs = A.sum(axis=0)
            rhs = (b - np.einsum("mij,mj->mi", A, rx)).sum(axis=0)
            return np.linalg.solve(lhs, rhs)

        theta = T.theta
        t = solve_t(theta)

        # Gauss-Newton on theta for the responsibility-weighted quadratic.
        R = rot_z(theta)
        dR = np.array(
            [[-math.sin(theta), -math.cos(theta), 0.0],
             [math.cos(theta), -math.sin(theta), 0.0],
             [0.0, 0.0, 0.0]]
        )
        rx = scan @ R.T + t
        drx = scan @ dR.T                                            # (M, 3)
        resid = np.einsum("mij,mj->mi", A, rx) - b                   # precision-weighted
        grad = float(np.einsum("mi,mi->", drx, resid))
        hess = float(np.einsum("mi,mij,mj->", drx, A, drx))
        if hess <= 0:
            return RigidTransformZ(theta, tuple(t))
        step = -grad / hess

        # Armijo guard on the TRUE NLL (monotone outer iteration).
        base = self.nll(scan, RigidTransformZ(theta, tuple(t)))
        alpha = 1.0
        for _ in range(20):
            cand_theta = theta + alpha * step
            cand_t = solve_t(cand_theta)
            if self.nll(scan, RigidTransformZ(cand_theta, tuple(cand_t))) <= base + 1e-15:
                return RigidTransformZ(cand_theta, tuple(cand_t))
            alpha *= 0.5
        return RigidTransformZ(theta, tuple(t))

    def register_from(
        self, scan: np.ndarray, start: RigidTransformZ, max_iter: int | None = None
    ) -> tuple[RigidTransformZ, float, list[float]]:
        T = start
        trace = [self.nll(scan, T)]
        for _ in range(max_iter if max_iter is not None else self.max_iter):
            gamma = self._responsibilities(T.apply(scan))
            T = self._m_step(scan, T, gamma)
            trace.append(self.nll(scan, T))
            if abs(trace[-2] - trace[-1]) < self.tol:
                break
        return T, trace[-1], trace

    def register(
        self, scan: np.ndarray, n_starts: int = 8, accept_nll: float = 6.0
    ) -> RegistrationResult:
        """Coarse-to-fine multi-start registration.

        Stage A runs a few EM iterations from every yaw start at the coarse
        smoothing scale; stage B refines the best basin to convergence,
        then anneals to the fine scale for the final polish and the
        information matrix.  Fully deterministic; ties break by start index.
        """
        scan = _validated_scan(scan)
        if scan.shape[0] < 10:
            raise RegistrationError("scan has too few points")
        model_centroid = self.means.mean(axis=0)
        scan_centroid = scan.mean(axis=0)

        self._set_sigma(self.reg_sigma)
        best: tuple[int, float, RigidTransformZ] | None = None
        start_nlls: list[float] = []
        for k in range(n_starts):
            theta0 = 2.0 * math.pi * k / n_starts
            R0 = rot_z(theta0)
            t0 = model_centroid - R0 @ scan_centroid
            T0 = RigidTransformZ(theta0, tuple(t0))
            T, final_nll, _ = self.register_from(scan, T0, max_iter=6)
            start_nlls.append(final_nll)
            if best is None or final_nll < best[1] - 1e-12:
                best = (k, final_nll, T)
        assert best is not None

        T, _, trace_coarse = self.register_from(scan, best[2])

        self._set_sigma(self.fine_sigma)
        T, nll, trace_fine = self.register_from(scan, T)
        info = self._information_matrix(scan, T)
        self._set_sigma(self.reg_sigma)  # restore for reproducible reuse

        return RegistrationResult(
            transform=T,
            nll=nll,
            nll_trace=tuple(trace_fine),
            coarse_trace=tuple(trace_coarse),
            start_nlls=tuple(start_nlls),
            info_matrix=info,
            accepted=bool(nll < accept_nll),
            scan_digest=_scan_digest(scan),
            scene_version=self.scene_version,
        )

    def register_ply(
        self, path: str, n_starts: int = 8, accept_nll: float = 6.0
    ) -> RegistrationResult:
        """Register vertices from a standard external PLY artifact.

        This is an adapter boundary, not a dependency on a reconstruction
        engine.  For example, Geometry-Grounded-Gaussian-Splatting exports
        its post-processed extracted mesh as ``recon_post.ply``; the mesh
        vertices are treated as scan points and enter the same deterministic
        GMM likelihood and fit-quality gate as native point clouds.
        """
        from gat.geometry.scan_io import load_ply_points

        return self.register(load_ply_points(path), n_starts, accept_nll)

    def evidence(
        self, scan: np.ndarray, result: RegistrationResult
    ) -> ScanEvidenceReport:
        """Aggregate accepted primitive responsibilities by BIM element.

        The report is descriptive evidence, not a state update.  It is
        bound to the exact registered scan and canonical scene version so a
        transform cannot accidentally be reused for different observations
        or a changed model.  Rejected registrations never produce evidence.
        """
        posterior = self.posterior_at(scan, result)
        points = posterior.model_points
        gamma = posterior.responsibilities
        mahalanobis2 = posterior.mahalanobis2

        rows = tuple(sorted(self._elements_by_row))
        element_gamma = np.stack(
            [gamma[:, self.element_index == row].sum(axis=1) for row in rows],
            axis=1,
        )
        point_inlier_mass = element_gamma.sum(axis=1)
        total_inlier_mass = float(point_inlier_mass.sum())

        evidence_rows: list[ElementScanEvidence] = []
        for column, row in enumerate(rows):
            primitive_mask = self.element_index == row
            primitive_mass = gamma[:, primitive_mask].sum(axis=0)
            effective_points = float(primitive_mass.sum())
            primitive_count = int(primitive_mask.sum())

            if effective_points > 0.0:
                normalized_primitive_mass = primitive_mass / effective_points
                support_diversity = float(
                    1.0
                    / (
                        primitive_count
                        * np.square(normalized_primitive_mass).sum()
                    )
                )
                mean_mahalanobis2 = float(
                    np.sum(gamma[:, primitive_mask] * mahalanobis2[:, primitive_mask])
                    / effective_points
                )
                point_share = np.divide(
                    element_gamma[:, column],
                    point_inlier_mass,
                    out=np.zeros_like(point_inlier_mass),
                    where=point_inlier_mass > 0.0,
                )
                assignment_confidence = float(
                    np.sum(element_gamma[:, column] * point_share)
                    / effective_points
                )
            else:
                support_diversity = 0.0
                mean_mahalanobis2 = 0.0
                assignment_confidence = 0.0

            element = self._elements_by_row[row]
            evidence_rows.append(
                ElementScanEvidence(
                    element_row=row,
                    element_name=element.name,
                    primitive_count=primitive_count,
                    effective_points=effective_points,
                    responsibility_fraction=(
                        effective_points / total_inlier_mass
                        if total_inlier_mass > 0.0
                        else 0.0
                    ),
                    mean_mahalanobis2=mean_mahalanobis2,
                    support_diversity=support_diversity,
                    assignment_confidence=assignment_confidence,
                )
            )

        return ScanEvidenceReport(
            scan_digest=result.scan_digest,
            scene_version=result.scene_version,
            point_count=points.shape[0],
            inlier_effective_points=total_inlier_mass,
            outlier_fraction=1.0 - total_inlier_mass / points.shape[0],
            elements=tuple(evidence_rows),
        )

    def posterior_at(
        self,
        scan: np.ndarray,
        result: RegistrationResult,
        transform: RigidTransformZ | None = None,
    ) -> RegisteredScanPosterior:
        """Return fine responsibilities at ``transform`` after all fit gates.

        Omitting ``transform`` reproduces the fitted registration posterior.
        Supplying a survey/SLAM pose lets downstream adapters use registration
        only for association while keeping pose provenance independent.
        """
        points = _validated_scan(scan)
        if not result.accepted:
            raise RegistrationError(
                "cannot report scan evidence: registration failed the fit gate"
            )
        if result.scene_version != self.scene_version:
            raise RegistrationError(
                "cannot report scan evidence: registration belongs to a different scene"
            )
        if result.scan_digest != _scan_digest(points):
            raise RegistrationError(
                "cannot report scan evidence: scan differs from the registered input"
            )

        pose = result.transform if transform is None else transform
        self._set_sigma(self.fine_sigma)
        try:
            model_points = pose.apply(points)
            gamma = self._responsibilities(model_points)
            delta = model_points[:, None, :] - self.means[None, :, :]
            mahalanobis2 = np.einsum(
                "mki,kij,mkj->mk", delta, self.inv_covs, delta
            )
        finally:
            self._set_sigma(self.reg_sigma)
        return RegisteredScanPosterior(
            model_points=model_points,
            responsibilities=gamma,
            mahalanobis2=mahalanobis2,
            scan_digest=result.scan_digest,
            scene_version=result.scene_version,
        )

    def _information_matrix(self, scan: np.ndarray, T: RigidTransformZ) -> np.ndarray:
        """Gauss-Newton Hessian of the total NLL at the optimum, over
        (theta, tx, ty, tz).  Total (not mean) — information adds over
        points, so more scan coverage means a tighter pose."""
        gamma = self._responsibilities(T.apply(scan))
        A = np.einsum("mk,kij->mij", gamma, self.inv_covs)
        dR = np.array(
            [[-math.sin(T.theta), -math.cos(T.theta), 0.0],
             [math.cos(T.theta), -math.sin(T.theta), 0.0],
             [0.0, 0.0, 0.0]]
        )
        drx = scan @ dR.T                                            # d model-point / d theta
        H = np.zeros((4, 4), dtype=np.float64)
        H[0, 0] = np.einsum("mi,mij,mj->", drx, A, drx)
        Ht_block = A.sum(axis=0)
        H[1:, 1:] = Ht_block
        cross = np.einsum("mij,mj->i", A, drx)
        H[0, 1:] = cross
        H[1:, 0] = cross
        return H

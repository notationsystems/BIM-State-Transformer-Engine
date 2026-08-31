"""Gaussian conditioning on observations — raw space only.

Observations of raw *or* derived quantities are conditioned on the RAW
belief, the only covariance in the system that is full-rank.  A derived
observation reaches raw space through its row of the total Jacobian
``H = G[k, :]`` evaluated at the current mean (one EKF measurement step);
the innovation uses the exact nonlinear prediction ``h(mu)``, never
``H @ mu``.

The update is the batch Joseph stabilized form:

    S  = H Sigma H^T + R          (innovation covariance, factored by Cholesky)
    K  = Sigma H^T S^{-1}         (via Cholesky solves — no explicit inverse)
    mu' = mu + K (y - h(mu))
    Sigma' = (I - K H) Sigma (I - K H)^T + K R K^T

Joseph form is a sum of PSD congruences, hence symmetric PSD by
construction for any gain — unlike the textbook subtractive form, which
drifts indefinite under roundoff.

Exact observations (``noise_sigma = 0``) are legal; a *redundant* set of
exact observations makes ``S`` singular and raises
:class:`~gat.errors.ConditioningError` naming the offending measurement
block.

``condition_linear_exact`` implements the closed-form partitioned-Gaussian
posterior through an independent code path.  It exists as an in-repo
oracle: the test suite asserts agreement with :func:`condition` to 1e-12
on linear cases, so a sign or transpose bug in either implementation is
caught by the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gat.errors import ConditioningError, NumericalError
from gat.gaussian.linalg import chol_psd, chol_solve, symmetrize
from gat.gaussian.state import GaussianState


@dataclass(frozen=True)
class ConditioningRecord:
    """Auditable record of one conditioning update."""

    innovations: tuple[float, ...]
    innovation_variances: tuple[float, ...]
    jitter: float
    max_gain: float


def condition(
    belief: GaussianState,
    H: np.ndarray,
    predicted: np.ndarray,
    observed: np.ndarray,
    noise_variances: np.ndarray,
) -> tuple[GaussianState, ConditioningRecord]:
    """Condition the raw belief on ``k`` simultaneous measurements.

    Args:
        belief: raw-space Gaussian belief (full-rank covariance).
        H: (k, n_raw) linearized observation matrix at the current mean.
        predicted: (k,) exact nonlinear predictions ``h(mu)``.
        observed: (k,) measured values.
        noise_variances: (k,) measurement noise variances; zero is legal
            (exact observation) as long as the joint innovation covariance
            stays positive definite.
    """
    n = len(belief.index)
    H = np.asarray(H, dtype=np.float64)
    k = H.shape[0]
    if H.shape != (k, n):
        raise ConditioningError(f"H has shape {H.shape}, expected ({k}, {n})")
    predicted = np.asarray(predicted, dtype=np.float64).reshape(k)
    observed = np.asarray(observed, dtype=np.float64).reshape(k)
    noise = np.asarray(noise_variances, dtype=np.float64).reshape(k)
    if (noise < 0).any():
        raise ConditioningError("negative measurement noise variance")

    sigma = belief.sigma
    R = np.diag(noise)
    S = symmetrize(H @ sigma @ H.T + R)
    try:
        L, jitter = chol_psd(S)
    except NumericalError as exc:
        raise ConditioningError(
            f"degenerate innovation covariance (redundant exact observations?): {exc}"
        ) from exc

    innovation = observed - predicted
    # K = Sigma H^T S^{-1}  computed as  (S^{-1} (H Sigma))^T via Cholesky solves
    K = chol_solve(L, H @ sigma).T

    mu_post = belief.mu + K @ innovation
    A = np.eye(n) - K @ H
    sigma_post = symmetrize(A @ sigma @ A.T + K @ R @ K.T)

    record = ConditioningRecord(
        innovations=tuple(float(x) for x in innovation),
        innovation_variances=tuple(float(x) for x in np.diag(S)),
        jitter=jitter,
        max_gain=float(np.abs(K).max()) if K.size else 0.0,
    )
    return belief.replace(mu=mu_post, sigma=sigma_post), record


def condition_linear_exact(
    mu: np.ndarray,
    sigma: np.ndarray,
    H: np.ndarray,
    observed: np.ndarray,
    noise_variances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form partitioned-Gaussian posterior (test oracle).

    For the linear model ``y = H x + eps``, computes the exact posterior
    over ``x`` by the textbook formula

        mu'    = mu + Sigma H^T (H Sigma H^T + R)^{-1} (y - H mu)
        Sigma' = Sigma - Sigma H^T (H Sigma H^T + R)^{-1} H Sigma

    using ``np.linalg.solve`` on the innovation block — an intentionally
    different code path from :func:`condition` (no Joseph form, no
    Cholesky ladder), used only by tests.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    R = np.diag(np.asarray(noise_variances, dtype=np.float64))

    S = H @ sigma @ H.T + R
    W = np.linalg.solve(S, H @ sigma)  # S^{-1} H Sigma
    mu_post = mu + W.T @ (observed - H @ mu)
    sigma_post = sigma - sigma @ H.T @ W
    return mu_post, symmetrize(sigma_post)

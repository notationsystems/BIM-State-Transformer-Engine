"""Closed-form Gaussian pair calculus.

Everything here is exact Gaussian algebra with hand-derived gradients:

* ``product_integral`` — ∫ N(x; mu_i, S_i) N(x; mu_j, S_j) dx
  = N(mu_i - mu_j; 0, S_i + S_j): the soft-overlap kernel.
* ``bhattacharyya`` — coefficient and distance between two Gaussians.
* ``mahalanobis2`` — squared Mahalanobis distance under a combined
  covariance.
* ``chi2_sf_3`` — the chi-square(3) survival function in closed form,
  ``S_3(t) = 1 - erf(sqrt(t/2)) + sqrt(2 t / pi) exp(-t/2)`` — reported as
  a *separation significance*, deliberately not labeled a clash
  probability (that is the signed-clearance ``Phi`` score in
  :mod:`gat.geometry.clash`).

Gradients: ``d ln I / d mu_i = -S^{-1} (mu_i - mu_j)`` and
``d ln I / d S = 0.5 (S^{-1} d d^T S^{-1} - S^{-1})`` with
``S = S_i + S_j``, ``d = mu_i - mu_j`` — cross-checked in tests against
dual numbers and central differences.
"""

from __future__ import annotations

import math

import numpy as np

_LOG_2PI = math.log(2.0 * math.pi)


def _batched(mu_i, S_i, mu_j, S_j):
    mu_i = np.atleast_2d(np.asarray(mu_i, dtype=np.float64))
    mu_j = np.atleast_2d(np.asarray(mu_j, dtype=np.float64))
    S_i = np.asarray(S_i, dtype=np.float64)
    S_j = np.asarray(S_j, dtype=np.float64)
    if S_i.ndim == 2:
        S_i = S_i[None]
    if S_j.ndim == 2:
        S_j = S_j[None]
    return mu_i, S_i, mu_j, S_j


def log_product_integral(mu_i, S_i, mu_j, S_j) -> np.ndarray:
    """log ∫ N_i N_j — batched over leading axes."""
    mu_i, S_i, mu_j, S_j = _batched(mu_i, S_i, mu_j, S_j)
    S = S_i + S_j
    d = mu_i - mu_j
    sign, logdet = np.linalg.slogdet(S)
    if (sign <= 0).any():
        raise ValueError("combined covariance is not positive definite")
    solve = np.linalg.solve(S, d[..., None])[..., 0]
    m2 = np.einsum("...i,...i->...", d, solve)
    dim = mu_i.shape[-1]
    return -0.5 * (dim * _LOG_2PI + logdet + m2)


def product_integral(mu_i, S_i, mu_j, S_j) -> np.ndarray:
    return np.exp(log_product_integral(mu_i, S_i, mu_j, S_j))


def product_integral_grad_mu(mu_i, S_i, mu_j, S_j) -> np.ndarray:
    """d(log I)/d(mu_i) = -S^{-1} (mu_i - mu_j) — batched."""
    mu_i, S_i, mu_j, S_j = _batched(mu_i, S_i, mu_j, S_j)
    S = S_i + S_j
    d = mu_i - mu_j
    return -np.linalg.solve(S, d[..., None])[..., 0]


def mahalanobis2(mu_i, S_i, mu_j, S_j) -> np.ndarray:
    """(mu_i - mu_j)^T (S_i + S_j)^{-1} (mu_i - mu_j) — batched."""
    mu_i, S_i, mu_j, S_j = _batched(mu_i, S_i, mu_j, S_j)
    S = S_i + S_j
    d = mu_i - mu_j
    solve = np.linalg.solve(S, d[..., None])[..., 0]
    return np.einsum("...i,...i->...", d, solve)


def bhattacharyya_distance(mu_i, S_i, mu_j, S_j) -> np.ndarray:
    """D_B = m2/8 (under the average covariance) + log-det term."""
    mu_i, S_i, mu_j, S_j = _batched(mu_i, S_i, mu_j, S_j)
    S_bar = 0.5 * (S_i + S_j)
    d = mu_i - mu_j
    solve = np.linalg.solve(S_bar, d[..., None])[..., 0]
    m2 = np.einsum("...i,...i->...", d, solve)
    _, logdet_bar = np.linalg.slogdet(S_bar)
    _, logdet_i = np.linalg.slogdet(S_i)
    _, logdet_j = np.linalg.slogdet(S_j)
    return 0.125 * m2 + 0.5 * (logdet_bar - 0.5 * (logdet_i + logdet_j))


def bhattacharyya_coefficient(mu_i, S_i, mu_j, S_j) -> np.ndarray:
    """BC = exp(-D_B) in (0, 1]; 1 iff the Gaussians coincide."""
    return np.exp(-bhattacharyya_distance(mu_i, S_i, mu_j, S_j))


def chi2_sf_3(t) -> np.ndarray:
    """Survival function of the chi-square distribution with 3 dof.

    Closed form: S_3(t) = 1 - erf(sqrt(t/2)) + sqrt(2 t / pi) exp(-t/2).
    """
    t = np.asarray(t, dtype=np.float64)
    root = np.sqrt(np.clip(t, 0.0, None) / 2.0)
    erf_vec = np.vectorize(math.erf)
    return 1.0 - erf_vec(root) + np.sqrt(2.0 * np.clip(t, 0.0, None) / math.pi) * np.exp(
        -np.clip(t, 0.0, None) / 2.0
    )


def normal_cdf(x) -> np.ndarray:
    """Standard normal CDF via erf (vectorized, closed form)."""
    x = np.asarray(x, dtype=np.float64)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def pairwise_overlap_mass(
    means_a: np.ndarray,
    covs_a: np.ndarray,
    weights_a: np.ndarray,
    means_b: np.ndarray,
    covs_b: np.ndarray,
    weights_b: np.ndarray,
) -> float:
    """Total soft-overlap mass between two weighted Gaussian sets.

    sum_{k,l} w_k w_l ∫ N_k N_l — the continuous analogue of intersection
    volume between the two soft solids (units m^6 · m^-3 = m^3).
    """
    S = covs_a[:, None] + covs_b[None, :]                    # (K, L, 3, 3)
    d = means_a[:, None, :] - means_b[None, :, :]            # (K, L, 3)
    sign, logdet = np.linalg.slogdet(S)
    solve = np.linalg.solve(S, d[..., None])[..., 0]
    m2 = np.einsum("klx,klx->kl", d, solve)
    log_int = -0.5 * (3.0 * _LOG_2PI + logdet + m2)
    return float(np.einsum("k,l,kl->", weights_a, weights_b, np.exp(log_int)))

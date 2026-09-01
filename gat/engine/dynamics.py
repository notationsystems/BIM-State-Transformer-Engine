"""Explicit temporal linear-Gaussian dynamics for architectural belief.

The first temporal contract is deliberately small and exact. For selected raw
variables ``x`` over a declared interval ``dt``::

    x_next = A x + b + w,       w ~ N(0, Q)

Untargeted raw variables follow identity dynamics. Cross-covariances are
transported by the same full transition matrix and ``Q`` is injected only into
the selected block. The result is an ordinary first-class GAT transformation:
derived state is rebuilt, invariants run, the session commits or rolls back,
and the closed ledger can replay it.

This is a calibrated process model, not a learned dynamics system and not a
general FEP agent. Model identity and calibration digest are mandatory so
process noise cannot enter belief anonymously.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

import numpy as np

from gat.engine.binding import GaussianBinding
from gat.engine.executor import ExecutionResult, World, execute
from gat.engine.transform import Transformation, _require_raw
from gat.engine.verify import VerificationReport
from gat.errors import NumericalError
from gat.gaussian.linalg import chol_psd, max_asymmetry, symmetrize
from gat.gaussian.state import GaussianState
from gat.ids import VarId


_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")


class EvolveLinearGaussian(Transformation):
    """One exact temporal process transition over selected raw variables."""

    name = "evolve_linear_gaussian"

    def __init__(
        self,
        targets: tuple[VarId, ...],
        transition: np.ndarray,
        offset: np.ndarray,
        process_covariance: np.ndarray,
        elapsed_seconds: float,
        model_id: str,
        calibration_digest: str,
    ):
        targets = tuple(targets)
        if not targets or len(set(targets)) != len(targets):
            raise ValueError("process targets must be non-empty and unique")
        n = len(targets)
        transition = np.asarray(transition, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64).reshape(-1)
        process_covariance = np.asarray(process_covariance, dtype=np.float64)
        if transition.shape != (n, n):
            raise ValueError(f"transition must have shape {(n, n)}")
        if offset.shape != (n,):
            raise ValueError(f"offset must have shape {(n,)}")
        if process_covariance.shape != (n, n):
            raise ValueError(f"process_covariance must have shape {(n, n)}")
        if not (
            np.isfinite(transition).all()
            and np.isfinite(offset).all()
            and np.isfinite(process_covariance).all()
        ):
            raise ValueError("process matrices and offset must be finite")
        scale = max(float(np.abs(process_covariance).max(initial=0.0)), 1.0)
        if max_asymmetry(process_covariance) > 1e-12 * scale:
            raise ValueError("process_covariance must be symmetric")
        process_covariance = symmetrize(process_covariance)
        try:
            _, jitter = chol_psd(process_covariance)
        except NumericalError as exc:
            raise ValueError(f"process_covariance must be PSD: {exc}") from exc
        # A negative direction that is only hidden by the certification jitter
        # is accepted under the same 1e-12 numerical regime as engine beliefs.
        self.psd_jitter = jitter
        if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
            raise ValueError("elapsed_seconds must be finite and positive")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("model_id must be a non-empty string")
        if (
            not isinstance(calibration_digest, str)
            or _DIGEST_RE.fullmatch(calibration_digest) is None
        ):
            raise ValueError("calibration_digest must be a lowercase SHA-256 digest")
        self.targets = targets
        self.transition = transition.copy()
        self.offset = offset.copy()
        self.process_covariance = process_covariance.copy()
        for array in (self.transition, self.offset, self.process_covariance):
            array.setflags(write=False)
        self.elapsed_seconds = float(elapsed_seconds)
        self.model_id = model_id
        self.calibration_digest = calibration_digest

    def params(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "elapsed_seconds": self.elapsed_seconds,
            "targets": len(self.targets),
            "calibration": self.calibration_digest,
        }

    def target_vars(self) -> tuple[VarId, ...]:
        return self.targets

    def full_transition(self, binding: GaussianBinding) -> np.ndarray:
        rows = np.array(
            [_require_raw(binding, var, self.name) for var in self.targets],
            dtype=np.intp,
        )
        full = np.eye(binding.n_raw)
        full[np.ix_(rows, rows)] = self.transition
        return full

    def apply(self, binding: GaussianBinding, belief: GaussianState) -> GaussianState:
        rows = np.array(
            [_require_raw(binding, var, self.name) for var in self.targets],
            dtype=np.intp,
        )
        F = np.eye(binding.n_raw)
        F[np.ix_(rows, rows)] = self.transition
        mu = F @ belief.mu
        mu[rows] += self.offset
        sigma = F @ belief.sigma @ F.T
        sigma[np.ix_(rows, rows)] += self.process_covariance
        return GaussianState(belief.index, mu, sigma)


@dataclass(frozen=True)
class ProcessForecastStep:
    index: int
    elapsed_seconds: float
    cumulative_seconds: float
    prior_world_digest: str
    result_world_digest: str
    verification: VerificationReport


@dataclass(frozen=True)
class ProcessRollout:
    initial_world_digest: str
    final_world: World
    steps: tuple[ProcessForecastStep, ...]

    @property
    def elapsed_seconds(self) -> float:
        return self.steps[-1].cumulative_seconds if self.steps else 0.0


def forecast_process(
    world: World,
    process: EvolveLinearGaussian,
    *,
    steps: int = 1,
) -> ProcessRollout:
    """Roll a calibrated process forward without mutating a session."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise ValueError("steps must be a positive integer")
    current = world
    records: list[ProcessForecastStep] = []
    cumulative = 0.0
    for index in range(steps):
        before = current
        result: ExecutionResult = execute(before, process, strict=True)
        cumulative += process.elapsed_seconds
        records.append(
            ProcessForecastStep(
                index=index,
                elapsed_seconds=process.elapsed_seconds,
                cumulative_seconds=cumulative,
                prior_world_digest=before.digest(),
                result_world_digest=result.world.digest(),
                verification=result.report,
            )
        )
        current = result.world
    return ProcessRollout(world.digest(), current, tuple(records))


__all__ = [
    "EvolveLinearGaussian",
    "ProcessForecastStep",
    "ProcessRollout",
    "forecast_process",
]

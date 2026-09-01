"""One-step active inference for uncertainty-aware BIM observations.

GAT already distinguishes *observations* from *interventions*: an
``ObserveQuantity`` conditions the Gaussian architectural belief, while a
``SetParameter`` changes it.  This module adds the policy that comes before
an observation: given a decision-relevant quantity, which available
measurement should be taken next?

The implementation deliberately uses a small, inspectable, linear-Gaussian
slice of active inference rather than claiming to be a general Free Energy
Principle simulator.  Around the current belief, a candidate sensor obeys

    y = h(mu) + H (x - mu) + eps,    eps ~ N(0, r)

where ``x`` is GAT's canonical raw belief.  For a preferred target ``t`` the
planner computes the one-step epistemic value

    I(t ; y) = 1/2 log(V_t / V_t|y)

and a pragmatic risk ``-log P(t >= minimum)``.  Its reported expected-free-
energy proxy is ``risk + action_cost - I``: lower is preferred.  Action cost
is expressed in nats so survey time, money, access, or safety burden must be
calibrated onto the same prior-surprise scale.  As an observation does not
itself change the building, expected risk is identical across passive
candidates before an outcome is known; the policy trades target-relevant
information against measurement burden.  A subsequent real measurement is
committed through the existing ``ObserveQuantity`` -> propagate -> verify
pipeline.

All calculations are deterministic and operate in full-rank raw space.  The
first-order validity limits are the same as GAT's normal derived-observation
conditioning.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from gat.engine.executor import World
from gat.engine.propagate import jacobian_rows
from gat.engine.transform import ObserveQuantity
from gat.ids import VarId


_MIN_VARIANCE = 1e-300


@dataclass(frozen=True)
class ObservationCandidate:
    """An available scalar sensing action.

    ``noise_sigma`` is the calibrated standard deviation of the measurement
    device or procedure in the variable's declared unit.  Exact observations
    are supported by :class:`ObserveQuantity`, but do not have a finite
    differential-entropy information score and are therefore rejected here.
    ``cost_nats`` is the calibrated prior-surprise burden of taking the
    measurement, not a raw currency or duration.
    """

    var: VarId
    noise_sigma: float
    label: str = ""
    cost_nats: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.noise_sigma) or self.noise_sigma <= 0.0:
            raise ValueError("observation candidate noise_sigma must be finite and positive")
        if not math.isfinite(self.cost_nats) or self.cost_nats < 0.0:
            raise ValueError("observation candidate cost_nats must be finite and non-negative")

    @property
    def name(self) -> str:
        return self.label or str(self.var)

    def observe(self, value: float) -> ObserveQuantity:
        """Turn the selected action and an acquired value into GAT evidence."""
        return ObserveQuantity.single(self.var, value, self.noise_sigma)


@dataclass(frozen=True)
class MinimumPreference:
    """A decision preference expressed as ``target >= minimum``.

    This mirrors GAT's existing compliance-margin convention.  It is an
    *evaluation preference*, not an intervention: choosing to observe cannot
    improve the building state by itself.
    """

    target: VarId
    minimum: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum):
            raise ValueError("preference minimum must be finite")


@dataclass(frozen=True)
class ObservationPlan:
    """Auditable score for one candidate observation.

    ``epistemic_value`` and ``action_cost`` are in nats.  ``pragmatic_risk``
    is the negative log prior probability of satisfying the supplied minimum
    preference.  The latter is ``None`` when planning general state
    information without a decision target.
    """

    candidate: ObservationCandidate
    predicted_measurement: float
    measurement_sigma: float
    epistemic_value: float
    action_cost: float
    expected_free_energy: float
    target: VarId | None = None
    target_mean: float | None = None
    target_sigma: float | None = None
    posterior_target_sigma: float | None = None
    p_satisfies_preference: float | None = None
    pragmatic_risk: float | None = None

    @property
    def net_epistemic_value(self) -> float:
        """Expected information gain after the action's declared burden."""
        return self.epistemic_value - self.action_cost

    def describe(self) -> str:
        target = "all raw state" if self.target is None else str(self.target)
        return (
            f"observe {self.candidate.name}: target {target}; "
            f"epistemic {self.epistemic_value:.6f} nat; "
            f"cost {self.action_cost:.6f} nat; "
            f"EFE {self.expected_free_energy:.6f}"
        )


def plan_observations(
    world: World,
    candidates: Iterable[ObservationCandidate],
    preference: MinimumPreference | None = None,
) -> tuple[ObservationPlan, ...]:
    """Score candidate observations from lowest expected-free-energy first.

    With a preference, epistemic value is the mutual information between the
    observation and the preferred target.  Without one, it is the mutual
    information between the observation and all raw state.  Ties are sorted
    by stable architectural identity and calibrated sensor noise.
    """
    candidates = tuple(candidates)
    if not candidates:
        raise ValueError("at least one observation candidate is required")

    target_row: np.ndarray | None = None
    target_mean: float | None = None
    target_var: float | None = None
    p_satisfies: float | None = None
    risk: float | None = None

    if preference is not None:
        target_h, target_predicted = jacobian_rows(
            world.binding, world.belief, (preference.target,)
        )
        target_row = target_h[0]
        target_mean = float(target_predicted[0])
        target_var = float(target_row @ world.belief.sigma @ target_row)
        target_var = max(target_var, 0.0)
        target_sigma = math.sqrt(target_var)
        if target_sigma <= 1e-15:
            p_satisfies = 1.0 if target_mean >= preference.minimum else 0.0
        else:
            p_satisfies = _normal_cdf((target_mean - preference.minimum) / target_sigma)
        risk = -math.log(max(p_satisfies, _MIN_VARIANCE))

    plans: list[ObservationPlan] = []
    for candidate in candidates:
        H, predicted = jacobian_rows(world.binding, world.belief, (candidate.var,))
        row = H[0]
        noise_var = candidate.noise_sigma**2
        measurement_var = float(row @ world.belief.sigma @ row) + noise_var
        if measurement_var <= 0.0:
            # Positive noise makes this unreachable for valid candidates, but
            # keeping the check near the calculation makes the invariant
            # explicit if a future sensor model changes the assumptions.
            raise ValueError(f"non-positive innovation variance for {candidate.var}")

        if target_row is None:
            # I(x ; y) = 1/2 log(|H Sigma H^T + R| / |R|) for one scalar y.
            epistemic = 0.5 * math.log(measurement_var / noise_var)
            plans.append(
                ObservationPlan(
                    candidate=candidate,
                    predicted_measurement=float(predicted[0]),
                    measurement_sigma=math.sqrt(measurement_var),
                    epistemic_value=epistemic,
                    action_cost=candidate.cost_nats,
                    expected_free_energy=candidate.cost_nats - epistemic,
                )
            )
            continue

        assert target_var is not None
        cov_target_measurement = float(target_row @ world.belief.sigma @ row)
        if target_var <= _MIN_VARIANCE:
            epistemic = 0.0
            posterior_target_var = target_var
        else:
            posterior_target_var = target_var - (
                cov_target_measurement**2 / measurement_var
            )
            # A finite-noise observation of a positive-variance target has a
            # positive posterior variance.  The clamp only absorbs float64
            # round-off and keeps the log calculation well-defined.
            posterior_target_var = max(posterior_target_var, _MIN_VARIANCE)
            epistemic = 0.5 * math.log(target_var / posterior_target_var)

        assert target_mean is not None and p_satisfies is not None and risk is not None
        plans.append(
            ObservationPlan(
                candidate=candidate,
                predicted_measurement=float(predicted[0]),
                measurement_sigma=math.sqrt(measurement_var),
                epistemic_value=epistemic,
                action_cost=candidate.cost_nats,
                expected_free_energy=risk + candidate.cost_nats - epistemic,
                target=preference.target,
                target_mean=target_mean,
                target_sigma=math.sqrt(target_var),
                posterior_target_sigma=math.sqrt(posterior_target_var),
                p_satisfies_preference=p_satisfies,
                pragmatic_risk=risk,
            )
        )

    return tuple(
        sorted(
            plans,
            key=lambda p: (
                p.expected_free_energy,
                str(p.candidate.var),
                p.candidate.noise_sigma,
                p.candidate.label,
            ),
        )
    )


def select_observation(
    world: World,
    candidates: Iterable[ObservationCandidate],
    preference: MinimumPreference | None = None,
) -> ObservationPlan:
    """Return the deterministic minimum-EFE observation plan."""
    return plan_observations(world, candidates, preference)[0]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

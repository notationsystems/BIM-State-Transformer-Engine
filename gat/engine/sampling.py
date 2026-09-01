"""Sampling the architectural belief: realizations, calibration, variation.

The belief ``N(mu_raw, Sigma_raw)`` is a distribution over plausible
buildings.  Sampling it yields *realizations* — concrete as-builts whose
imperfections are correlated exactly as the model says (walls sharing a
storey height move together).  Three uses:

* **Variation** — each sample is a physically consistent building for
  downstream art/visualization pipelines (see
  :mod:`gat.geometry.variations`).
* **Realization checking** — running the invariant registry on samples
  measures how often the *modeled* uncertainty admits states that violate
  hard constraints, which the analytic layer only bounds.
* **Calibration** — Monte-Carlo estimates of clearance probabilities are
  an independent witness for the delta-method ``P(clash)`` scores: the
  analytic number is a linearization, the empirical number is the ground
  truth of the model, and the tests assert they agree within Monte-Carlo
  error.

Everything is seeded and deterministic: same world + same seed = the same
samples, bit for bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gat.engine.executor import World
from gat.engine.verify import Status, run_invariants
from gat.gaussian.linalg import chol_psd
from gat.geometry.gaussianize import OrientedBox, rot_z


def sample_raw(world: World, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` raw-vector samples from the belief — (n, n_raw), seeded."""
    if n < 1:
        raise ValueError("need at least one sample")
    L, _ = chol_psd(world.belief.sigma)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, world.binding.n_raw))
    return world.belief.mu[None, :] + z @ L.T


def sample_worlds(world: World, n: int, seed: int) -> tuple[World, ...]:
    """Realized worlds: sampled means, original covariance, full pushforward."""
    samples = sample_raw(world, n, seed)
    out = []
    for i in range(n):
        belief = world.belief.replace(mu=samples[i])
        out.append(world.with_belief(belief))
    return tuple(out)


@dataclass(frozen=True)
class SampleRealization:
    index: int
    passed: bool
    failures: tuple[str, ...]   # "INV-ID [subject]" strings


@dataclass(frozen=True)
class SampleReport:
    seed: int
    realizations: tuple[SampleRealization, ...]
    violation_rates: tuple[tuple[str, float], ...]  # sorted, only nonzero

    @property
    def n(self) -> int:
        return len(self.realizations)

    @property
    def pass_rate(self) -> float:
        if not self.realizations:
            return 1.0
        return sum(1 for r in self.realizations if r.passed) / len(self.realizations)

    def render(self) -> str:
        lines = [
            f"sampled realizations: {self.n} (seed {self.seed}); "
            f"pass rate {self.pass_rate:.3f}"
        ]
        for key, rate in self.violation_rates:
            lines.append(f"  {rate:6.3f}  {key}")
        return "\n".join(lines)


def sample_report(world: World, n: int, seed: int) -> SampleReport:
    """Run the invariant registry on ``n`` realizations of the belief."""
    realizations: list[SampleRealization] = []
    counts: dict[str, int] = {}
    for i, sampled in enumerate(sample_worlds(world, n, seed)):
        report = run_invariants(sampled)
        failures = tuple(
            f"{r.invariant_id} [{r.subject}]"
            for r in report.results
            if r.status is Status.FAIL
        )
        for key in failures:
            counts[key] = counts.get(key, 0) + 1
        realizations.append(SampleRealization(i, not failures, failures))
    rates = tuple(
        sorted(((key, c / n) for key, c in counts.items()), key=lambda kv: (-kv[1], kv[0]))
    )
    return SampleReport(seed, tuple(realizations), rates)


# ---------------------------------------------------------------------------
# Monte-Carlo clearance calibration
# ---------------------------------------------------------------------------


def sat_clearance(box_a: OrientedBox, box_b: OrientedBox) -> float:
    """Separating-axis clearance between two z-aligned oriented boxes.

    ``max_u (|u . delta| - r_a(u) - r_b(u))`` over the vertical axis and
    both boxes' local in-plane axes — the same clearance definition the
    clash layer scores, on bare boxes.
    """
    delta = box_b.center() - box_a.center()
    axes = [np.array([0.0, 0.0, 1.0])]
    for box in (box_a, box_b):
        R = rot_z(box.angle)
        axes.append(R[:, 0])
        axes.append(R[:, 1])
    best = -math.inf
    for u in axes:
        proj = abs(float(u @ delta))
        r_a = _support(box_a, u)
        r_b = _support(box_b, u)
        best = max(best, proj - r_a - r_b)
    return best


def _support(box: OrientedBox, direction: np.ndarray) -> float:
    R = rot_z(box.angle)
    h = 0.5 * np.asarray(box.extents)
    return float(np.abs(direction @ R) @ h)


@dataclass(frozen=True)
class EmpiricalClearance:
    element_a: str
    element_b: str
    n: int
    seed: int
    mean: float
    std: float
    p_clash: float                 # P(clearance < -penetration_tol), empirical
    mc_standard_error: float       # binomial SE of p_clash

    def render(self) -> str:
        return (
            f"{self.element_a} x {self.element_b}: empirical clearance "
            f"{self.mean:+.4f} +- {self.std:.4f} m over {self.n} realizations; "
            f"P(clash) {self.p_clash:.4f} (+- {self.mc_standard_error:.4f} MC)"
        )


def empirical_pair_clearance(
    world: World,
    name_a: str,
    name_b: str,
    n: int = 4000,
    seed: int = 0,
    penetration_tol: float = 0.01,
) -> EmpiricalClearance:
    """Monte-Carlo clearance between two scene elements under the belief.

    Element boxes are rebuilt per realization from the sampled extents
    (placements are deterministic in v0), with derived extents (e.g. wall
    heights bound to the storey) re-evaluated through the dependency DAG —
    so shared parameters stay correlated across the pair, exactly as in
    the analytic relative-Jacobian score this function witnesses.
    """
    from gat.geometry.stateio import derive_scene

    scene = derive_scene(world)
    elem_a = scene.element_by_name(name_a)
    elem_b = scene.element_by_name(name_b)

    deps = world.binding.deps
    raw_vars = world.binding.raw_index.vars
    samples = sample_raw(world, n, seed)

    clearances = np.empty(n, dtype=np.float64)
    for i in range(n):
        env = {v: float(samples[i, k]) for k, v in enumerate(raw_vars)}
        derived = deps.evaluate(env)
        env.update(derived)

        boxes = []
        for element in (elem_a, elem_b):
            extents = tuple(
                env[var] if var is not None else const
                for var, const in zip(
                    element.extent_vars, element.box.extents
                )
            )
            boxes.append(
                OrientedBox(element.box.origin, element.box.angle, extents)
            )
        clearances[i] = sat_clearance(boxes[0], boxes[1])

    p = float(np.mean(clearances < -penetration_tol))
    se = math.sqrt(max(p * (1.0 - p), 1e-12) / n)
    return EmpiricalClearance(
        element_a=name_a,
        element_b=name_b,
        n=n,
        seed=seed,
        mean=float(clearances.mean()),
        std=float(clearances.std(ddof=1)) if n > 1 else 0.0,
        p_clash=p,
        mc_standard_error=se,
    )

"""Probabilistic clash detection over the Gaussian scene.

Replaces boolean intersection tests with calibrated probabilistic scoring
under geometric uncertainty.  For each candidate pair the detector reports:

* ``clearance`` — the signed first-order clearance along the center axis:
  ``c = ||p_b - p_a|| - r_a(u) - r_b(u)`` with ``r`` the box support radius
  along the pair direction ``u`` (negative = interpenetrating).
* ``sigma`` — the delta-method standard deviation of that clearance under
  the *joint* raw belief, using the relative Jacobian
  ``g = u (J_b - J_a) - dr_a/dx - dr_b/dx`` so that shared parameters
  cancel exactly (correlation-aware; per-element inflation would not).
* ``p_clash = Phi(-c / sigma)`` — the probability that the true clearance
  is negative: a probability of a real event, the headline score.
* ``overlap_mass`` — the soft intersection volume between the two Gaussian
  sets (severity of an actual interpenetration).
* ``separation_sig`` — chi-square(3) survival of the minimum primitive
  Mahalanobis separation.  This is a *significance*, deliberately not
  called a probability.

Pairs that the relationship graph explains (a door in its host wall, a
space against its bounding walls) are exempt: expected contact, not clash.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gat.geometry.overlap import (
    chi2_sf_3,
    mahalanobis2,
    normal_cdf,
    pairwise_overlap_mass,
)
from gat.geometry.stateio import GeometryScene, SceneElement, rot_z, support_radius


@dataclass(frozen=True)
class ClashItem:
    element_a: str
    element_b: str
    clearance: float
    sigma: float
    p_clash: float
    overlap_mass: float
    separation_sig: float
    witness: tuple[float, float, float]

    def render(self) -> str:
        return (
            f"{self.element_a:<12} x {self.element_b:<12} "
            f"clearance {self.clearance:+.4f} +- {self.sigma:.4f} m   "
            f"P(clash) {self.p_clash:6.4f}   overlap {self.overlap_mass:.5f} m3   "
            f"sep-sig {self.separation_sig:.3g}"
        )


@dataclass(frozen=True)
class ClashReport:
    items: tuple[ClashItem, ...]
    n_pairs_considered: int
    n_pairs_broadphase: int

    def worst(self) -> ClashItem | None:
        return self.items[0] if self.items else None

    def render(self) -> str:
        lines = [
            f"clash report: {len(self.items)} scored pairs "
            f"({self.n_pairs_broadphase} past broad phase of {self.n_pairs_considered})"
        ]
        lines.extend(item.render() for item in self.items)
        return "\n".join(lines)


def _clearance_gradient(
    scene: GeometryScene,
    a: SceneElement,
    b: SceneElement,
    direction: np.ndarray,
) -> np.ndarray:
    """d(clearance)/d(raw) treating the pair direction as fixed (first order)."""
    g = direction @ (
        scene.center_jacobian_wrt_raw(b) - scene.center_jacobian_wrt_raw(a)
    )
    for element, sign in ((a, -1.0), (b, -1.0)):
        R = rot_z(element.box.angle)
        proj = np.abs(direction @ R)  # |u . R e_i| per local axis
        g = g + sign * 0.5 * proj @ scene.extent_jacobians[element.row]
    return g


def _separation_axes(a: SceneElement, b: SceneElement) -> np.ndarray:
    """Candidate separating axes for two z-aligned oriented boxes: both
    boxes' local in-plane axes plus the vertical (deduplicated)."""
    axes = [np.array([0.0, 0.0, 1.0])]
    for element in (a, b):
        R = rot_z(element.box.angle)
        axes.append(R[:, 0])
        axes.append(R[:, 1])
    out: list[np.ndarray] = []
    for u in axes:
        if not any(abs(abs(u @ v) - 1.0) < 1e-9 for v in out):
            out.append(u)
    return np.asarray(out)


def score_pair(
    scene: GeometryScene,
    a: SceneElement,
    b: SceneElement,
    extra_var: float = 0.0,
    penetration_tol: float = 0.01,
) -> ClashItem:
    """Score one element pair; ``extra_var`` adds exogenous position variance
    (e.g. an ad-hoc proposed element with its own placement uncertainty).

    Clearance is the separating-axis form ``max_u (|u . delta| - r_a - r_b)``
    over the boxes' local axes — exact separation for the axis-parallel
    cases that dominate buildings, and a conservative bound otherwise.
    ``p_clash`` is the probability of penetrating deeper than
    ``penetration_tol`` (corner *contact* between joined walls is expected
    construction adjacency, not clash).
    """
    center_a = a.box.center()
    center_b = b.box.center()
    delta = center_b - center_a

    best_clearance = -np.inf
    direction = np.array([1.0, 0.0, 0.0])
    for u in _separation_axes(a, b):
        proj = float(u @ delta)
        axis = u if proj >= 0 else -u
        c_u = abs(proj) - support_radius(a, axis) - support_radius(b, axis)
        if c_u > best_clearance:
            best_clearance = c_u
            direction = axis
    clearance = float(best_clearance)

    g = _clearance_gradient(scene, a, b, direction)
    var = float(g @ scene.world.belief.sigma @ g) + extra_var
    sigma = float(np.sqrt(max(var, 0.0)))

    if sigma > 1e-12:
        p_clash = float(normal_cdf(-(clearance + penetration_tol) / sigma))
    else:
        p_clash = 1.0 if clearance < -penetration_tol else 0.0

    prims_a = scene.cloud.of_element(a.row)
    prims_b = scene.cloud.of_element(b.row)
    overlap = pairwise_overlap_mass(
        prims_a.means, prims_a.covs, prims_a.weights,
        prims_b.means, prims_b.covs, prims_b.weights,
    )
    m2 = mahalanobis2(
        prims_a.means[:, None, :], prims_a.covs[:, None, :, :],
        prims_b.means[None, :, :], prims_b.covs[None, :, :, :],
    )
    k, l = np.unravel_index(int(np.argmin(m2)), m2.shape)
    sep_sig = float(chi2_sf_3(float(m2[k, l])))
    witness = 0.5 * (prims_a.means[k] + prims_b.means[l])

    return ClashItem(
        element_a=a.name,
        element_b=b.name,
        clearance=clearance,
        sigma=sigma,
        p_clash=p_clash,
        overlap_mass=overlap,
        separation_sig=sep_sig,
        witness=(float(witness[0]), float(witness[1]), float(witness[2])),
    )


def detect(
    scene: GeometryScene,
    max_clearance: float = 0.5,
    min_p: float = 0.0,
) -> ClashReport:
    """Score every non-exempt solid pair within broad-phase range."""
    scene.check_fresh(scene.world)
    solids = [e for e in scene.elements if e.is_solid]
    considered = 0
    passed: list[tuple[SceneElement, SceneElement]] = []
    for i, a in enumerate(solids):
        lo_a, hi_a = a.aabb(margin=0.5 * max_clearance)
        for b in solids[i + 1 :]:
            considered += 1
            pair = (min(a.row, b.row), max(a.row, b.row))
            if pair in scene.exempt_pairs:
                continue
            lo_b, hi_b = b.aabb(margin=0.5 * max_clearance)
            if (lo_a > hi_b).any() or (lo_b > hi_a).any():
                continue
            passed.append((a, b))

    items = [score_pair(scene, a, b) for a, b in passed]
    items = [it for it in items if it.p_clash >= min_p]
    items.sort(key=lambda it: (-it.p_clash, -it.overlap_mass, it.element_a, it.element_b))
    return ClashReport(tuple(items), considered, len(passed))


def score_proposed_box(
    scene: GeometryScene,
    box,
    position_sigma: float = 0.0,
    max_clearance: float = 0.5,
) -> ClashReport:
    """Score an ad-hoc proposed element (not in the state) against the scene.

    The proposed box is deterministic geometry plus optional isotropic
    placement uncertainty ``position_sigma`` — the classic coordination
    question: *may this duct/beam/fixture go here?*
    """
    from gat.geometry.gaussianize import gaussianize_box
    from gat.geometry.primitives import GaussianCloud, N_FEATURES

    scene.check_fresh(scene.world)
    means, covs, weights, fractions = gaussianize_box(box, 0.4)
    n_raw = scene.world.binding.n_raw
    base_count = len(scene.cloud)
    ghost = SceneElement(
        row=len(scene.elements),
        entity_id=None,  # type: ignore[arg-type]
        name="<proposed>",
        box=box,
        extent_vars=(None, None, None),
        prim_start=base_count,
        prim_end=base_count + means.shape[0],
        is_solid=True,
    )
    ghost_cloud = GaussianCloud(
        means, covs, weights,
        np.zeros((means.shape[0], N_FEATURES)),
        np.full(means.shape[0], ghost.row, dtype=np.intp),
        scene.version,
    )
    ghost_cloud.extras["fractions"] = fractions

    # Temporarily extend the scene views the scorer reads.
    extended = GeometryScene(
        world=scene.world,
        elements=scene.elements + (ghost,),
        cloud=_concat_with(scene.cloud, ghost_cloud),
        extent_jacobians=np.concatenate(
            [scene.extent_jacobians, np.zeros((1, 3, n_raw))]
        ),
        version=scene.version,
        exempt_pairs=scene.exempt_pairs,
    )
    items = []
    considered = 0
    lo_g, hi_g = ghost.aabb(margin=0.5 * max_clearance)
    for element in extended.elements[:-1]:
        if not element.is_solid:
            continue
        considered += 1
        lo, hi = element.aabb(margin=0.5 * max_clearance)
        if (lo > hi_g).any() or (lo_g > hi).any():
            continue
        items.append(
            score_pair(extended, element, ghost, extra_var=position_sigma**2)
        )
    items.sort(key=lambda it: (-it.p_clash, -it.overlap_mass, it.element_a))
    return ClashReport(tuple(items), considered, len(items))


def _concat_with(cloud, ghost):
    from gat.geometry.primitives import GaussianCloud

    merged = GaussianCloud(
        np.concatenate([cloud.means, ghost.means]),
        np.concatenate([cloud.covs, ghost.covs]),
        np.concatenate([cloud.weights, ghost.weights]),
        np.concatenate([cloud.features, ghost.features]),
        np.concatenate([cloud.element_index, ghost.element_index]),
        cloud.version,
    )
    merged.extras["fractions"] = np.concatenate(
        [cloud.extras["fractions"], ghost.extras["fractions"]]
    )
    return merged

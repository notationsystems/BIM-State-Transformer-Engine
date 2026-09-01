"""The bridge between the canonical state and the geometric Gaussian layer.

The cloud is a *derived intermediate representation*: it is lowered from
the world (placements + dimensional parameters), carries the world digest
as its version, and every query that couples to parameter uncertainty
chains through the extent Jacobians back into the raw belief.  State is
primary; geometry is a view.  A scene whose version does not match the
current world digest is stale — using it is a hard error, never a silent
approximation.

Uncertainty coupling: each element's box extents are state variables (raw
or derived), so ``d extents / d raw = H_extent`` comes from the engine's
Jacobian rows.  For a *pair* of elements the position uncertainty of the
gap between them uses the relative Jacobian

    Sigma_rel = (J_a - J_b) Sigma_raw (J_a - J_b)^T

which correctly cancels shared parameters (two elements driven by the same
storey height do not jitter relative to each other in z) — per-element
inflation would silently drop those cross terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from gat.engine.executor import World
from gat.engine.propagate import jacobian_rows
from gat.errors import GatError
from gat.geometry.gaussianize import (
    OrientedBox,
    gaussianize_box,
    mean_jacobian_wrt_extents,
    rot_z,
)
from gat.geometry.primitives import (
    CLASS_CHANNEL,
    FEATURE_NAMES,
    N_FEATURES,
    GaussianCloud,
)
from gat.ids import EntityId, VarId
from gat.ir.core import RelKind

#: Constant leaf thickness for doors (not a state variable in v0).
DOOR_LEAF_THICKNESS = 0.05

#: Default thermal transmittance by (class, external?) [W/m2K].
DEFAULT_U = {
    ("IfcWall", True): 0.25,
    ("IfcWall", False): 0.80,
    ("IfcDoor", True): 1.80,
    ("IfcDoor", False): 1.80,
    ("IfcSpace", True): 0.0,
    ("IfcSpace", False): 0.0,
}


@dataclass(frozen=True)
class SceneElement:
    row: int                     # element row in the scene table
    entity_id: EntityId
    name: str
    box: OrientedBox
    extent_vars: tuple[VarId | None, VarId | None, VarId | None]
    prim_start: int              # slice into the cloud arrays
    prim_end: int
    is_solid: bool               # participates in clash detection

    def aabb(self, margin: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
        """World-frame axis-aligned bounds of the box, padded by margin."""
        R = rot_z(self.box.angle)
        E = np.asarray(self.box.extents)
        corners_local = (
            np.array(
                [[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)],
                dtype=np.float64,
            )
            * E
        )
        corners = np.asarray(self.box.origin) + corners_local @ R.T
        return corners.min(axis=0) - margin, corners.max(axis=0) + margin


@dataclass
class GeometryScene:
    world: World
    elements: tuple[SceneElement, ...]
    cloud: GaussianCloud
    extent_jacobians: np.ndarray   # (n_elements, 3, n_raw): d extents / d raw
    version: str
    exempt_pairs: frozenset[tuple[int, int]]
    element_affinity: np.ndarray = None  # type: ignore[assignment]  # (n_elem, n_elem)

    def element_by_name(self, name: str) -> SceneElement:
        matches = [e for e in self.elements if e.name == name]
        if len(matches) != 1:
            raise KeyError(f"{len(matches)} scene elements named {name!r}")
        return matches[0]

    def check_fresh(self, world: World) -> None:
        if self.version != world.digest():
            raise GatError(
                "stale geometry scene: the world has changed since this scene "
                "was derived — re-derive before querying"
            )

    def mean_jacobian_wrt_raw(self, element: SceneElement) -> np.ndarray:
        """d(primitive means)/d(raw) for one element — (K, 3, n_raw)."""
        fractions = self.cloud.extras["fractions"][
            element.prim_start : element.prim_end
        ]
        J_mean_ext = mean_jacobian_wrt_extents(element.box, fractions)  # (K,3,3)
        return np.einsum("kxe,er->kxr", J_mean_ext, self.extent_jacobians[element.row])

    def center_jacobian_wrt_raw(self, element: SceneElement) -> np.ndarray:
        """d(box center)/d(raw) — (3, n_raw).  Center = origin + R (E/2)."""
        R = rot_z(element.box.angle)
        return 0.5 * np.einsum("xe,er->xr", R, self.extent_jacobians[element.row])


def _extent_value(world: World, var: VarId | None, const: float) -> float:
    if var is None:
        return const
    return world.full.mean(var)


def derive_scene(
    world: World,
    spacing: float = 0.75,
    space_spacing: float = 1.5,
) -> GeometryScene:
    """Lower the world into a Gaussian geometric scene."""
    module = world.module
    graph = world.graph

    storeys = [eid for eid in module.entities if eid.ifc_class == "IfcBuildingStorey"]
    storey = storeys[0] if storeys else None

    elements: list[SceneElement] = []
    clouds_means: list[np.ndarray] = []
    clouds_covs: list[np.ndarray] = []
    clouds_weights: list[np.ndarray] = []
    clouds_feats: list[np.ndarray] = []
    clouds_elem: list[np.ndarray] = []
    fractions_all: list[np.ndarray] = []
    extent_jacs: list[np.ndarray] = []

    n_raw = world.binding.n_raw
    prim_cursor = 0

    def external_wall(eid: EntityId) -> bool:
        # Set by lowering from IfcRelSpaceBoundary's EXTERNAL flag.
        return bool(module.entities[eid].attrs.get("external", False))

    plan: list[tuple[EntityId, tuple[VarId | None, ...], float, bool]] = []
    for eid in module.entities:
        entity = module.entities[eid]
        if entity.placement is None:
            continue
        if eid.ifc_class == "IfcWall":
            plan.append(
                (
                    eid,
                    (VarId(eid, "Length"), VarId(eid, "Width"), VarId(eid, "Height")),
                    spacing,
                    True,
                )
            )
        elif eid.ifc_class == "IfcSpace":
            assert storey is not None
            plan.append(
                (
                    eid,
                    (VarId(eid, "Length"), VarId(eid, "Width"), VarId(storey, "ClearHeight")),
                    space_spacing,
                    False,
                )
            )
        elif eid.ifc_class == "IfcDoor":
            plan.append(
                (eid, (VarId(eid, "Width"), None, VarId(eid, "Height")), spacing, True)
            )

    for row, (eid, extent_vars, elem_spacing, is_solid) in enumerate(plan):
        entity = module.entities[eid]
        assert entity.placement is not None
        extents = (
            _extent_value(world, extent_vars[0], 0.0),
            _extent_value(world, extent_vars[1], DOOR_LEAF_THICKNESS),
            _extent_value(world, extent_vars[2], 0.0),
        )
        box = OrientedBox(
            origin=(entity.placement.x, entity.placement.y, entity.placement.z),
            angle=entity.placement.angle,
            extents=extents,
        )
        means, covs, weights, fractions = gaussianize_box(box, elem_spacing)

        # Extent Jacobian rows back into raw space (zero row for constants).
        J_ext = np.zeros((3, n_raw), dtype=np.float64)
        live = [v for v in extent_vars if v is not None]
        if live:
            H, _ = jacobian_rows(world.binding, world.belief, tuple(live))
            cursor = 0
            for axis, var in enumerate(extent_vars):
                if var is not None:
                    J_ext[axis] = H[cursor]
                    cursor += 1

        feats = np.zeros((means.shape[0], N_FEATURES), dtype=np.float64)
        cls_channel = CLASS_CHANNEL.get(eid.ifc_class)
        if cls_channel is not None:
            feats[:, cls_channel] = 1.0
        feats[:, FEATURE_NAMES.index("log_volume")] = np.log(weights)
        is_ext = external_wall(eid) if eid.ifc_class == "IfcWall" else True
        feats[:, FEATURE_NAMES.index("u_value")] = DEFAULT_U.get(
            (eid.ifc_class, is_ext), 0.0
        )
        feats[:, FEATURE_NAMES.index("load_bearing")] = (
            1.0 if eid.ifc_class == "IfcWall" else 0.0
        )
        feats[:, FEATURE_NAMES.index("external")] = 1.0 if is_ext else 0.0
        feats[:, FEATURE_NAMES.index("storey_index")] = 0.0

        count = means.shape[0]
        elements.append(
            SceneElement(
                row=row,
                entity_id=eid,
                name=entity.name,
                box=box,
                extent_vars=extent_vars,  # type: ignore[arg-type]
                prim_start=prim_cursor,
                prim_end=prim_cursor + count,
                is_solid=is_solid,
            )
        )
        prim_cursor += count
        clouds_means.append(means)
        clouds_covs.append(covs)
        clouds_weights.append(weights)
        clouds_feats.append(feats)
        clouds_elem.append(np.full(count, row, dtype=np.intp))
        fractions_all.append(fractions)
        extent_jacs.append(J_ext)

    # -- graph-derived element affinity (structural attention prior) ------
    n_elem = len(elements)
    row_of_eid = {e.entity_id: e.row for e in elements}
    affinity = np.zeros((n_elem, n_elem), dtype=np.float64)

    def _relate(a: EntityId, b: EntityId, strength: float) -> None:
        ra, rb = row_of_eid.get(a), row_of_eid.get(b)
        if ra is not None and rb is not None and ra != rb:
            affinity[ra, rb] = max(affinity[ra, rb], strength)
            affinity[rb, ra] = max(affinity[rb, ra], strength)

    for eid in module.entities:
        if eid.ifc_class == "IfcWall":
            for space in graph.spaces_of_wall(eid):
                _relate(eid, space, 1.0)
        if eid.ifc_class == "IfcDoor":
            for opening in graph.opening_of_filler(eid):
                for wall in graph.wall_of_opening(opening):
                    _relate(eid, wall, 1.0)
    # Walls sharing a bounded space are construction-adjacent.
    for eid in module.entities:
        if eid.ifc_class == "IfcSpace":
            walls = graph.bounding_walls(eid)
            for i, wa in enumerate(walls):
                for wb in walls[i + 1 :]:
                    _relate(wa, wb, 0.5)

    version = world.digest()
    cloud = GaussianCloud(
        np.concatenate(clouds_means),
        np.concatenate(clouds_covs),
        np.concatenate(clouds_weights),
        np.concatenate(clouds_feats),
        np.concatenate(clouds_elem),
        version,
    )
    cloud.extras["fractions"] = np.concatenate(fractions_all)
    cloud.validate()

    # -- exemption pairs: expected contacts, not clashes -------------------
    row_of = {e.entity_id: e.row for e in elements}
    exempt: set[tuple[int, int]] = set()

    def _exempt(a: EntityId, b: EntityId) -> None:
        if a in row_of and b in row_of:
            pair = (min(row_of[a], row_of[b]), max(row_of[a], row_of[b]))
            exempt.add(pair)

    for eid in module.entities:
        if eid.ifc_class == "IfcDoor":
            for opening in graph.opening_of_filler(eid):
                for wall in graph.wall_of_opening(opening):
                    _exempt(eid, wall)
                    for space in graph.spaces_of_wall(wall):
                        _exempt(eid, space)
        if eid.ifc_class == "IfcSpace":
            for wall in graph.bounding_walls(eid):
                _exempt(eid, wall)

    return GeometryScene(
        world=world,
        elements=tuple(elements),
        cloud=cloud,
        extent_jacobians=np.stack(extent_jacs) if extent_jacs else np.zeros((0, 3, n_raw)),
        version=version,
        exempt_pairs=frozenset(exempt),
        element_affinity=affinity,
    )


def relative_covariance(
    scene: GeometryScene, a: SceneElement, b: SceneElement
) -> np.ndarray:
    """Sigma_rel of the two box centers: (J_a - J_b) Sigma (J_a - J_b)^T."""
    J = scene.center_jacobian_wrt_raw(a) - scene.center_jacobian_wrt_raw(b)
    return J @ scene.world.belief.sigma @ J.T


def support_radius(element: SceneElement, direction: np.ndarray) -> float:
    """Half-extent of the element's box along a world-frame direction.

    For a box with rotation R and half-extents h, the support radius along
    unit direction u is sum_i h_i |u . R e_i|.
    """
    R = rot_z(element.box.angle)
    h = 0.5 * np.asarray(element.box.extents)
    return float(np.abs(direction @ R) @ h)

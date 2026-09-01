"""The Architectural Intermediate Representation.

The IR realizes the canonical state tuple ``S = (X, mu, Sigma, G, C, R)``
from the project vision (README §6): a :class:`Module` carries the symbolic
components — entities ``X``, relationship graph ``G``, constraints ``C``,
and representation metadata ``R`` — while the continuous belief ``(mu,
Sigma)`` lives in the Gaussian backend (:mod:`gat.gaussian.state`), bound to
the IR through variable identities.

Discrete semantics (IFC classes, names, relationships, placements) are
never Gaussianized: they stay symbolic here, per README §4 / §14.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Mapping, Union

from gat.ids import EntityId, VarId
from gat.ir.exprs import Expr


class Role(Enum):
    """Whether a quantity slot is canonical belief or a derived pushforward."""

    RAW = "raw"
    DERIVED = "derived"


class Unit(Enum):
    M = "m"
    M2 = "m2"
    M3 = "m3"
    CURRENCY = "cur"
    CURRENCY_PER_M3 = "cur/m3"


class RelKind(Enum):
    AGGREGATES = "aggregates"          # spatial decomposition: project > building > storey
    CONTAINS = "contains"              # storey contains elements
    BOUNDS = "bounds"                  # wall bounds space
    VOIDS = "voids"                    # opening voids wall
    FILLS = "fills"                    # door fills opening


@dataclass(frozen=True, order=True)
class Rel:
    """A typed, directed relationship edge between two entities."""

    kind: RelKind = field(compare=False)
    source: EntityId
    target: EntityId
    source_ref: int | None = field(default=None, compare=False)

    def sort_key(self) -> tuple:
        return (self.kind.value, self.source, self.target)


@dataclass(frozen=True)
class Placement:
    """Deterministic local placement of an entity in world coordinates.

    v0 keeps placements out of the Gaussian layer: origin and rotation are
    exact metadata.  The geometric Gaussian layer derives primitive means
    from placements plus (uncertain) dimensional parameters.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    angle: float = 0.0  # rotation about +Z, radians

    def transform_point(self, local_x: float, local_y: float, local_z: float) -> tuple[float, float, float]:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return (
            self.x + c * local_x - s * local_y,
            self.y + s * local_x + c * local_y,
            self.z + local_z,
        )


@dataclass(frozen=True)
class QtySlot:
    """One continuous quantity owned by an entity.

    RAW slots carry a prior ``N(prior_mu, prior_sigma^2)`` and no defining
    expression; DERIVED slots carry an :class:`~gat.ir.exprs.Expr` over
    other slots and no prior.  ``source_ref`` is the STEP instance id of the
    originating ``IfcQuantity*`` record when the slot round-trips to IFC.
    """

    var: VarId
    role: Role
    unit: Unit
    prior_mu: float = 0.0
    prior_sigma: float = 0.0
    expr: Expr | None = None
    source_ref: int | None = None

    def __post_init__(self) -> None:
        if self.role is Role.RAW and self.expr is not None:
            raise ValueError(f"raw slot {self.var} must not carry an expression")
        if self.role is Role.DERIVED and self.expr is None:
            raise ValueError(f"derived slot {self.var} requires an expression")


@dataclass(frozen=True)
class NonNegative:
    """The quantity must be physically non-negative."""

    var: VarId
    tol: float = 1e-9


@dataclass(frozen=True)
class LessEqual:
    """``lhs <= rhs`` must hold at the state mean."""

    lhs: VarId
    rhs: VarId
    tol: float = 1e-9


@dataclass(frozen=True)
class ExprEquals:
    """``var`` must equal ``expr`` at the mean — a redundant restatement of a
    derived definition through an independent code path, checking lowering
    itself."""

    var: VarId
    expr: Expr
    tol: float = 1e-9


Constraint = Union[NonNegative, LessEqual, ExprEquals]


def _constraint_sort_key(c: Constraint) -> tuple:
    if isinstance(c, NonNegative):
        return ("1-nonneg", c.var, c.var)
    if isinstance(c, LessEqual):
        return ("2-lesseq", c.lhs, c.rhs)
    return ("3-expreq", c.var, c.var)


@dataclass(frozen=True)
class Entity:
    """An architectural entity: symbolic identity plus its quantity slots."""

    id: EntityId
    name: str
    attrs: Mapping[str, str | int | float] = field(default_factory=dict)
    slots: Mapping[str, QtySlot] = field(default_factory=dict)
    placement: Placement | None = None
    source_ref: int | None = None

    def var(self, quantity: str) -> VarId:
        return VarId(self.id, quantity)


@dataclass
class Module:
    """The Architectural IR container.

    Entities are keyed by :class:`EntityId` and held in sorted-key order;
    relationship edges and constraints are sorted tuples.  The module is
    treated as immutable after lowering — transformations act on the
    Gaussian belief, never on the symbolic structure, in v0.
    """

    entities: dict[EntityId, Entity]
    rels: tuple[Rel, ...]
    constraints: tuple[Constraint, ...]
    meta: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.entities = dict(sorted(self.entities.items()))
        self.rels = tuple(sorted(self.rels, key=Rel.sort_key))
        self.constraints = tuple(sorted(self.constraints, key=_constraint_sort_key))

    # -- queries -----------------------------------------------------------

    def entity(self, eid: EntityId) -> Entity:
        return self.entities[eid]

    def slot(self, var: VarId) -> QtySlot:
        entity = self.entities.get(var.entity)
        if entity is None:
            raise KeyError(f"unknown entity {var.entity}")
        slot = entity.slots.get(var.quantity)
        if slot is None:
            raise KeyError(f"entity {var.entity} has no quantity {var.quantity!r}")
        return slot

    def all_slots(self) -> Iterator[QtySlot]:
        """All quantity slots, in deterministic (entity, quantity) order."""
        for eid in self.entities:
            entity = self.entities[eid]
            for qname in sorted(entity.slots):
                yield entity.slots[qname]

    def raw_vars(self) -> tuple[VarId, ...]:
        return tuple(s.var for s in self.all_slots() if s.role is Role.RAW)

    def derived_vars(self) -> tuple[VarId, ...]:
        return tuple(s.var for s in self.all_slots() if s.role is Role.DERIVED)

    def entities_of_class(self, ifc_class: str) -> tuple[Entity, ...]:
        return tuple(e for eid, e in self.entities.items() if eid.ifc_class == ifc_class)

    def digest(self) -> str:
        """SHA-256 of the deterministic printer output."""
        from gat.ir.printer import print_module

        return hashlib.sha256(print_module(self).encode("utf-8")).hexdigest()

"""Canonical identities for entities and state variables.

``EntityId`` and ``VarId`` are frozen, totally ordered value objects.  Their
sort order is the ONLY tiebreak used anywhere in the engine (variable
indexing, report ordering, ready-queue ordering in topological sorts), which
is what makes every downstream artifact deterministic for a given model.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class EntityId:
    """Identity of an architectural entity: IFC class + GlobalId."""

    ifc_class: str
    global_id: str

    def __str__(self) -> str:
        return f"{self.ifc_class}:{self.global_id}"


@dataclass(frozen=True, order=True)
class VarId:
    """Identity of one continuous state variable owned by an entity."""

    entity: EntityId
    quantity: str

    def __str__(self) -> str:
        return f"{self.entity.global_id}.{self.quantity}"

    @property
    def qualified(self) -> str:
        """Fully qualified form including the IFC class."""
        return f"{self.entity}.{self.quantity}"

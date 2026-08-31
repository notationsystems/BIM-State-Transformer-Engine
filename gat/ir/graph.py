"""Typed relationship graph over the Architectural IR.

A thin, deterministic query layer over the module's :class:`~gat.ir.core.Rel`
edges.  All query results are tuples in sorted order, so anything computed
from graph traversals is reproducible.
"""

from __future__ import annotations

from collections import defaultdict

from gat.ids import EntityId
from gat.ir.core import Module, Rel, RelKind


class RelationshipGraph:
    """Adjacency view of a module's typed relationship edges."""

    def __init__(self, rels: tuple[Rel, ...]):
        self._out: dict[tuple[RelKind, EntityId], list[EntityId]] = defaultdict(list)
        self._in: dict[tuple[RelKind, EntityId], list[EntityId]] = defaultdict(list)
        self._edges = tuple(sorted(rels, key=Rel.sort_key))
        for rel in self._edges:
            self._out[(rel.kind, rel.source)].append(rel.target)
            self._in[(rel.kind, rel.target)].append(rel.source)
        for adj in (self._out, self._in):
            for key in adj:
                adj[key] = sorted(adj[key])

    @classmethod
    def of(cls, module: Module) -> "RelationshipGraph":
        return cls(module.rels)

    # -- generic queries ---------------------------------------------------

    def edges(self) -> tuple[Rel, ...]:
        return self._edges

    def targets(self, kind: RelKind, source: EntityId) -> tuple[EntityId, ...]:
        return tuple(self._out.get((kind, source), ()))

    def sources(self, kind: RelKind, target: EntityId) -> tuple[EntityId, ...]:
        return tuple(self._in.get((kind, target), ()))

    # -- architectural queries ---------------------------------------------

    def openings_of_wall(self, wall: EntityId) -> tuple[EntityId, ...]:
        """Openings that void the given wall."""
        return self.sources(RelKind.VOIDS, wall)

    def wall_of_opening(self, opening: EntityId) -> tuple[EntityId, ...]:
        return self.targets(RelKind.VOIDS, opening)

    def fillers_of_opening(self, opening: EntityId) -> tuple[EntityId, ...]:
        """Doors (or other elements) filling the given opening."""
        return self.sources(RelKind.FILLS, opening)

    def opening_of_filler(self, filler: EntityId) -> tuple[EntityId, ...]:
        return self.targets(RelKind.FILLS, filler)

    def bounding_walls(self, space: EntityId) -> tuple[EntityId, ...]:
        return self.sources(RelKind.BOUNDS, space)

    def spaces_of_wall(self, wall: EntityId) -> tuple[EntityId, ...]:
        return self.targets(RelKind.BOUNDS, wall)

    def contained_elements(self, container: EntityId) -> tuple[EntityId, ...]:
        return self.targets(RelKind.CONTAINS, container)

    def container_of(self, element: EntityId) -> tuple[EntityId, ...]:
        return self.sources(RelKind.CONTAINS, element)

    def aggregated_children(self, parent: EntityId) -> tuple[EntityId, ...]:
        return self.targets(RelKind.AGGREGATES, parent)

    def spatial_edges(self) -> tuple[Rel, ...]:
        """AGGREGATES + CONTAINS edges — the spatial decomposition tree."""
        return tuple(
            r for r in self._edges if r.kind in (RelKind.AGGREGATES, RelKind.CONTAINS)
        )

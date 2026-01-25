"""
directed_marked_strips.py

Directed variants of marked strips/annuli. Directions are per edge and
restrict traversal: you may enter IN (or UNDIRECTED) edges and exit OUT
(or UNDIRECTED) edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable

from marked_strips import BoundaryEdge, EdgeRef, MarkedAnnulus, MarkedStrip
from square import Side


class EdgeDir(str, Enum):
    IN = "in"
    OUT = "out"
    UNDIRECTED = "undirected"


def _dir_allows_in(d: EdgeDir) -> bool:
    return d in (EdgeDir.IN, EdgeDir.UNDIRECTED)


def _dir_allows_out(d: EdgeDir) -> bool:
    return d in (EdgeDir.OUT, EdgeDir.UNDIRECTED)


@dataclass
class DirectedMarkedStrip:
    """
    Wrapper around MarkedStrip with per-edge direction constraints.
    """

    base: MarkedStrip
    _edge_dirs: Dict[EdgeRef, EdgeDir] = field(default_factory=dict)

    def edge_direction(self, e: EdgeRef) -> EdgeDir:
        return self._edge_dirs.get(e, EdgeDir.UNDIRECTED)

    def set_edge_direction(self, e: EdgeRef, direction: EdgeDir) -> None:
        self._edge_dirs[e] = direction

    def clear_edge_direction(self, e: EdgeRef) -> None:
        self._edge_dirs.pop(e, None)

    def validate_directions(self) -> bool:
        """
        Validate that every paired edge is either undirected or IN/OUT.
        """
        for p in self.base.all_pairs():
            ea = EdgeRef(p.a.side, p.a.i)
            eb = EdgeRef(p.b.side, p.b.i)
            da = self.edge_direction(ea)
            db = self.edge_direction(eb)
            if da == EdgeDir.UNDIRECTED and db == EdgeDir.UNDIRECTED:
                continue
            if _dir_allows_out(da) and _dir_allows_in(db):
                continue
            if _dir_allows_out(db) and _dir_allows_in(da):
                continue
            return False

        seen: set[int] = set()
        for e in self.base.all_edge_refs():
            if not self.base.is_interior_edge(e):
                continue
            edge_obj = self.base.square(e.i).edge(e.side)
            if id(edge_obj) in seen:
                continue
            seen.add(id(edge_obj))
            e2 = self.base._interior_pair(e)
            if e2 is None:
                continue
            da = self.edge_direction(e)
            db = self.edge_direction(e2)
            if da == EdgeDir.UNDIRECTED and db == EdgeDir.UNDIRECTED:
                continue
            if _dir_allows_out(da) and _dir_allows_in(db):
                continue
            if _dir_allows_out(db) and _dir_allows_in(da):
                continue
            return False

        return True

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getstate__(self):
        return {"base": self.base, "_edge_dirs": self._edge_dirs}

    def __setstate__(self, state):
        object.__setattr__(self, "base", state["base"])
        object.__setattr__(self, "_edge_dirs", state.get("_edge_dirs", {}))

    def __str__(self) -> str:
        header = f"DirectedMarkedStrip(N={self.N})"
        return header + "\n" + MarkedStrip._render_ascii(
            self, self.base.strip.squares, wrap=False
        )


@dataclass
class DirectedMarkedAnnulus:
    """
    Wrapper around MarkedAnnulus with per-edge direction constraints.
    """

    base: MarkedAnnulus
    _edge_dirs: Dict[EdgeRef, EdgeDir] = field(default_factory=dict)

    def edge_direction(self, e: EdgeRef) -> EdgeDir:
        return self._edge_dirs.get(e, EdgeDir.UNDIRECTED)

    def set_edge_direction(self, e: EdgeRef, direction: EdgeDir) -> None:
        self._edge_dirs[e] = direction

    def clear_edge_direction(self, e: EdgeRef) -> None:
        self._edge_dirs.pop(e, None)

    def validate_directions(self) -> bool:
        """
        Validate that every paired edge is either undirected or IN/OUT.
        """
        for p in self.base.all_pairs():
            ea = EdgeRef(p.a.side, p.a.i)
            eb = EdgeRef(p.b.side, p.b.i)
            da = self.edge_direction(ea)
            db = self.edge_direction(eb)
            if da == EdgeDir.UNDIRECTED and db == EdgeDir.UNDIRECTED:
                continue
            if _dir_allows_out(da) and _dir_allows_in(db):
                continue
            if _dir_allows_out(db) and _dir_allows_in(da):
                continue
            return False

        seen: set[int] = set()
        for e in self.base.all_edge_refs():
            if not self.base.is_interior_edge(e):
                continue
            edge_obj = self.base.square(e.i).edge(e.side)
            if id(edge_obj) in seen:
                continue
            seen.add(id(edge_obj))
            e2 = self.base._interior_pair(e)
            if e2 is None:
                continue
            da = self.edge_direction(e)
            db = self.edge_direction(e2)
            if da == EdgeDir.UNDIRECTED and db == EdgeDir.UNDIRECTED:
                continue
            if _dir_allows_out(da) and _dir_allows_in(db):
                continue
            if _dir_allows_out(db) and _dir_allows_in(da):
                continue
            return False

        return True

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.base, name)

    def __getstate__(self):
        return {"base": self.base, "_edge_dirs": self._edge_dirs}

    def __setstate__(self, state):
        object.__setattr__(self, "base", state["base"])
        object.__setattr__(self, "_edge_dirs", state.get("_edge_dirs", {}))

    def __str__(self) -> str:
        header = f"DirectedMarkedAnnulus(N={self.N})"
        return header + "\n" + MarkedAnnulus._render_ascii(
            self, self.base.annulus.squares, wrap=True
        )

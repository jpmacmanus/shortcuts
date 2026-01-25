"""
chords.py

Chord model for a square, with crossing detection using the boundary's
cyclic order of ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from edge import Port
from square import Side, Square


@dataclass(frozen=True)
class BoundaryPoint:
    """
    A port on a specific side of a square.

    Ports are identified by object identity; the same Port must not appear on
    multiple edges.
    """

    side: Side
    port: Port

    def __str__(self) -> str:
        return f"{self.side.short()}[{self.port}]"


@dataclass(frozen=True)
class Chord:
    """Chord connecting two boundary points on distinct sides."""

    a: BoundaryPoint
    b: BoundaryPoint

    def __post_init__(self) -> None:
        if self.a.side == self.b.side:
            raise ValueError("Chord endpoints must be on distinct sides")
        if self.a.port is self.b.port:
            # Allow LEFT/RIGHT reuse for the N=1 annulus self-identified edge case.
            if {self.a.side, self.b.side} != {Side.LEFT, Side.RIGHT}:
                raise ValueError("Chord endpoints must be distinct ports")


def boundary_cyclic_order(square: Square) -> List[BoundaryPoint]:
    """
    Return the cyclic order of boundary points around the square.

    Orientation convention (counterclockwise):
      - TOP is oriented right -> left
      - LEFT is oriented top -> bottom
      - BOTTOM is oriented left -> right
      - RIGHT is oriented bottom -> top

    Ports are assumed to be stored in edge order matching each edge's direction.
    """
    order: List[BoundaryPoint] = []

    for p in square.top.ports():
        order.append(BoundaryPoint(Side.TOP, p))
    for p in square.left.ports():
        order.append(BoundaryPoint(Side.LEFT, p))
    for p in square.bottom.ports():
        order.append(BoundaryPoint(Side.BOTTOM, p))
    for p in square.right.ports():
        order.append(BoundaryPoint(Side.RIGHT, p))

    return order


def positions_in_order(order: Sequence[BoundaryPoint]) -> Dict[BoundaryPoint, int]:
    """Map each boundary point to its index in the cyclic order."""
    pos: Dict[BoundaryPoint, int] = {}
    for idx, p in enumerate(order):
        if p in pos:
            raise ValueError("Duplicate boundary point in cyclic order")
        pos[p] = idx
    return pos


def chords_cross(
    chord1: Chord,
    chord2: Chord,
    pos: Dict[BoundaryPoint, int],
) -> bool:
    """
    Determine if two chords cross (endpoints alternate in cyclic order).

    Assumes all endpoints are pairwise distinct.
    """
    a, b = chord1.a, chord1.b
    c, d = chord2.a, chord2.b

    pa, pb = pos[a], pos[b]
    pc, pd = pos[c], pos[d]

    if pa == pb or pc == pd:
        raise ValueError("Chord has identical endpoints")

    if pa > pb:
        pa, pb = pb, pa
    if pc > pd:
        pc, pd = pd, pc

    def between(x: int) -> bool:
        return pa < x < pb

    return between(pc) != between(pd)


def chords_cross_in_square(square: Square, chord1: Chord, chord2: Chord) -> bool:
    """
    Convenience wrapper: compute cyclic order from the square and test crossing.
    """
    order = boundary_cyclic_order(square)
    pos = positions_in_order(order)
    return chords_cross(chord1, chord2, pos)

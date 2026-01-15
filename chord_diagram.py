# chord_diagram.py
"""
Square-local data structures for track schematics.

This file encodes the key complication you flagged:

- A single square may contain multiple disjoint track segments (chords).
- Marked boundary edges (TOP/BOTTOM sides of a square) may be intersected multiple
  times, yielding multiple boundary points ("slots") on that side.
- Vertical sides (LEFT/RIGHT) are allowed, but (for your search) you typically
  enforce at most one intersection per vertical edge, so each vertical side has
  at most one slot in a given square.

We model each square as a circle of boundary points in a fixed cyclic order.
A chord is an unordered pair of boundary points. A set of chords is valid if
it is noncrossing (embedded) in that cyclic order.

This file provides:
- PortKind and BoundaryPoint primitives
- A boundary cyclic ordering model parameterized by (top_slots, bottom_slots,
  use_left, use_right)
- A noncrossing predicate for inserting chords
- A SquareChordSet container with pretty-printing

Run directly for small sanity demos:
    python chord_diagram.py
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple


class PortKind(str, Enum):
    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"

    def short(self) -> str:
        return {"top": "T", "right": "R", "bottom": "B", "left": "L"}[self.value]


@dataclass(frozen=True, order=True)
class BoundaryPoint:
    """
    A specific intersection point on a specific side of a square.

    For TOP/BOTTOM, slot is a positive integer 1..k.
    For LEFT/RIGHT, slot should always be 1 (if used at all).
    """
    kind: PortKind
    slot: int = 1

    def __post_init__(self) -> None:
        if self.slot <= 0:
            raise ValueError("slot must be a positive integer")

        if self.kind in (PortKind.LEFT, PortKind.RIGHT) and self.slot != 1:
            raise ValueError("LEFT/RIGHT can only use slot=1 in this model")

    def __str__(self) -> str:
        if self.kind in (PortKind.LEFT, PortKind.RIGHT):
            return f"{self.kind.short()}"
        return f"{self.kind.short()}{self.slot}"

    def __repr__(self) -> str:
        return f"BoundaryPoint(kind={self.kind.value!r}, slot={self.slot})"


Chord = Tuple[BoundaryPoint, BoundaryPoint]


def normalize_chord(a: BoundaryPoint, b: BoundaryPoint) -> Chord:
    """
    Store chords canonically as (min, max) under the dataclass ordering.
    """
    return (a, b) if a <= b else (b, a)


def boundary_cyclic_order(
    top_slots: int,
    bottom_slots: int,
    use_left: bool,
    use_right: bool,
) -> List[BoundaryPoint]:
    """
    Return the cyclic order of boundary points around the square.

    Convention:
      - TOP points appear left-to-right: T1, T2, ..., T_top_slots
      - then RIGHT (if present): R
      - then BOTTOM points appear right-to-left: B_bottom_slots, ..., B2, B1
        (this choice makes the order consistent around the perimeter)
      - then LEFT (if present): L

    This is purely a combinatorial convention; it is used to test noncrossing.
    """
    if top_slots < 0 or bottom_slots < 0:
        raise ValueError("slot counts must be nonnegative")

    order: List[BoundaryPoint] = []
    for k in range(1, top_slots + 1):
        order.append(BoundaryPoint(PortKind.TOP, k))

    if use_right:
        order.append(BoundaryPoint(PortKind.RIGHT, 1))

    for k in range(bottom_slots, 0, -1):
        order.append(BoundaryPoint(PortKind.BOTTOM, k))

    if use_left:
        order.append(BoundaryPoint(PortKind.LEFT, 1))

    return order


def positions_in_order(order: Sequence[BoundaryPoint]) -> Dict[BoundaryPoint, int]:
    """
    Map each boundary point to its index in the cyclic order.
    """
    pos: Dict[BoundaryPoint, int] = {}
    for idx, p in enumerate(order):
        if p in pos:
            raise ValueError(f"Duplicate boundary point in cyclic order: {p}")
        pos[p] = idx
    return pos


def chords_cross_in_cyclic_order(
    chord1: Chord,
    chord2: Chord,
    pos: Dict[BoundaryPoint, int],
) -> bool:
    """
    Determine if two chords cross (their endpoints alternate) with respect to a
    cyclic order encoded by the position map `pos`.

    Standard criterion:
      Let chord1 endpoints have positions a<b (after swapping).
      Let chord2 endpoints positions c<d.
      chord2 crosses chord1 iff exactly one of {c,d} lies strictly between a and b.

    This is correct on a circle when "between(a,b)" is interpreted along the
    increasing direction in the chosen cut of the circle.
    """
    a, b = chord1
    c, d = chord2

    pa, pb = pos[a], pos[b]
    pc, pd = pos[c], pos[d]

    if pa == pb or pc == pd:
        raise ValueError("Chord has identical endpoints (not allowed).")

    # Normalize to pa < pb, pc < pd in the linear indexing.
    if pa > pb:
        pa, pb = pb, pa
    if pc > pd:
        pc, pd = pd, pc

    def between(x: int) -> bool:
        return pa < x < pb

    in1 = between(pc)
    in2 = between(pd)
    return in1 != in2  # XOR


@dataclass(frozen=True)
class SquareChordSet:
    """
    Immutable set of chords in a square, together with enough parameters
    to define the cyclic order used for noncrossing checks.

    Note: top_slots and bottom_slots are the CURRENT numbers of available slots
    on the TOP/BOTTOM sides (i.e. how many distinct boundary points exist so far).
    LEFT/RIGHT presence is determined by whether any chord endpoint uses them.
    """
    chords: FrozenSet[Chord] = frozenset()
    top_slots: int = 0
    bottom_slots: int = 0

    def __post_init__(self) -> None:
        if self.top_slots < 0 or self.bottom_slots < 0:
            raise ValueError("slot counts must be nonnegative")
        # Basic chord sanity:
        for a, b in self.chords:
            if a == b:
                raise ValueError("Chord endpoints must be distinct")
            if a.kind in (PortKind.LEFT, PortKind.RIGHT) and a.slot != 1:
                raise ValueError("LEFT/RIGHT endpoints must have slot=1")
            if b.kind in (PortKind.LEFT, PortKind.RIGHT) and b.slot != 1:
                raise ValueError("LEFT/RIGHT endpoints must have slot=1")

    def uses_left(self) -> bool:
        return any(p.kind == PortKind.LEFT for ch in self.chords for p in ch)

    def uses_right(self) -> bool:
        return any(p.kind == PortKind.RIGHT for ch in self.chords for p in ch)

    def cyclic_order(self) -> List[BoundaryPoint]:
        return boundary_cyclic_order(
            top_slots=self.top_slots,
            bottom_slots=self.bottom_slots,
            use_left=self.uses_left(),
            use_right=self.uses_right(),
        )

    def can_add_chord(self, a: BoundaryPoint, b: BoundaryPoint) -> bool:
        """
        Check whether adding chord (a,b) would preserve noncrossing embeddedness.
        Assumes any needed slots for a/b already exist (i.e. slots <= top_slots/bottom_slots).
        """
        if a == b:
            return False

        # Ensure a/b are compatible with current slot counts.
        if a.kind == PortKind.TOP and a.slot > self.top_slots:
            return False
        if a.kind == PortKind.BOTTOM and a.slot > self.bottom_slots:
            return False
        if b.kind == PortKind.TOP and b.slot > self.top_slots:
            return False
        if b.kind == PortKind.BOTTOM and b.slot > self.bottom_slots:
            return False

        # Determine cyclic order including LEFT/RIGHT if needed by the proposed chord.
        use_left = self.uses_left() or (a.kind == PortKind.LEFT) or (b.kind == PortKind.LEFT)
        use_right = self.uses_right() or (a.kind == PortKind.RIGHT) or (b.kind == PortKind.RIGHT)

        order = boundary_cyclic_order(self.top_slots, self.bottom_slots, use_left, use_right)
        pos = positions_in_order(order)

        new_ch = normalize_chord(a, b)
        # Forbid duplicate chords:
        if new_ch in self.chords:
            return False

        for old in self.chords:
            if chords_cross_in_cyclic_order(new_ch, old, pos):
                return False
        return True

    def add_chord(self, a: BoundaryPoint, b: BoundaryPoint) -> "SquareChordSet":
        """
        Return a new SquareChordSet with chord added; raises if illegal.
        """
        if not self.can_add_chord(a, b):
            raise ValueError(f"Chord {a}-{b} is not addable (would cross or is invalid).")
        new_ch = normalize_chord(a, b)
        return SquareChordSet(
            chords=self.chords | frozenset({new_ch}),
            top_slots=self.top_slots,
            bottom_slots=self.bottom_slots,
        )

    def __str__(self) -> str:
        chs = sorted(self.chords)
        ch_str = ", ".join([f"{a}-{b}" for (a, b) in chs]) if chs else "∅"
        return f"SquareChordSet(T={self.top_slots}, B={self.bottom_slots}, chords={ch_str})"


def _demo_noncrossing() -> None:
    print("=== Demo: noncrossing chord insertion ===")
    S = SquareChordSet(top_slots=3, bottom_slots=3)
    # Add two noncrossing chords
    S = S.add_chord(BoundaryPoint(PortKind.TOP, 1), BoundaryPoint(PortKind.BOTTOM, 1))
    S = S.add_chord(BoundaryPoint(PortKind.TOP, 3), BoundaryPoint(PortKind.BOTTOM, 3))
    print(S)

    # This chord should cross the first one in the chosen cyclic order
    a = BoundaryPoint(PortKind.TOP, 3)
    b = BoundaryPoint(PortKind.BOTTOM, 1)
    print(f"Can add {a}-{b}? {S.can_add_chord(a,b)}")
    try:
        S.add_chord(a, b)
    except ValueError as ex:
        print("Expected rejection:", ex)
    print()


def _demo_left_right() -> None:
    print("=== Demo: including LEFT/RIGHT boundary points ===")
    S = SquareChordSet(top_slots=2, bottom_slots=2)
    # Add chord involving RIGHT (vertical side)
    S = S.add_chord(BoundaryPoint(PortKind.TOP, 1), BoundaryPoint(PortKind.RIGHT, 1))
    print("After adding T1-R:", S)
    print("Cyclic order:", " ".join(str(p) for p in S.cyclic_order()))

    # Now add another chord involving LEFT
    S = S.add_chord(BoundaryPoint(PortKind.LEFT, 1), BoundaryPoint(PortKind.BOTTOM, 1))
    print("After adding L-B1:", S)
    print("Cyclic order:", " ".join(str(p) for p in S.cyclic_order()))
    print()


if __name__ == "__main__":
    _demo_noncrossing()
    _demo_left_right()

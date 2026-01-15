# track_state.py
"""
Core state representation and "one step" transition plumbing for track schematics.

This file encodes the state transitions used by the shortcut search:

- A track schematic is built by adding chords in squares.
- Multiple disjoint segments per square are supported.
- Marked boundary edges (TOP/BOTTOM) can be used multiple times via slots.
- Unmarked boundary edges are forbidden.
- Vertical edges are counted for lambda; we provide a standard "use at most once"
  mechanism via a bitmask.

IMPORTANT (OR slot reversal):

Marked edges can have multiple ports ("slots"), ordered left-to-right on TOP edges
(and right-to-left on BOTTOM edges in the cyclic-order convention in chord_diagram).

For an orientation-reversing (OR) marked identification, the gluing reverses this
left-to-right order. Under the monotone slot-creation convention in this project
(exiting a marked edge always creates the next new slot on that edge), the correct
transport rule is:

- OP pair: new slot k on edge e corresponds to new slot k on the paired edge e'.
- OR pair: new slot k on edge e corresponds to a *new leftmost* slot on the paired
  edge e'. Equivalently: we must PREPEND a slot to e' (shifting existing slot indices
  on e' up by 1), and land on slot 1.

This file implements that behaviour by shifting chord endpoints in the destination
square whenever an OR gluing creates a new slot.

Run directly to see a small demo.
    python track_state.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from annulus import BoundaryEdge, MarkedAnnulus, Side
from chord_diagram import BoundaryPoint, PortKind, SquareChordSet


def vertical_edge_id(ann: MarkedAnnulus, square_i: int, exit_kind: PortKind) -> int:
    """Delegate vertical-edge indexing to the surface (annulus vs strip)."""
    if exit_kind == PortKind.RIGHT:
        return ann.vertical_edge_id(square_i, go_right=True)
    if exit_kind == PortKind.LEFT:
        return ann.vertical_edge_id(square_i, go_right=False)
    raise ValueError("exit_kind must be LEFT or RIGHT")


def boundary_edge_for(square_i: int, kind: PortKind) -> BoundaryEdge:
    if kind == PortKind.TOP:
        return BoundaryEdge(Side.TOP, square_i)
    if kind == PortKind.BOTTOM:
        return BoundaryEdge(Side.BOTTOM, square_i)
    raise ValueError("kind must be TOP or BOTTOM")


def _shift_square_boundary_slots(sq: SquareChordSet, kind: PortKind, delta: int) -> SquareChordSet:
    """Shift slot indices on TOP/BOTTOM endpoints in a SquareChordSet by +delta."""
    if kind not in (PortKind.TOP, PortKind.BOTTOM):
        raise ValueError("Can only shift TOP/BOTTOM slots")
    if delta == 0:
        return sq

    new_chords = set()
    for a, b in sq.chords:
        if a.kind == kind:
            a = BoundaryPoint(a.kind, a.slot + delta)
        if b.kind == kind:
            b = BoundaryPoint(b.kind, b.slot + delta)
        new_chords.add((a, b))

    top_slots = sq.top_slots
    bottom_slots = sq.bottom_slots
    if kind == PortKind.TOP:
        top_slots += delta
    else:
        bottom_slots += delta

    return SquareChordSet(chords=frozenset(new_chords), top_slots=top_slots, bottom_slots=bottom_slots)


@dataclass(frozen=True)
class Position:
    """A specific boundary point on a specific square."""
    square: int
    point: BoundaryPoint

    def __str__(self) -> str:
        return f"Q{self.square}:{self.point}"


@dataclass(frozen=True)
class TrackState:
    """Immutable global state of a partially-built track schematic component."""
    pos: Position
    lam: int
    used_vertical: int  # bitmask over N vertical edges
    # Slot counts for TOP/BOTTOM boundary edges; LEFT/RIGHT are always slot 1 if used.
    slot_counts: Tuple[Tuple[BoundaryEdge, int], ...]
    # Per-square chord sets, as an immutable mapping (square index -> SquareChordSet)
    squares: Tuple[Tuple[int, SquareChordSet], ...]

    def slot_count(self, e: BoundaryEdge) -> int:
        return dict(self.slot_counts).get(e, 0)

    def square_chords(self, i: int) -> SquareChordSet:
        return dict(self.squares).get(i, SquareChordSet(top_slots=0, bottom_slots=0))

    def __str__(self) -> str:
        sc = dict(self.squares)
        parts = [
            f"TrackState(pos={self.pos}, lam={self.lam}, used_vertical=0b{self.used_vertical:b})",
            f"  slot_counts={dict(self.slot_counts)}",
            f"  squares_with_chords={sorted(sc.keys())}",
        ]
        return "\n".join(parts)


def _update_square_mapping(
    squares: Tuple[Tuple[int, SquareChordSet], ...],
    i: int,
    new_val: SquareChordSet,
) -> Tuple[Tuple[int, SquareChordSet], ...]:
    d = dict(squares)
    d[i] = new_val
    return tuple(sorted(d.items()))


def _update_slot_counts(
    slot_counts: Tuple[Tuple[BoundaryEdge, int], ...],
    e: BoundaryEdge,
    new_count: int,
) -> Tuple[Tuple[BoundaryEdge, int], ...]:
    d = dict(slot_counts)
    d[e] = new_count
    return tuple(sorted(d.items()))


def glue_across(ann: MarkedAnnulus, square_i: int, exit_point: BoundaryPoint) -> Position:
    """
    Glue across the side that exit_point lies on.

    NOTE: For TOP/BOTTOM, this preserves slot index. The OR slot-reversal correction
    is handled in step(); glue_across is retained for backward compatibility.
    """
    k = exit_point.kind

    if k == PortKind.RIGHT:
        j = ann.next_square_horizontal(square_i, go_right=True)
        return Position(j, BoundaryPoint(PortKind.LEFT, 1))
    if k == PortKind.LEFT:
        j = ann.next_square_horizontal(square_i, go_right=False)
        return Position(j, BoundaryPoint(PortKind.RIGHT, 1))

    if k in (PortKind.TOP, PortKind.BOTTOM):
        e = boundary_edge_for(square_i, k)
        if not ann.is_marked(e):
            raise ValueError(f"Unmarked boundary edge used: {e}")
        e2, _rev = ann.pair_info(e)
        kind2 = PortKind.TOP if e2.side == Side.TOP else PortKind.BOTTOM
        return Position(e2.i, BoundaryPoint(kind2, exit_point.slot))

    raise ValueError(f"Unsupported exit kind: {k}")


def step(
    ann: MarkedAnnulus,
    st: TrackState,
    exit_kind: PortKind,
) -> TrackState:
    """
    Perform one step by creating a chord inside the current square from st.pos.point
    to an exit point on side exit_kind.

    Slot rule:
      - If exiting via LEFT/RIGHT, slot is always 1.
      - If exiting via TOP/BOTTOM, we allocate the next slot on that boundary edge:
          slot = current_count + 1
        (This is the "monotone / synchronized creation" convention.)

    OR correction:
      - OP pair: new slot k on e maps to new slot k on e2.
      - OR pair: new slot k on e maps to a new *leftmost* slot on e2, so we:
          (a) increment slot_count(e2),
          (b) shift all existing TOP/BOTTOM endpoints on e2 by +1 in that destination square,
          (c) land at slot 1.
    """
    i = st.pos.square
    N = ann.N

    # Determine the exit point and allocate boundary slots if needed.
    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        exit_point = BoundaryPoint(exit_kind, 1)
        new_slot_counts = st.slot_counts
    else:
        e = boundary_edge_for(i, exit_kind)
        if not ann.is_marked(e):
            raise ValueError(f"Cannot exit to unmarked boundary edge: {e}")
        k = st.slot_count(e) + 1
        exit_point = BoundaryPoint(exit_kind, k)
        new_slot_counts = _update_slot_counts(st.slot_counts, e, k)

    # Prepare the square chord set with enough TOP/BOTTOM slots to include any new slot.
    sq = st.square_chords(i)
    top_slots = sq.top_slots
    bottom_slots = sq.bottom_slots

    if exit_point.kind == PortKind.TOP:
        top_slots = max(top_slots, exit_point.slot)
    if exit_point.kind == PortKind.BOTTOM:
        bottom_slots = max(bottom_slots, exit_point.slot)

    if st.pos.point.kind == PortKind.TOP:
        top_slots = max(top_slots, st.pos.point.slot)
    if st.pos.point.kind == PortKind.BOTTOM:
        bottom_slots = max(bottom_slots, st.pos.point.slot)

    sq = SquareChordSet(chords=sq.chords, top_slots=top_slots, bottom_slots=bottom_slots)

    # Add the chord inside square i, enforcing noncrossing.
    sq2 = sq.add_chord(st.pos.point, exit_point)
    new_squares = _update_square_mapping(st.squares, i, sq2)

    # Update lambda and used vertical edges if exiting via LEFT/RIGHT.
    lam = st.lam
    used_vertical = st.used_vertical
    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        vid = vertical_edge_id(ann, i, exit_kind)
        if (used_vertical >> vid) & 1:
            raise ValueError(f"Vertical edge {vid} already used (at most once).")
        used_vertical |= (1 << vid)
        lam += 1

        # Glue across to the next square / boundary.
        pos2 = glue_across(ann, i, exit_point)

        return TrackState(
            pos=pos2,
            lam=lam,
            used_vertical=used_vertical,
            slot_counts=new_slot_counts,
            squares=new_squares,
        )

    # TOP/BOTTOM gluing with OP/OR slot behaviour.
    e = boundary_edge_for(i, exit_kind)
    e2, is_rev = ann.pair_info(e)
    dest_kind = PortKind.TOP if e2.side == Side.TOP else PortKind.BOTTOM

    k = exit_point.slot
    cur2 = dict(new_slot_counts).get(e2, 0)

    if not is_rev:
        # OP: slot k corresponds to slot k.
        if k > cur2:
            new_slot_counts = _update_slot_counts(new_slot_counts, e2, k)
        pos2 = Position(e2.i, BoundaryPoint(dest_kind, k))
    else:
        # OR: slot k corresponds to a new leftmost slot on e2.
        # Under synchronized creation we expect cur2 == k-1; nevertheless we force count to k.
        new_slot_counts = _update_slot_counts(new_slot_counts, e2, max(cur2 + 1, k))

        # Shift chord endpoints in the destination square along that boundary kind by +1.
        sq_dest = dict(new_squares).get(e2.i)
        if sq_dest is not None:
            sq_dest2 = _shift_square_boundary_slots(sq_dest, dest_kind, +1)
            new_squares = _update_square_mapping(new_squares, e2.i, sq_dest2)

        # Land on the new leftmost slot.
        pos2 = Position(e2.i, BoundaryPoint(dest_kind, 1))

    return TrackState(
        pos=pos2,
        lam=lam,
        used_vertical=used_vertical,
        slot_counts=new_slot_counts,
        squares=new_squares,
    )


def initial_state(start_square: int, start_point: BoundaryPoint) -> TrackState:
    """Construct an initial TrackState with no chords placed yet."""
    return TrackState(
        pos=Position(start_square, start_point),
        lam=0,
        used_vertical=0,
        slot_counts=tuple(),
        squares=tuple(),
    )


def _demo() -> None:
    from annulus import MarkedAnnulus, Side

    print("=== Demo: stepping and square-local noncrossing ===")
    ann = MarkedAnnulus(N=4)
    # Mark: T0 <-> B2 (preserving), T1 <-> B3 (preserving)
    ann.add_marked_pair(ann.edge(Side.TOP, 0), ann.edge(Side.BOTTOM, 2), orientation_reversing=False)
    ann.add_marked_pair(ann.edge(Side.TOP, 1), ann.edge(Side.BOTTOM, 3), orientation_reversing=False)
    print(ann)
    print()

    st = initial_state(start_square=0, start_point=BoundaryPoint(PortKind.LEFT, 1))
    print("Start:")
    print(st)
    print()

    # Step 1: inside Q0 connect L -> TOP (allocates T0 slot 1)
    st = step(ann, st, PortKind.TOP)
    print("After step to TOP:")
    print(st)
    print()

    # Step 2: now at paired edge (B2 slot 1) on Q2; connect B -> RIGHT (vertical)
    st = step(ann, st, PortKind.RIGHT)
    print("After step to RIGHT (uses a vertical edge, lam increments):")
    print(st)
    print()

    # Step 3: now at Q3:LEFT; connect LEFT -> BOTTOM (allocates B3 slot 1)
    st = step(ann, st, PortKind.BOTTOM)
    print("After step to BOTTOM:")
    print(st)
    print()


if __name__ == "__main__":
    _demo()

# search_shortcuts.py
"""
BFS shortcut search with safe pruning.

FIX (strip support):
- In strips, there is no vertical edge to the right of the last square (and no vertical
  edge to the left of the first square). The previous version always offered LEFT/RIGHT
  moves and always tried to compute global points for them, which triggers:
    ValueError: No vertical edge to the right of the last square.

This version:
- Generates LEFT/RIGHT exits only when they exist (using ann.has_vertical_edge / try-catch).
- Computes global points for vertical endpoints only when the vertical edge exists.
- Keeps the same interface and behaviour for annuli.

Interface compatibility preserved:
- ShortcutResult(st, dx, dy, lam, word)
- find_shortcut(..., start_square=None, start_point=None, max_steps=None, max_nodes=..., verbose=False, simple_only=False)
- format_shortcut(...)

Safe optimisations included:
1) Pair-class slot budgets (cap per marked pair class + total).
2) Degree-1 feasibility prune (lower bound on steps needed to close).
3) Smaller visited key + chord fingerprint.
4) Simple-only pruning: forbid horizontal steps against the current dominant direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

from annulus import BoundaryEdge, MarkedAnnulus, Side
from chord_diagram import BoundaryPoint, PortKind
from track_state import TrackState, initial_state, step, vertical_edge_id, boundary_edge_for


# ----------------------------
# Global degree bookkeeping
# ----------------------------

GlobalPoint = Union[Tuple[str, int], Tuple[str, BoundaryEdge, int]]


def _side_code(side: Side) -> int:
    if side == Side.TOP:
        return 0
    if side == Side.BOTTOM:
        return 1
    v = getattr(side, "value", None)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return 0 if v.lower().startswith("t") else 1
    return 0


def canonical_boundary_edge(ann: MarkedAnnulus, e: BoundaryEdge) -> BoundaryEdge:
    if ann.is_marked(e):
        e2, _rev = ann.pair_info(e)
        return min(e, e2)
    return e


def _try_vertical_edge_id(ann: MarkedAnnulus, square_i: int, kind: PortKind) -> Optional[int]:
    """
    For strips, vertical edges do not exist beyond the ends. The underlying
    annulus/strip object raises ValueError in that case. We treat that as “no edge”.
    """
    try:
        return int(vertical_edge_id(ann, square_i, kind))
    except ValueError:
        return None


def global_point_for(ann: MarkedAnnulus, square_i: int, p: BoundaryPoint) -> GlobalPoint:
    if p.kind == PortKind.RIGHT:
        vid = _try_vertical_edge_id(ann, square_i, PortKind.RIGHT)
        if vid is None:
            raise ValueError("No vertical edge to the RIGHT at this square (strip end).")
        return ("V", vid)

    if p.kind == PortKind.LEFT:
        vid = _try_vertical_edge_id(ann, square_i, PortKind.LEFT)
        if vid is None:
            raise ValueError("No vertical edge to the LEFT at this square (strip end).")
        return ("V", vid)

    if p.kind == PortKind.TOP:
        rep = canonical_boundary_edge(ann, BoundaryEdge(Side.TOP, square_i))
        return ("B", rep, p.slot)

    if p.kind == PortKind.BOTTOM:
        rep = canonical_boundary_edge(ann, BoundaryEdge(Side.BOTTOM, square_i))
        return ("B", rep, p.slot)

    raise ValueError("Unknown point kind")


def _gp_sort_key(gp: GlobalPoint) -> Tuple:
    if gp[0] == "V":
        return (0, gp[1])
    return (1, _side_code(gp[1].side), gp[1].i, gp[2])


def degree_inc(
    degrees: Tuple[Tuple[GlobalPoint, int], ...],
    gp: GlobalPoint,
    delta: int = 1,
) -> Tuple[Tuple[GlobalPoint, int], ...]:
    d = dict(degrees)
    d[gp] = d.get(gp, 0) + delta
    if d[gp] == 0:
        del d[gp]
    return tuple(sorted(d.items(), key=lambda kv: _gp_sort_key(kv[0])))


def max_degree(degrees: Tuple[Tuple[GlobalPoint, int], ...]) -> int:
    return max((v for _, v in degrees), default=0)


def all_degrees_two(degrees: Tuple[Tuple[GlobalPoint, int], ...]) -> bool:
    return all(v == 2 for _, v in degrees)


def count_degree_one(degrees: Tuple[Tuple[GlobalPoint, int], ...]) -> int:
    return sum(1 for _, v in degrees if v == 1)


# ----------------------------
# Section 5.4 bookkeeping
# ----------------------------

def square_is_marked(ann: MarkedAnnulus, i: int) -> bool:
    return ann.is_marked(ann.edge(Side.TOP, i)) or ann.is_marked(ann.edge(Side.BOTTOM, i))


@dataclass(frozen=True)
class ShortcutBookkeeping:
    X: int = 0
    dominant_right: bool = True
    run_dir: int = 0
    run_len: int = 0

    Y: int = 0
    s_sign: int = 1
    turn_parity: int = 0

    rev_parity: int = 0
    piece_start_side: Optional[Side] = None

    def dominant_dir(self) -> int:
        return 1 if self.dominant_right else -1


def update_bookkeeping(
    ann: MarkedAnnulus,
    st_before: TrackState,
    exit_kind: PortKind,
    bk: ShortcutBookkeeping,
    *,
    simple_only: bool = False,
) -> Optional[ShortcutBookkeeping]:
    i = st_before.pos.square

    X = bk.X
    dominant_right = bk.dominant_right
    run_dir = bk.run_dir
    run_len = bk.run_len

    Y = bk.Y
    s = bk.s_sign
    turn_parity = bk.turn_parity

    rev_parity = bk.rev_parity
    piece_start_side = bk.piece_start_side

    # ---- horizontal ----
    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        d = 1 if exit_kind == PortKind.RIGHT else -1

        if simple_only:
            dom = 1 if dominant_right else -1
            if d != dom:
                return None

        if run_len == 0:
            run_dir = d
            run_len = 1
        else:
            if d == run_dir:
                run_len += 1
            else:
                dom_dir = 1 if dominant_right else -1
                X += (run_len if run_dir == dom_dir else -run_len)
                run_dir = d
                run_len = 1

        # terminate run upon entering a marked square (annuli) or marked square in strip too
        j = (i + d) % ann.N
        if square_is_marked(ann, j) and run_len > 0:
            dom_dir = 1 if dominant_right else -1
            X += (run_len if run_dir == dom_dir else -run_len)
            run_dir = 0
            run_len = 0

    # ---- vertical ----
    if exit_kind in (PortKind.TOP, PortKind.BOTTOM):
        end_side = Side.TOP if exit_kind == PortKind.TOP else Side.BOTTOM
        if piece_start_side is None:
            piece_start_side = end_side

        if piece_start_side == end_side:
            s = -s
            turn_parity ^= 1
        else:
            Y += s

        e = boundary_edge_for(i, exit_kind)
        e2, is_rev = ann.pair_info(e)
        piece_start_side = e2.side

        if is_rev:
            dominant_right = not dominant_right
            rev_parity ^= 1

    return ShortcutBookkeeping(
        X=X,
        dominant_right=dominant_right,
        run_dir=run_dir,
        run_len=run_len,
        Y=Y,
        s_sign=s,
        turn_parity=turn_parity,
        rev_parity=rev_parity,
        piece_start_side=piece_start_side,
    )


def final_displacement(bk: ShortcutBookkeeping) -> Tuple[int, int]:
    dx = 0 if (bk.rev_parity & 1) else abs(bk.X)
    dy = 0 if (bk.turn_parity & 1) else abs(bk.Y)
    return dx, dy


# ----------------------------
# Search node / result
# ----------------------------

@dataclass(frozen=True)
class ShortcutNode:
    st: TrackState
    degrees: Tuple[Tuple[GlobalPoint, int], ...]
    bk: ShortcutBookkeeping
    word: Tuple[PortKind, ...]


@dataclass(frozen=True)
class ShortcutResult:
    st: TrackState
    dx: int
    dy: int
    lam: int
    word: Tuple[PortKind, ...]


def format_shortcut(sc: ShortcutResult) -> str:
    return f"shortcut: lambda={sc.lam}, dx={sc.dx}, dy={sc.dy}, steps={len(sc.word)}"


# ----------------------------
# Slot budgets (pair-class)
# ----------------------------

def _pair_class_reps(ann: MarkedAnnulus) -> Dict[BoundaryEdge, Tuple[BoundaryEdge, BoundaryEdge]]:
    reps: Dict[BoundaryEdge, Tuple[BoundaryEdge, BoundaryEdge]] = {}
    for p in ann.all_pairs():
        rep = min(p.a, p.b)
        reps[rep] = (p.a, p.b)
    return reps


def slot_budget_ok(
    ann: MarkedAnnulus,
    st: TrackState,
    *,
    max_pair_slots: int,
    max_total_pair_slots: int,
) -> bool:
    reps = _pair_class_reps(ann)
    total = 0
    for _rep, (a, b) in reps.items():
        sa = st.slot_count(a)
        sb = st.slot_count(b)
        m = sa if sa >= sb else sb
        if m > max_pair_slots:
            return False
        total += m
        if total > max_total_pair_slots:
            return False
    return True


# ----------------------------
# Start positions
# ----------------------------

def _start_positions_on_marked_edges(ann: MarkedAnnulus) -> List[Tuple[int, BoundaryPoint]]:
    starts: List[Tuple[int, BoundaryPoint]] = []
    for i in range(ann.N):
        if ann.is_marked(ann.edge(Side.TOP, i)):
            starts.append((i, BoundaryPoint(PortKind.TOP, 1)))
        if ann.is_marked(ann.edge(Side.BOTTOM, i)):
            starts.append((i, BoundaryPoint(PortKind.BOTTOM, 1)))
    return starts


# ----------------------------
# Move ordering / existence (strip-safe)
# ----------------------------

def possible_exit_kinds(ann: MarkedAnnulus, st: TrackState) -> List[PortKind]:
    i = st.pos.square
    kinds: List[PortKind] = []

    # Include LEFT/RIGHT only if the corresponding vertical edge exists.
    if _try_vertical_edge_id(ann, i, PortKind.LEFT) is not None:
        kinds.append(PortKind.LEFT)
    if _try_vertical_edge_id(ann, i, PortKind.RIGHT) is not None:
        kinds.append(PortKind.RIGHT)

    if ann.is_marked(ann.edge(Side.TOP, i)):
        kinds.append(PortKind.TOP)
    if ann.is_marked(ann.edge(Side.BOTTOM, i)):
        kinds.append(PortKind.BOTTOM)
    return kinds


def _exit_cost(ann: MarkedAnnulus, st: TrackState, exit_kind: PortKind) -> int:
    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        return 0
    i = st.pos.square
    e = boundary_edge_for(i, exit_kind)
    if not ann.is_marked(e):
        return 10_000
    e2, _rev = ann.pair_info(e)
    k = st.slot_count(e) + 1
    other = st.slot_count(e2)
    resulting = k if k >= other else other
    return 10 + resulting


def ordered_exit_kinds(ann: MarkedAnnulus, st: TrackState) -> List[PortKind]:
    return sorted(possible_exit_kinds(ann, st), key=lambda k: _exit_cost(ann, st, k))


# ----------------------------
# Visited key: coarse + chord fingerprint
# ----------------------------

def chord_fingerprint(st: TrackState) -> Tuple[Tuple[int, int, int, int], ...]:
    out = []
    for i, sq in st.squares:
        try:
            h = hash(sq.chords)
        except Exception:
            h = hash(tuple(sorted(sq.chords)))
        out.append((i, h, getattr(sq, "top_slots", 0), getattr(sq, "bottom_slots", 0)))
    return tuple(sorted(out))


# ----------------------------
# Transition + shortcut predicate
# ----------------------------

def apply_move(
    ann: MarkedAnnulus,
    node: ShortcutNode,
    exit_kind: PortKind,
    *,
    simple_only: bool,
) -> Optional[ShortcutNode]:
    st = node.st
    i = st.pos.square

    # Determine exit endpoint for degree bookkeeping.
    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        # If this edge doesn't exist (strip end), skip.
        if _try_vertical_edge_id(ann, i, exit_kind) is None:
            return None
        b_pt = BoundaryPoint(exit_kind, 1)
    else:
        e = boundary_edge_for(i, exit_kind)
        if not ann.is_marked(e):
            return None
        b_pt = BoundaryPoint(exit_kind, st.slot_count(e) + 1)

    # Degrees update (guard vertical endpoints for strip ends).
    try:
        gp_a = global_point_for(ann, i, st.pos.point)
        gp_b = global_point_for(ann, i, b_pt)
    except ValueError:
        return None

    degrees = degree_inc(node.degrees, gp_a, 1)
    degrees = degree_inc(degrees, gp_b, 1)
    if max_degree(degrees) > 2:
        return None

    bk2 = update_bookkeeping(ann, st, exit_kind, node.bk, simple_only=simple_only)
    if bk2 is None:
        return None

    try:
        st2 = step(ann, st, exit_kind)
    except Exception:
        return None

    return ShortcutNode(st=st2, degrees=degrees, bk=bk2, word=node.word + (exit_kind,))


def is_shortcut(
    ann: MarkedAnnulus,
    start_pos,
    node: ShortcutNode,
    *,
    simple_only: bool,
) -> Optional[ShortcutResult]:
    if not node.word:
        return None
    if node.st.pos != start_pos:
        return None
    if not all_degrees_two(node.degrees):
        return None

    lam = node.st.lam
    if lam >= ann.N:
        return None

    dx, dy = final_displacement(node.bk)
    if dx == 0 or dy == 0:
        return None

    if simple_only and dx != lam:
        return None

    return ShortcutResult(st=node.st, dx=dx, dy=dy, lam=lam, word=node.word)


# ----------------------------
# Main BFS
# ----------------------------

def find_shortcut(
    ann: MarkedAnnulus,
    *,
    start_square: Optional[int] = None,
    start_point: Optional[BoundaryPoint] = None,
    max_steps: Optional[int] = None,
    max_nodes: int = 250_000,
    verbose: bool = False,
    simple_only: bool = False,
    max_pair_slots: int = 8,
    max_total_pair_slots: int = 64,
) -> Optional[ShortcutResult]:
    if max_steps is None:
        max_steps = 8 * ann.N + 8

    if start_square is not None or start_point is not None:
        if start_square is None or start_point is None:
            raise TypeError("start_square and start_point must be provided together")
        starts = [(start_square, start_point)]
    else:
        starts = _start_positions_on_marked_edges(ann)

    if not starts:
        return None

    for (sq0, pt0) in starts:
        st0 = initial_state(sq0, pt0)
        start_pos = st0.pos

        start_side = Side.TOP if pt0.kind == PortKind.TOP else Side.BOTTOM

        root = ShortcutNode(
            st=st0,
            degrees=tuple(),
            bk=ShortcutBookkeeping(piece_start_side=start_side),
            word=tuple(),
        )

        q: Deque[ShortcutNode] = deque([root])
        seen: Set[Tuple] = set()
        expanded = 0

        while q:
            node = q.popleft()
            expanded += 1
            if expanded > max_nodes:
                if verbose:
                    print(f"[search_shortcuts] hit max_nodes={max_nodes} (start={sq0}:{pt0})")
                break

            depth = len(node.word)
            if depth > max_steps:
                continue
            if node.st.lam >= ann.N:
                continue

            if not slot_budget_ok(
                ann,
                node.st,
                max_pair_slots=max_pair_slots,
                max_total_pair_slots=max_total_pair_slots,
            ):
                continue

            # degree-1 feasibility
            u1 = count_degree_one(node.degrees)
            remaining = max_steps - depth
            if (u1 + 1) // 2 > remaining:
                continue

            res = is_shortcut(ann, start_pos, node, simple_only=simple_only)
            if res is not None:
                return res

            key = (
                node.st.pos.square,
                node.st.pos.point.kind,
                node.st.pos.point.slot,
                node.st.lam,
                node.st.used_vertical,
                node.st.slot_counts,
                node.degrees,
                node.bk,
                chord_fingerprint(node.st),
            )
            if key in seen:
                continue
            seen.add(key)

            for exit_kind in ordered_exit_kinds(ann, node.st):
                # Early filter for simple-only horizontal moves (cheap)
                if simple_only and exit_kind in (PortKind.LEFT, PortKind.RIGHT):
                    dom = node.bk.dominant_dir()
                    if exit_kind == PortKind.LEFT and dom != -1:
                        continue
                    if exit_kind == PortKind.RIGHT and dom != 1:
                        continue

                nxt = apply_move(ann, node, exit_kind, simple_only=simple_only)
                if nxt is None:
                    continue
                q.append(nxt)

    return None


if __name__ == "__main__":
    from annulus import MarkedAnnulus, Side

    ann = MarkedAnnulus(N=2)
    ann.add_marked_pair(ann.edge(Side.BOTTOM, 1), ann.edge(Side.TOP, 0), orientation_reversing=False)

    res = find_shortcut(ann, max_steps=20, max_nodes=50_000, verbose=True)
    if res is None:
        print("No shortcut found within demo bounds.")
    else:
        print(format_shortcut(res))
        print("word:", " ".join(k.value.upper() for k in res.word))

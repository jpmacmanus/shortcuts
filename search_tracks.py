
# search_tracks.py
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from typing import List, Optional, Tuple, Set, Union, Deque

from annulus import BoundaryEdge, MarkedAnnulus, Side
from chord_diagram import BoundaryPoint, PortKind
from track_state import TrackState, initial_state, step, vertical_edge_id, boundary_edge_for

# GlobalPoint keys:
#   ("V", vid) or ("B", rep_edge, slot) where rep_edge is canonical in its marked pair.
GlobalPoint = Union[Tuple[str, int], Tuple[str, BoundaryEdge, int]]


def canonical_boundary_edge(ann: MarkedAnnulus, e: BoundaryEdge) -> BoundaryEdge:
    """
    Canonicalize boundary edges modulo the marked-pair identification.

    If e is marked and paired with e', return min(e,e') under BoundaryEdge ordering.
    Otherwise return e (unmarked boundary edges should not appear in valid tracks anyway).
    """
    if ann.is_marked(e):
        e2, _rev = ann.pair_info(e)
        return min(e, e2)
    return e


def global_point_for(ann: MarkedAnnulus, square_i: int, p: BoundaryPoint) -> GlobalPoint:
    if p.kind == PortKind.RIGHT:
        return ("V", vertical_edge_id(ann, square_i, PortKind.RIGHT))
    if p.kind == PortKind.LEFT:
        return ("V", vertical_edge_id(ann, square_i, PortKind.LEFT))
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
    # ("B", BoundaryEdge, slot)
    return (1, gp[1].side.value, gp[1].i, gp[2])


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


@dataclass(frozen=True)
class SearchNode:
    st: TrackState
    degrees: Tuple[Tuple[GlobalPoint, int], ...]
    word: Tuple[PortKind, ...]  # exit kinds used


def possible_exit_kinds(ann: MarkedAnnulus, st: TrackState) -> List[PortKind]:
    """
    Heuristic order: vertical first, then boundary (marked only).
    """
    i = st.pos.square
    kinds: List[PortKind] = [PortKind.LEFT, PortKind.RIGHT]
    if ann.is_marked(ann.edge(Side.TOP, i)):
        kinds.append(PortKind.TOP)
    if ann.is_marked(ann.edge(Side.BOTTOM, i)):
        kinds.append(PortKind.BOTTOM)
    return kinds


def apply_move(ann: MarkedAnnulus, node: SearchNode, exit_kind: PortKind) -> Optional[SearchNode]:
    st = node.st
    i = st.pos.square

    a_pt = st.pos.point

    # Mirror exit-point allocation used by track_state.step()
    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        b_pt = BoundaryPoint(exit_kind, 1)
    else:
        e = boundary_edge_for(i, exit_kind)
        if not ann.is_marked(e):
            return None
        b_pt = BoundaryPoint(exit_kind, st.slot_count(e) + 1)

    gp_a = global_point_for(ann, i, a_pt)
    gp_b = global_point_for(ann, i, b_pt)

    degrees = degree_inc(node.degrees, gp_a, 1)
    degrees = degree_inc(degrees, gp_b, 1)
    if max_degree(degrees) > 2:
        return None

    try:
        st2 = step(ann, st, exit_kind)
    except Exception:
        return None

    return SearchNode(st=st2, degrees=degrees, word=node.word + (exit_kind,))


def is_closed(start_pos, node: SearchNode) -> bool:
    return (node.word != ()) and (node.st.pos == start_pos) and all_degrees_two(node.degrees)


def find_closed_track_bfs(
    ann: MarkedAnnulus,
    *,
    start_square: int = 0,
    start_point: BoundaryPoint = BoundaryPoint(PortKind.LEFT, 1),
    max_lam: int,
    max_steps: int,
    max_nodes: int = 200000,
) -> Optional[SearchNode]:
    st0 = initial_state(start_square, start_point)
    start_pos = st0.pos
    root = SearchNode(st=st0, degrees=tuple(), word=tuple())

    q: Deque[SearchNode] = deque([root])
    seen: Set[Tuple] = set()
    explored = 0

    while q:
        node = q.popleft()
        explored += 1
        if explored > max_nodes:
            return None

        if node.st.lam > max_lam:
            continue
        if len(node.word) > max_steps:
            continue

        if is_closed(start_pos, node):
            return node

        key = (node.st, node.degrees)
        if key in seen:
            continue
        seen.add(key)

        for exit_kind in possible_exit_kinds(ann, node.st):
            nxt = apply_move(ann, node, exit_kind)
            if nxt is None:
                continue
            if nxt.st.lam > max_lam:
                continue
            if len(nxt.word) > max_steps:
                continue
            q.append(nxt)

    return None


def find_closed_track_iterative(
    ann: MarkedAnnulus,
    *,
    start_square: int = 0,
    start_point: BoundaryPoint = BoundaryPoint(PortKind.LEFT, 1),
    max_lam: Optional[int] = None,
    max_steps: Optional[int] = None,
    max_nodes: int = 200000,
) -> Optional[SearchNode]:
    if max_lam is None:
        max_lam = ann.N
    if max_steps is None:
        max_steps = 6 * ann.N + 6

    for lam_bound in range(0, max_lam + 1):
        res = find_closed_track_bfs(
            ann,
            start_square=start_square,
            start_point=start_point,
            max_lam=lam_bound,
            max_steps=max_steps,
            max_nodes=max_nodes,
        )
        if res is not None:
            return res
    return None


if __name__ == "__main__":
    ann = MarkedAnnulus(N=2)
    print(ann)
    print("search_tracks.py loaded. Import and call find_closed_track_iterative(...).")

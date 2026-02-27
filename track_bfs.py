"""
track_bfs.py

Primitive BFS search for a valid pattern on a marked strip/annulus.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools
import sys
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

from chord_diagram import ChordDiagram
from marked_strips import BoundaryEdge, EdgeRef, MarkedAnnulus, MarkedStrip
from pattern import Pattern
from shortcut_completeness import is_complete, reduce_complete_set
from square import Side
from track_state import TrackState


MarkedSurface = Union[MarkedStrip, MarkedAnnulus]


_COLOR_RESET = "\033[0m"
_COLOR_SUCCESS = "\033[32m"


def _edge_dir_value(surface: MarkedSurface, e: EdgeRef) -> str | None:
    if not hasattr(surface, "edge_direction"):
        return None
    d = surface.edge_direction(e)  # type: ignore[attr-defined]
    return getattr(d, "value", d)


def _dir_allows_in(val: str | None) -> bool:
    return val in (None, "in", "undirected")


def _edge_index(edge_ports, port) -> int:
    return list(edge_ports).index(port)


def _diagram_key(diagram: ChordDiagram) -> Tuple:
    # Canonicalize a diagram by chord endpoints in current port order.
    key = []
    for ch in diagram.chords:
        a = ch.a
        b = ch.b
        key.append((a.side, _edge_index(diagram.square.edge(a.side).ports(), a.port),
                    b.side, _edge_index(diagram.square.edge(b.side).ports(), b.port)))
    return tuple(sorted(key))


def _state_key(st: TrackState) -> Tuple:
    # State identity: cursor position + chord layouts (no weights/counters).
    sq_i, bp = st.cursor
    cursor_key = (
        sq_i,
        bp.side,
        _edge_index(st.surface.square(sq_i).edge(bp.side).ports(), bp.port),
    )
    diagrams_key = tuple(_diagram_key(st.diagrams[i]) for i in range(st.surface.N))
    return (cursor_key, diagrams_key)


def _start_edges(surface: MarkedSurface, *, allow_bottom: bool) -> list[EdgeRef]:
    # Prefer marked TOP edges; optionally include marked BOTTOM edges.
    starts: list[EdgeRef] = []
    for p in surface.all_pairs():
        if p.a.side == Side.TOP:
            starts.append(EdgeRef(p.a.side, p.a.i))
        if p.b.side == Side.TOP:
            starts.append(EdgeRef(p.b.side, p.b.i))
        if allow_bottom:
            if p.a.side == Side.BOTTOM:
                starts.append(EdgeRef(p.a.side, p.a.i))
            if p.b.side == Side.BOTTOM:
                starts.append(EdgeRef(p.b.side, p.b.i))

    if starts:
        seen: set[tuple[Side, int]] = set()
        unique: list[EdgeRef] = []
        for e in starts:
            if not _dir_allows_in(_edge_dir_value(surface, e)):
                continue
            key = (e.side, e.i)
            if key in seen:
                continue
            seen.add(key)
            unique.append(e)
        return unique

    # Fallback: any interior IN edge.
    for e in surface.all_edge_refs():
        if surface.is_interior_edge(e) and _dir_allows_in(_edge_dir_value(surface, e)):
            starts.append(e)

    if not starts:
        raise ValueError("No marked or interior edges available to start")
    return starts


def find_pattern_bfs(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_patterns: int = 1,
    max_ports_per_edge: int | None = 5,
    multiple_interior_edge_crossings: bool = True,
    one_dir_only: bool = False,
    dominant_dir_only: bool = False,
    even_turning: bool = False,
    dy_nonzero: bool = False,
    even_or_pair_count: bool = False,
    dx_all_pos_or_all_neg: bool = False,
    debug: bool = False,
) -> list[Pattern]:
    """
    Run a BFS to find a valid pattern. Returns None if no pattern found within bounds.
    """
    found: list[Pattern] = []
    for start_edge in _start_edges(surface, allow_bottom=not dy_nonzero):
        st0 = TrackState.initialize(surface, start_edge=start_edge)
        if st0 is None:
            continue

        q: Deque[TrackState] = deque([st0])
        seen: Set[Tuple] = set()
        parents: Dict[Tuple, Tuple[Optional[Tuple], TrackState]] = {}
        expanded = 0

        while q:
            st = q.popleft()
            expanded += 1
            if expanded > max_nodes:
                break

            key = _state_key(st)
            if key in seen:
                continue
            seen.add(key)
            if key not in parents:
                parents[key] = (None, st)

            pat = Pattern(surface=st.surface, diagrams=st.diagrams)
            if pat.validate() and _has_any_chord(st):
                if even_turning and (st.turn_count % 2 != 0):
                    pass
                elif dy_nonzero and (st.dy == 0):
                    pass
                elif even_or_pair_count and (st.or_pair_count % 2 != 0):
                    pass
                elif dx_all_pos_or_all_neg and not _dx_all_pos_or_all_neg(st):
                    pass
                else:
                    found.append(pat)
                    if debug:
                        path_states: list[TrackState] = []
                        cur_key: Optional[Tuple] = key
                        while cur_key is not None:
                            prev_key, cur_state = parents[cur_key]
                            path_states.append(cur_state)
                            cur_key = prev_key
                        path_states.reverse()
                        print(f"{_COLOR_SUCCESS}Pattern accepted. Path length: {len(path_states)}{_COLOR_RESET}")
                        for i, st_path in enumerate(path_states, start=1):
                            sq_i, bp = st_path.cursor
                            print(
                                "Step {i}: cursor={side}@{sq} "
                                "turns={turn} last_h={last_h} dy={dy} dy_mod={dy_mod} "
                                "or_pairs={or_pairs} dom_x={dom} last_dir={last_dir}".format(
                                    i=i,
                                    side=bp.side.name.lower(),
                                    sq=sq_i,
                                    turn=st_path.turn_count,
                                    last_h=st_path.last_hor_edge_top,
                                    dy=st_path.dy,
                                    dy_mod=st_path.dy_modifier,
                                    or_pairs=st_path.or_pair_count,
                                    dom=st_path.dominant_x_dir,
                                    last_dir=st_path.last_moved_dir,
                                )
                            )
                            print(f"Step {i}:")
                            print(st_path.render())
                            print()
                    if len(found) >= max_patterns:
                        return found
                # fall through to enqueue neighbors if not accepted

            moves = st.moves(
                multiple_interior_edge_crossings=multiple_interior_edge_crossings,
                one_dir_only=one_dir_only,
                dominant_dir_only=dominant_dir_only,
                max_ports_per_edge=max_ports_per_edge,
            )
            for nxt in moves:
                nxt_key = _state_key(nxt)
                if nxt_key not in parents:
                    parents[nxt_key] = (key, nxt)
                q.append(nxt)

    return found


def _has_any_chord(st: TrackState) -> bool:
    for i in range(st.surface.N):
        if st.diagrams[i].chords:
            return True
    return False


def _dx_all_pos_or_all_neg(st: TrackState) -> bool:
    if not st.dx:
        return False
    all_pos = all(v >= 0 for v in st.dx)
    all_neg = all(v <= 0 for v in st.dx)
    return all_pos or all_neg


def _uses_all_interior_edges(st: TrackState) -> bool:
    # Detect whether every interior edge has at least one interior crossing.
    all_interior: set[int] = set()
    for e in st.surface.all_edge_refs():
        if not st.surface.is_interior_edge(e):
            continue
        edge_obj = st.surface.square(e.i).edge(e.side)
        all_interior.add(id(edge_obj))
    if not all_interior:
        return False
    by_edge = _interior_crossings_by_edge(st)
    used_interior = {eid for eid, crossings in by_edge.items() if crossings}
    return used_interior.issuperset(all_interior)


def _uses_all_marked_edges(st: TrackState) -> bool:
    # Detect whether every marked edge has been used by some chord endpoint.
    marked_edges = st.surface.marked_edges()
    if not marked_edges:
        return True
    used_marked: set[BoundaryEdge] = set()
    for i in range(st.surface.N):
        diagram = st.diagrams[i]
        for ch in diagram.chords:
            for bp in (ch.a, ch.b):
                be = BoundaryEdge(bp.side, i)
                if st.surface.is_marked(be):
                    used_marked.add(be)
    return used_marked.issuperset(marked_edges)


def _marked_edges_used(st: TrackState) -> set[BoundaryEdge]:
    used_marked: set[BoundaryEdge] = set()
    for i in range(st.surface.N):
        diagram = st.diagrams[i]
        for ch in diagram.chords:
            for bp in (ch.a, ch.b):
                be = BoundaryEdge(bp.side, i)
                if st.surface.is_marked(be):
                    used_marked.add(be)
    return used_marked


def _guaranteed_marked_edges(sets: List[TrackState]) -> set[BoundaryEdge]:
    if not sets:
        return set()
    marked_all = set(sets[0].surface.marked_edges())
    if not marked_all:
        return set()
    guaranteed = marked_all.copy()
    for st in sets:
        guaranteed &= _marked_edges_used(st)
    return guaranteed


def _nontrivial_simple_ok(st: TrackState) -> bool:
    return st.dy != 0 and (st.turn_count % 2 == 0)


def _nontrivial_complete_ok(cands: List[TrackState]) -> bool:
    if not cands:
        return False
    if not is_complete(cands):
        return False
    return all(st.or_pair_count % 2 == 0 for st in cands)


def _minimal_cover(
    items: List[tuple[set[BoundaryEdge], object]],
    target: set[BoundaryEdge],
) -> Optional[List[object]]:
    if not target:
        return []
    items = [(edges, obj) for edges, obj in items if edges]
    if not items:
        return None
    # Exact search for small sets
    if len(items) <= 20:
        for r in range(1, len(items) + 1):
            for combo in itertools.combinations(items, r):
                covered = set()
                for edges, _ in combo:
                    covered |= edges
                if covered.issuperset(target):
                    return [obj for _, obj in combo]
    # Greedy fallback
    covered: set[BoundaryEdge] = set()
    chosen: List[object] = []
    remaining = items[:]
    while not covered.issuperset(target) and remaining:
        remaining.sort(key=lambda it: len(it[0] - covered), reverse=True)
        edges, obj = remaining.pop(0)
        if not edges - covered:
            continue
        chosen.append(obj)
        covered |= edges
    if covered.issuperset(target):
        return chosen
    return None

def _has_interior_edge_multiple_uses(st: TrackState) -> bool:
    """
    Return True if some interior edge is crossed more than once.
    """
    by_edge = _interior_crossings_by_edge(st)
    return any(len(crossings) > 1 for crossings in by_edge.values())


def _interior_crossings_by_edge(st: TrackState) -> Dict[int, Set[Tuple[int, ...]]]:
    """
    Group interior crossings by interior edge object id.

    A single geometric crossing contributes a pair of matched ports on the two
    identified sides of an interior edge; we collapse that pair to one key so
    it is counted once (important for annulus N=1 identified left/right edge).
    """
    counts: Dict[int, Set[Tuple[int, ...]]] = {}
    for i in range(st.surface.N):
        diagram = st.diagrams[i]
        for ch in diagram.chords:
            for bp in (ch.a, ch.b):
                e = EdgeRef(bp.side, i)
                if st.surface.is_interior_edge(e):
                    edge_obj = st.surface.square(i).edge(bp.side)
                    edge_id = id(edge_obj)
                    pair = st.surface.paired_port(e, bp.port)
                    if pair is None:
                        crossing_key = (id(bp.port),)
                    else:
                        if isinstance(pair, tuple):
                            p2 = pair[1]
                        else:
                            p2 = pair
                        crossing_key = tuple(sorted((id(bp.port), id(p2))))
                    counts.setdefault(edge_id, set()).add(crossing_key)
    return counts


def _simple_constraints_ok(st: TrackState) -> bool:
    if not st.is_closed():
        return False
    if not _has_any_chord(st):
        return False
    pat = Pattern(surface=st.surface, diagrams=st.diagrams)
    if not pat.validate():
        return False
    if st.turn_count % 2 != 0:
        return False
    if st.dy == 0:
        return False
    if st.or_pair_count % 2 != 0:
        return False
    if not _dx_all_pos_or_all_neg(st):
        return False
    return True


def simple_shortcut_bfs(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_patterns: int = 1,
    max_ports_per_edge: int | None = 5,
    debug: bool = False,
) -> list[Pattern]:
    """
    Convenience wrapper with strict flags for simple shortcuts.
    """
    return find_pattern_bfs(
        surface,
        max_nodes=max_nodes,
        max_patterns=max_patterns,
        max_ports_per_edge=max_ports_per_edge,
        even_turning=True,
        dy_nonzero=True,
        even_or_pair_count=True,
        one_dir_only=True,
        multiple_interior_edge_crossings=False,
        dx_all_pos_or_all_neg=True,
        debug=debug,
    )


def candidate_simple_shortcut_bfs(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_patterns: int = 1,
    max_ports_per_edge: int | None = 5,
    debug: bool = False,
) -> list[Pattern]:
    """
    Wrapper for candidate simple shortcuts (like simple_shortcut_bfs but without one_dir_only).
    """
    return find_pattern_bfs(
        surface,
        max_nodes=max_nodes,
        max_patterns=max_patterns,
        max_ports_per_edge=max_ports_per_edge,
        even_turning=True,
        dy_nonzero=True,
        even_or_pair_count=True,
        one_dir_only=False,
        multiple_interior_edge_crossings=False,
        debug=debug,
    )


def candidate_simple_shortcut_states(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_states: int = 1_000,
    max_ports_per_edge: int | None = 5,
    debug: bool = False,
) -> List[TrackState]:
    """
    Return closed TrackStates that qualify as candidate simple shortcuts.
    """
    found: List[TrackState] = []
    for start_edge in _start_edges(surface):
        st0 = TrackState.initialize(surface, start_edge=start_edge)
        if st0 is None:
            continue

        q: Deque[TrackState] = deque([st0])
        seen: Set[Tuple] = set()
        expanded = 0

        while q:
            st = q.popleft()
            expanded += 1
            if expanded > max_nodes:
                break

            key = _state_key(st)
            if key in seen:
                continue
            seen.add(key)

            pat = Pattern(surface=st.surface, diagrams=st.diagrams)
            if pat.validate() and _has_any_chord(st) and st.is_closed():
                if (st.turn_count % 2 == 0) and (st.dy != 0) and (st.or_pair_count % 2 == 0):
                    found.append(st)
                    if debug:
                        print(f"Candidate found. Total: {len(found)}")
                    if len(found) >= max_states:
                        return found

            moves = st.moves(
                multiple_interior_edge_crossings=False,
                one_dir_only=False,
                max_ports_per_edge=max_ports_per_edge,
            )
            for nxt in moves:
                q.append(nxt)

    return found


def simple_shortcut_states(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_states: int = 1_000,
    max_ports_per_edge: int | None = 5,
    debug: bool = False,
) -> List[TrackState]:
    """
    Return closed TrackStates that qualify as simple shortcuts (one_dir_only=True).
    """
    found: List[TrackState] = []
    for start_edge in _start_edges(surface):
        st0 = TrackState.initialize(surface, start_edge=start_edge)
        if st0 is None:
            continue

        q: Deque[TrackState] = deque([st0])
        seen: Set[Tuple] = set()
        expanded = 0

        while q:
            st = q.popleft()
            expanded += 1
            if expanded > max_nodes:
                break

            key = _state_key(st)
            if key in seen:
                continue
            seen.add(key)

            pat = Pattern(surface=st.surface, diagrams=st.diagrams)
            if pat.validate() and _has_any_chord(st) and st.is_closed():
                if (
                    (st.turn_count % 2 == 0)
                    and (st.dy != 0)
                    and (st.or_pair_count % 2 == 0)
                    and _dx_all_pos_or_all_neg(st)
                ):
                    found.append(st)
                    if debug:
                        print(f"Simple shortcut found. Total: {len(found)}")
                    if len(found) >= max_states:
                        return found

            moves = st.moves(
                multiple_interior_edge_crossings=False,
                one_dir_only=True,
                max_ports_per_edge=max_ports_per_edge,
            )
            for nxt in moves:
                q.append(nxt)

    return found


def _candidate_or_dx_shortcut(
    surface: MarkedSurface,
    *,
    max_nodes: int,
    max_candidates: int,
    max_ports_per_edge: int | None,
    multiple_interior_edge_crossings: bool,
    require_even_turning: bool,
    require_even_or_pairs: bool,
    require_dy_nonzero: bool,
    reject_all_interior_used: bool,
    require_all_marked_used: bool = False,
    longcut_mode: bool = False,
    dominant_dir_only: bool = False,
    cover_marked_edges: bool = False,
    require_nontrivial: bool = False,
    require_dx_sign_for_simple: bool = True,
    debug: bool,
    debug_counts: Optional[dict] = None,
    progress: bool = False,
    progress_interval: int = 200,
    progress_hook=None,
    stream_hook=None,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
    trace_color: str = "",
    trace_reset: str = "",
    trace_accepted: bool = False,
    trace_accepted_max: int | None = None,
    trace_accepted_color: str = "",
) -> tuple[TrackState | None, List[TrackState]]:
    """
    Run candidate BFS; return first candidate with dx sign condition, plus all candidates found.
    """
    found: List[TrackState] = []
    starts = _start_edges(surface, allow_bottom=not require_dy_nonzero)
    for start_idx, start_edge in enumerate(starts, start=1):
        st0 = TrackState.initialize(surface, start_edge=start_edge)
        if st0 is None:
            continue

        parents: Dict[Tuple, Tuple[Optional[Tuple], TrackState]] = {}
        if trace_accepted:
            parents[_state_key(st0)] = (None, st0)

        if trace_steps:
            print(
                f"{trace_color}=== Start edge {start_idx}/{len(starts)}: "
                f"{start_edge.side.name.lower()}@{start_edge.i} ==={trace_reset}"
            )
            print()

        q: Deque[TrackState] = deque([st0])
        seen: Set[Tuple] = set()
        expanded = 0  # dequeued states
        generated = 0  # total enqueued successors

        while q:
            st = q.popleft()
            expanded += 1
            if progress and (expanded == 1 or expanded % max(progress_interval, 1) == 0):
                width = 24
                frac = min(expanded / max_nodes, 1.0) if max_nodes else 0.0
                filled = int(width * frac)
                bar = "#" * filled + "-" * (width - filled)
                msg = (
                    f"start {start_idx}/{len(starts)} "
                    f"[{bar}] {expanded}/{max_nodes} "
                    f"queue {len(q)} seen {len(seen)}"
                )
                if progress_hook is not None:
                    progress_hook(
                        start_idx=start_idx,
                        starts_total=len(starts),
                        expanded=expanded,
                        max_nodes=max_nodes,
                        queue_len=len(q),
                        seen_len=len(seen),
                        msg=msg,
                    )
                else:
                    sys.stdout.write("\r" + msg)
                    sys.stdout.flush()
            if expanded > max_nodes:
                break

            key = _state_key(st)
            if key in seen:
                continue
            seen.add(key)

            if stream_hook is not None:
                sq_i, bp = st.cursor
                stream_hook(
                    {
                        "type": "step",
                        "start_idx": start_idx,
                        "starts_total": len(starts),
                        "expanded": expanded,
                        "max_nodes": max_nodes,
                        "queue": len(q),
                        "seen": len(seen),
                        "generated": generated,
                        "cursor": f"{bp.side.name.lower()}@{sq_i}",
                        "dy": st.dy,
                        "dx": st.dx_linear_form(pretty=True),
                        "turns": st.turn_count,
                        "or_pairs": st.or_pair_count,
                        "render": st.render(),
                    }
                )

            trace_active = trace_steps and (trace_max_steps is None or expanded <= trace_max_steps)
            if trace_active:
                print(f"{trace_color}Step {expanded} (max {max_nodes}):{trace_reset}")
                print(
                    f"{trace_color}"
                    f"queue={len(q)} seen={len(seen)} generated={generated}"
                    f"{trace_reset}"
                )
                print(f"{trace_color}{st.render()}{trace_reset}")
                print()

            pat = Pattern(surface=st.surface, diagrams=st.diagrams)
            if pat.validate() and _has_any_chord(st) and st.is_closed():
                ok = True
                reasons: List[str] = []
                if require_even_turning:
                    if st.turn_count % 2 != 0:
                        ok = False
                        reasons.append("turn_count is odd")
                if require_even_or_pairs:
                    if st.or_pair_count % 2 != 0:
                        ok = False
                        reasons.append("or_pair_count is odd")
                if require_dy_nonzero:
                    if st.dy == 0:
                        ok = False
                        reasons.append("dy is zero")
                if reject_all_interior_used:
                    if _uses_all_interior_edges(st):
                        ok = False
                        reasons.append("uses all interior edges")
                if require_all_marked_used and not cover_marked_edges:
                    if not _uses_all_marked_edges(st):
                        ok = False
                        reasons.append("not all marked edges are used")
                if longcut_mode:
                    if not _uses_all_interior_edges(st):
                        ok = False
                        reasons.append("longcut: not all interior edges used")
                    if not _has_interior_edge_multiple_uses(st):
                        ok = False
                        reasons.append("longcut: no interior edge has multiple uses")
                if ok:
                    found.append(st)
                    immediate_ok = (
                        (not require_dx_sign_for_simple or _dx_all_pos_or_all_neg(st))
                        and not cover_marked_edges
                        and not require_nontrivial
                    )
                    if immediate_ok:
                        if stream_hook is not None:
                            stream_hook(
                                {
                                    "type": "closed_accepted",
                                    "expanded": expanded,
                                    "dy": st.dy,
                                    "dx": st.dx_linear_form(pretty=True),
                                    "turns": st.turn_count,
                                    "or_pairs": st.or_pair_count,
                                    "render": st.render(),
                                }
                            )
                        if progress and progress_hook is None:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        if trace_accepted:
                            path_states: list[TrackState] = []
                            cur_key: Optional[Tuple] = _state_key(st)
                            while cur_key is not None and cur_key in parents:
                                prev_key, cur_state = parents[cur_key]
                                path_states.append(cur_state)
                                cur_key = prev_key
                            path_states.reverse()
                            color = trace_accepted_color or trace_color
                            print(
                                f"{color}Pattern accepted. "
                                f"Path length: {len(path_states)}{trace_reset}"
                            )
                            for i, st_path in enumerate(path_states, start=1):
                                if trace_accepted_max is not None and i > trace_accepted_max:
                                    print(f"{color}...{trace_reset}")
                                    break
                                sq_i, bp = st_path.cursor
                                print(
                                    f"{color}Step {i}: cursor={bp.side.name.lower()}@{sq_i} "
                                    f"turns={st_path.turn_count} dy={st_path.dy} "
                                    f"or_pairs={st_path.or_pair_count} dom_x={st_path.dominant_x_dir}{trace_reset}"
                                )
                                print(f"{color}{st_path.render()}{trace_reset}")
                                print()
                        return st, found
                    else:
                        extra_reasons: List[str] = []
                        if cover_marked_edges:
                            extra_reasons.append("collecting candidates for marked-edge cover")
                        if require_nontrivial and not _nontrivial_simple_ok(st):
                            extra_reasons.append("nontrivial-simple condition failed (need dy!=0 and even turns)")
                        if (
                            not require_nontrivial
                            and require_dx_sign_for_simple
                            and not _dx_all_pos_or_all_neg(st)
                        ):
                            extra_reasons.append("dx coefficients are not all same sign")
                        if stream_hook is not None:
                            stream_hook(
                                {
                                    "type": "closed_rejected",
                                    "expanded": expanded,
                                    "dy": st.dy,
                                    "dx": st.dx_linear_form(pretty=True),
                                    "turns": st.turn_count,
                                    "or_pairs": st.or_pair_count,
                                    "reasons": extra_reasons or ["candidate accepted but deferred"],
                                    "render": st.render(),
                                }
                            )
                    if debug:
                        print(f"Candidate found. Total: {len(found)}")
                    if len(found) >= max_candidates:
                        if progress and progress_hook is None:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        return None, found
                else:
                    if stream_hook is not None:
                        stream_hook(
                            {
                                "type": "closed_rejected",
                                "expanded": expanded,
                                "dy": st.dy,
                                "dx": st.dx_linear_form(pretty=True),
                                "turns": st.turn_count,
                                "or_pairs": st.or_pair_count,
                                "reasons": reasons or ["failed acceptance constraints"],
                                "render": st.render(),
                            }
                        )

            moves = st.moves(
                multiple_interior_edge_crossings=multiple_interior_edge_crossings,
                one_dir_only=False,
                dominant_dir_only=dominant_dir_only,
                max_ports_per_edge=max_ports_per_edge,
                debug_counts=debug_counts,
            )
            generated += len(moves)
            if trace_active:
                print(f"{trace_color}moves={len(moves)} total_generated={generated}{trace_reset}")
                print()
            for nxt in moves:
                nxt_key = _state_key(nxt)
                if trace_accepted and nxt_key not in parents:
                    parents[nxt_key] = (_state_key(st), nxt)
                q.append(nxt)

        if progress and progress_hook is None:
            sys.stdout.write("\n")
            sys.stdout.flush()

    return None, found


def collect_candidate_states(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_states: int = 1_000,
    max_ports_per_edge: int | None = 5,
    multiple_interior_edge_crossings: bool = False,
    require_even_turning: bool = True,
    require_even_or_pairs: bool = True,
    require_dy_nonzero: bool = True,
    reject_all_interior_used: bool = False,
    require_all_marked_used: bool = False,
    longcut_mode: bool = False,
    dominant_dir_only: bool = False,
    debug: bool = False,
    debug_counts: Optional[dict] = None,
    trace_steps: bool = False,
    trace_step_color: str = "",
    trace_candidate_color: str = "",
    trace_reset: str = "",
    trace_max_steps: int | None = None,
) -> List[TrackState]:
    """
    Collect candidate TrackStates under the candidate constraints.
    """
    found: List[TrackState] = []
    starts = _start_edges(surface, allow_bottom=not require_dy_nonzero)
    for start_idx, start_edge in enumerate(starts, start=1):
        st0 = TrackState.initialize(surface, start_edge=start_edge)
        if st0 is None:
            continue

        if trace_steps:
            print(
                f"{trace_step_color}=== Start edge {start_idx}/{len(starts)}: "
                f"{start_edge.side.name.lower()}@{start_edge.i} ==={trace_reset}"
            )
            print()

        q: Deque[TrackState] = deque([st0])
        parent_key: Dict[Tuple, Optional[Tuple]] = {}
        step_id: Dict[Tuple, int] = {}
        seen: Set[Tuple] = set()
        expanded = 0
        generated = 0

        while q:
            st = q.popleft()
            expanded += 1
            if expanded > max_nodes:
                break

            key = _state_key(st)
            if key in seen:
                continue
            seen.add(key)
            if key not in parent_key:
                parent_key[key] = None
            if key not in step_id:
                step_id[key] = expanded

            trace_active = trace_steps and (trace_max_steps is None or expanded <= trace_max_steps)
            if trace_active:
                pid = parent_key.get(key)
                pid_str = "None" if pid is None else f"Step {step_id.get(pid, '?')}"
                print(
                    f"{trace_step_color}Step {expanded} (max {max_nodes}) "
                    f"(parent {pid_str}):{trace_reset}"
                )
                print(
                    f"{trace_step_color}"
                    f"queue={len(q)} seen={len(seen)} generated={generated}"
                    f"{trace_reset}"
                )
                print(f"{trace_step_color}{st.render()}{trace_reset}")
                print()

            pat = Pattern(surface=st.surface, diagrams=st.diagrams)
            if pat.validate() and _has_any_chord(st) and st.is_closed():
                ok = True
                if require_even_turning:
                    ok = ok and (st.turn_count % 2 == 0)
                if require_even_or_pairs:
                    ok = ok and (st.or_pair_count % 2 == 0)
                if require_dy_nonzero:
                    ok = ok and (st.dy != 0)
                if reject_all_interior_used:
                    ok = ok and (not _uses_all_interior_edges(st))
                if require_all_marked_used:
                    ok = ok and _uses_all_marked_edges(st)
                if longcut_mode:
                    ok = ok and _uses_all_interior_edges(st)
                    ok = ok and _has_interior_edge_multiple_uses(st)
                if ok:
                    found.append(st)
                    if debug:
                        print(f"Candidate found. Total: {len(found)}")
                    if trace_steps and (trace_max_steps is None or expanded <= trace_max_steps):
                        print(f"{trace_candidate_color}Candidate {len(found)} found.{trace_reset}")
                        print(f"{trace_candidate_color}{st.render()}{trace_reset}")
                        print()
                    if len(found) >= max_states:
                        return found

            moves = st.moves(
                multiple_interior_edge_crossings=multiple_interior_edge_crossings,
                one_dir_only=False,
                dominant_dir_only=dominant_dir_only,
                max_ports_per_edge=max_ports_per_edge,
                debug_counts=debug_counts,
            )
            generated += len(moves)
            if trace_active:
                print(f"{trace_step_color}moves={len(moves)} total_generated={generated}{trace_reset}")
                print()
            for nxt in moves:
                k2 = _state_key(nxt)
                if k2 not in parent_key:
                    parent_key[k2] = key
                if k2 not in step_id:
                    step_id[k2] = expanded + 1
                q.append(nxt)

    return found


def simple_shortcut_or_complete_set(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_patterns: int = 1,
    max_candidates: int = 1_000,
    max_ports_per_edge: int | None = 5,
    minimize_seed: int | None = None,
    max_minimize_size: int = 20,
    debug: bool = False,
    progress: bool = False,
    progress_interval: int = 200,
    progress_hook=None,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
) -> TrackState | List[TrackState] | None:
    """
    Run candidate BFS; if any candidate satisfies dx sign condition, return it.
    Otherwise return a minimized complete candidate set (if complete).
    """
    shortcut, candidates = _candidate_or_dx_shortcut(
        surface,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        max_ports_per_edge=max_ports_per_edge,
        multiple_interior_edge_crossings=False,
        require_even_turning=True,
        require_even_or_pairs=True,
        require_dy_nonzero=True,
        debug=debug,
        progress=progress,
        progress_interval=progress_interval,
        progress_hook=progress_hook,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
    )
    if shortcut is not None:
        return shortcut
    if not candidates:
        return None
    if not is_complete(candidates):
        return None
    return reduce_complete_set(
        candidates,
        seed=minimize_seed,
        max_minimize_size=max_minimize_size,
    )


def simple_shortcut_or_complete_set_with_candidates(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_patterns: int = 1,
    max_candidates: int = 1_000,
    max_ports_per_edge: int | None = 5,
    minimize_seed: int | None = None,
    max_minimize_size: int = 20,
    debug: bool = False,
    debug_counts: Optional[dict] = None,
    progress: bool = False,
    progress_interval: int = 200,
    progress_hook=None,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
) -> tuple[TrackState | List[TrackState] | None, List[TrackState]]:
    """
    Like simple_shortcut_or_complete_set, but also returns the candidate list.
    """
    shortcut, candidates = _candidate_or_dx_shortcut(
        surface,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        max_ports_per_edge=max_ports_per_edge,
        multiple_interior_edge_crossings=False,
        require_even_turning=True,
        require_even_or_pairs=True,
        require_dy_nonzero=True,
        debug=debug,
        debug_counts=debug_counts,
        progress=progress,
        progress_interval=progress_interval,
        progress_hook=progress_hook,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
    )
    if shortcut is not None:
        return shortcut, candidates
    if not candidates:
        return None, candidates
    if not is_complete(candidates):
        return None, candidates
    return (
        reduce_complete_set(
            candidates,
            seed=minimize_seed,
            max_minimize_size=max_minimize_size,
        ),
        candidates,
    )


def diagonal_track_or_complete_set(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_patterns: int = 1,
    max_candidates: int = 1_000,
    max_ports_per_edge: int | None = 5,
    minimize_seed: int | None = None,
    max_minimize_size: int = 20,
    debug: bool = False,
    progress: bool = False,
    progress_interval: int = 200,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
) -> TrackState | List[TrackState] | None:
    """
    Candidate-first search without the interior-edge max-port restriction.
    """
    shortcut, candidates = _candidate_or_dx_shortcut(
        surface,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        max_ports_per_edge=max_ports_per_edge,
        multiple_interior_edge_crossings=True,
        require_even_turning=True,
        require_even_or_pairs=True,
        require_dy_nonzero=True,
        debug=debug,
        progress=progress,
        progress_interval=progress_interval,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
    )
    if shortcut is not None:
        return shortcut
    if not candidates:
        return None
    if not is_complete(candidates):
        return None
    return reduce_complete_set(
        candidates,
        seed=minimize_seed,
        max_minimize_size=max_minimize_size,
    )


def diagonal_track_or_complete_set_with_candidates(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_patterns: int = 1,
    max_candidates: int = 1_000,
    max_ports_per_edge: int | None = 5,
    minimize_seed: int | None = None,
    max_minimize_size: int = 20,
    debug: bool = False,
    debug_counts: Optional[dict] = None,
    progress: bool = False,
    progress_interval: int = 200,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
) -> tuple[TrackState | List[TrackState] | None, List[TrackState]]:
    """
    Like diagonal_track_or_complete_set, but also returns the candidate list.
    """
    shortcut, candidates = _candidate_or_dx_shortcut(
        surface,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        max_ports_per_edge=max_ports_per_edge,
        multiple_interior_edge_crossings=True,
        require_even_turning=True,
        require_even_or_pairs=True,
        require_dy_nonzero=True,
        debug=debug,
        debug_counts=debug_counts,
        progress=progress,
        progress_interval=progress_interval,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
    )
    if shortcut is not None:
        return shortcut, candidates
    if not candidates:
        return None, candidates
    if not is_complete(candidates):
        return None, candidates
    return (
        reduce_complete_set(
            candidates,
            seed=minimize_seed,
            max_minimize_size=max_minimize_size,
        ),
        candidates,
    )


def search_shortcut_or_complete_set(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_candidates: int = 1_000,
    max_ports_per_edge: int | None = 5,
    minimize_seed: int | None = None,
    max_minimize_size: int = 20,
    multiple_interior_edge_crossings: bool = True,
    require_even_turning: bool = True,
    require_even_or_pairs: bool = True,
    require_dy_nonzero: bool = True,
    reject_all_interior_used: bool = False,
    require_all_marked_used: bool = False,
    longcut_mode: bool = False,
    dominant_dir_only: bool = False,
    allow_complete_set: bool = True,
    require_nontrivial: bool = False,
    debug: bool = False,
    progress: bool = False,
    progress_interval: int = 200,
    progress_hook=None,
    stream_hook=None,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
    trace_accepted: bool = False,
    trace_accepted_max: int | None = None,
    trace_accepted_color: str = "",
    trace_reset: str = "",
) -> TrackState | List[TrackState] | None:
    """
    Generic search with configurable constraints and optional completeness check.
    """
    allow_complete = allow_complete_set or require_nontrivial
    require_dx_sign_for_simple = allow_complete_set and not require_nontrivial
    shortcut, candidates = _candidate_or_dx_shortcut(
        surface,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        max_ports_per_edge=max_ports_per_edge,
        multiple_interior_edge_crossings=multiple_interior_edge_crossings,
        require_even_turning=require_even_turning,
        require_even_or_pairs=require_even_or_pairs,
        require_dy_nonzero=require_dy_nonzero,
        reject_all_interior_used=reject_all_interior_used,
        require_all_marked_used=require_all_marked_used,
        longcut_mode=longcut_mode,
        dominant_dir_only=dominant_dir_only,
        cover_marked_edges=require_all_marked_used,
        require_nontrivial=require_nontrivial,
        require_dx_sign_for_simple=require_dx_sign_for_simple,
        debug=debug,
        progress=progress,
        progress_interval=progress_interval,
        progress_hook=progress_hook,
        stream_hook=stream_hook,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
        trace_accepted=trace_accepted,
        trace_accepted_max=trace_accepted_max,
        trace_accepted_color=trace_accepted_color,
        trace_reset=trace_reset,
    )
    if require_nontrivial:
        simple_candidates = [st for st in candidates if _nontrivial_simple_ok(st)]
    elif require_dx_sign_for_simple:
        simple_candidates = [st for st in candidates if _dx_all_pos_or_all_neg(st)]
    else:
        simple_candidates = list(candidates)

    if not require_all_marked_used:
        if simple_candidates:
            return simple_candidates[0]
        if not allow_complete:
            return None
        if not candidates:
            return None
        if require_nontrivial:
            if not _nontrivial_complete_ok(candidates):
                return None
        else:
            if not is_complete(candidates):
                return None
        return reduce_complete_set(
            candidates,
            seed=minimize_seed,
            max_minimize_size=max_minimize_size,
        )

    marked_edges = set(surface.marked_edges())
    if not marked_edges:
        if shortcut is not None:
            return shortcut
        if allow_complete and candidates and (
            (not require_nontrivial and is_complete(candidates))
            or (require_nontrivial and _nontrivial_complete_ok(candidates))
        ):
            return reduce_complete_set(
                candidates,
                seed=minimize_seed,
                max_minimize_size=max_minimize_size,
            )
        return None

    items: List[tuple[set[BoundaryEdge], object]] = []
    for st in simple_candidates:
        items.append((_marked_edges_used(st), st))

    if allow_complete and candidates:
        # Add a global complete set if it exists.
        if (not require_nontrivial and is_complete(candidates)) or (
            require_nontrivial and _nontrivial_complete_ok(candidates)
        ):
            reduced = reduce_complete_set(
                candidates,
                seed=minimize_seed,
                max_minimize_size=max_minimize_size,
            )
            guaranteed = _guaranteed_marked_edges(reduced)
            if guaranteed:
                items.append((guaranteed, reduced))

        # Add per-marked-edge complete sets (candidates that all use that edge).
        for edge in marked_edges:
            edge_candidates = [st for st in candidates if edge in _marked_edges_used(st)]
            if not edge_candidates:
                continue
            if require_nontrivial:
                if not _nontrivial_complete_ok(edge_candidates):
                    continue
            elif not is_complete(edge_candidates):
                continue
            reduced = reduce_complete_set(
                edge_candidates,
                seed=minimize_seed,
                max_minimize_size=max_minimize_size,
            )
            guaranteed = _guaranteed_marked_edges(reduced)
            if edge in guaranteed:
                items.append((guaranteed, reduced))

    cover = _minimal_cover(items, marked_edges)
    if cover is None:
        return None
    return cover


def search_shortcut_or_complete_set_with_candidates(
    surface: MarkedSurface,
    *,
    max_nodes: int = 50_000,
    max_candidates: int = 1_000,
    max_ports_per_edge: int | None = 5,
    minimize_seed: int | None = None,
    max_minimize_size: int = 20,
    multiple_interior_edge_crossings: bool = True,
    require_even_turning: bool = True,
    require_even_or_pairs: bool = True,
    require_dy_nonzero: bool = True,
    reject_all_interior_used: bool = False,
    require_all_marked_used: bool = False,
    longcut_mode: bool = False,
    dominant_dir_only: bool = False,
    allow_complete_set: bool = True,
    require_nontrivial: bool = False,
    debug: bool = False,
    debug_counts: Optional[dict] = None,
    progress: bool = False,
    progress_interval: int = 200,
    progress_hook=None,
    stream_hook=None,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
    trace_accepted: bool = False,
    trace_accepted_max: int | None = None,
    trace_accepted_color: str = "",
    trace_reset: str = "",
) -> tuple[TrackState | List[TrackState] | None, List[TrackState]]:
    """
    Generic search with candidates returned for diagnostics.
    """
    allow_complete = allow_complete_set or require_nontrivial
    require_dx_sign_for_simple = allow_complete_set and not require_nontrivial
    shortcut, candidates = _candidate_or_dx_shortcut(
        surface,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        max_ports_per_edge=max_ports_per_edge,
        multiple_interior_edge_crossings=multiple_interior_edge_crossings,
        require_even_turning=require_even_turning,
        require_even_or_pairs=require_even_or_pairs,
        require_dy_nonzero=require_dy_nonzero,
        reject_all_interior_used=reject_all_interior_used,
        require_all_marked_used=require_all_marked_used,
        longcut_mode=longcut_mode,
        dominant_dir_only=dominant_dir_only,
        cover_marked_edges=require_all_marked_used,
        require_nontrivial=require_nontrivial,
        require_dx_sign_for_simple=require_dx_sign_for_simple,
        debug=debug,
        debug_counts=debug_counts,
        progress=progress,
        progress_interval=progress_interval,
        progress_hook=progress_hook,
        stream_hook=stream_hook,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
        trace_accepted=trace_accepted,
        trace_accepted_max=trace_accepted_max,
        trace_accepted_color=trace_accepted_color,
        trace_reset=trace_reset,
    )
    if require_nontrivial:
        simple_candidates = [st for st in candidates if _nontrivial_simple_ok(st)]
    elif require_dx_sign_for_simple:
        simple_candidates = [st for st in candidates if _dx_all_pos_or_all_neg(st)]
    else:
        simple_candidates = list(candidates)

    if not require_all_marked_used:
        if simple_candidates:
            return simple_candidates[0], candidates
        if not allow_complete:
            return None, candidates
        if not candidates:
            return None, candidates
        if require_nontrivial:
            if not _nontrivial_complete_ok(candidates):
                return None, candidates
        else:
            if not is_complete(candidates):
                return None, candidates
        return (
            reduce_complete_set(
                candidates,
                seed=minimize_seed,
                max_minimize_size=max_minimize_size,
            ),
            candidates,
        )

    marked_edges = set(surface.marked_edges())
    if not marked_edges:
        if shortcut is not None:
            return shortcut, candidates
        if allow_complete and candidates and (
            (not require_nontrivial and is_complete(candidates))
            or (require_nontrivial and _nontrivial_complete_ok(candidates))
        ):
            return (
                reduce_complete_set(
                    candidates,
                    seed=minimize_seed,
                    max_minimize_size=max_minimize_size,
                ),
                candidates,
            )
        return None, candidates

    items: List[tuple[set[BoundaryEdge], object]] = []
    for st in simple_candidates:
        items.append((_marked_edges_used(st), st))

    if allow_complete and candidates:
        if (not require_nontrivial and is_complete(candidates)) or (
            require_nontrivial and _nontrivial_complete_ok(candidates)
        ):
            reduced = reduce_complete_set(
                candidates,
                seed=minimize_seed,
                max_minimize_size=max_minimize_size,
            )
            guaranteed = _guaranteed_marked_edges(reduced)
            if guaranteed:
                items.append((guaranteed, reduced))

        for edge in marked_edges:
            edge_candidates = [st for st in candidates if edge in _marked_edges_used(st)]
            if not edge_candidates:
                continue
            if require_nontrivial:
                if not _nontrivial_complete_ok(edge_candidates):
                    continue
            elif not is_complete(edge_candidates):
                continue
            reduced = reduce_complete_set(
                edge_candidates,
                seed=minimize_seed,
                max_minimize_size=max_minimize_size,
            )
            guaranteed = _guaranteed_marked_edges(reduced)
            if edge in guaranteed:
                items.append((guaranteed, reduced))

    cover = _minimal_cover(items, marked_edges)
    if cover is None:
        return None, candidates
    return cover, candidates


def diagnose_simple_shortcut(
    surface: MarkedSurface,
    candidate: TrackState,
    *,
    max_nodes: int = 50_000,
    max_ports_per_edge: int | None = 5,
) -> str:
    """
    Diagnose why a candidate was not found as a simple shortcut.
    """
    if not _simple_constraints_ok(candidate):
        return "candidate fails simple shortcut constraints"

    target_key = _state_key(candidate)
    for start_edge in _start_edges(surface, allow_bottom=False):
        st0 = TrackState.initialize(surface, start_edge=start_edge)
        if st0 is None:
            continue

        q: Deque[TrackState] = deque([st0])
        seen: Set[Tuple] = set()
        expanded = 0

        while q:
            st = q.popleft()
            expanded += 1
            if expanded > max_nodes:
                break
            key = _state_key(st)
            if key in seen:
                continue
            seen.add(key)
            if key == target_key:
                return "reachable under one_dir_only"
            moves = st.moves(
                multiple_interior_edge_crossings=False,
                one_dir_only=True,
                max_ports_per_edge=max_ports_per_edge,
            )
            for nxt in moves:
                q.append(nxt)

    return "not reachable under one_dir_only"

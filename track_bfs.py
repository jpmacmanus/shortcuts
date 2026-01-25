"""
track_bfs.py

Primitive BFS search for a valid pattern on a marked strip/annulus.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import sys
from typing import Deque, Dict, List, Optional, Set, Tuple, Union

from chord_diagram import ChordDiagram
from marked_strips import EdgeRef, MarkedAnnulus, MarkedStrip
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
    # Detect whether every interior edge has been used by some chord endpoint.
    all_interior: set[int] = set()
    for e in st.surface.all_edge_refs():
        if not st.surface.is_interior_edge(e):
            continue
        edge_obj = st.surface.square(e.i).edge(e.side)
        all_interior.add(id(edge_obj))
    if not all_interior:
        return False

    used_interior: set[int] = set()
    for i in range(st.surface.N):
        diagram = st.diagrams[i]
        for ch in diagram.chords:
            for bp in (ch.a, ch.b):
                e = EdgeRef(bp.side, i)
                if st.surface.is_interior_edge(e):
                    edge_obj = st.surface.square(i).edge(bp.side)
                    used_interior.add(id(edge_obj))
    return used_interior.issuperset(all_interior)


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
    dominant_dir_only: bool = False,
    debug: bool,
    debug_counts: Optional[dict] = None,
    progress: bool = False,
    progress_interval: int = 200,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
    trace_color: str = "",
    trace_reset: str = "",
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
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()
            if expanded > max_nodes:
                break

            key = _state_key(st)
            if key in seen:
                continue
            seen.add(key)

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
                if require_even_turning:
                    ok = ok and (st.turn_count % 2 == 0)
                if require_even_or_pairs:
                    ok = ok and (st.or_pair_count % 2 == 0)
                if require_dy_nonzero:
                    ok = ok and (st.dy != 0)
                if reject_all_interior_used:
                    ok = ok and (not _uses_all_interior_edges(st))
                if ok:
                    found.append(st)
                    if _dx_all_pos_or_all_neg(st):
                        if progress:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        return st, found
                    if debug:
                        print(f"Candidate found. Total: {len(found)}")
                    if len(found) >= max_candidates:
                        if progress:
                            sys.stdout.write("\n")
                            sys.stdout.flush()
                        return None, found

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
                q.append(nxt)

        if progress:
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
    dominant_dir_only: bool = False,
    allow_complete_set: bool = True,
    debug: bool = False,
    progress: bool = False,
    progress_interval: int = 200,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
) -> TrackState | List[TrackState] | None:
    """
    Generic search with configurable constraints and optional completeness check.
    """
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
        dominant_dir_only=dominant_dir_only,
        debug=debug,
        progress=progress,
        progress_interval=progress_interval,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
    )
    if shortcut is not None:
        return shortcut
    if not allow_complete_set:
        return None
    if not candidates:
        return None
    if not is_complete(candidates):
        return None
    return reduce_complete_set(
        candidates,
        seed=minimize_seed,
        max_minimize_size=max_minimize_size,
    )


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
    dominant_dir_only: bool = False,
    allow_complete_set: bool = True,
    debug: bool = False,
    debug_counts: Optional[dict] = None,
    progress: bool = False,
    progress_interval: int = 200,
    trace_steps: bool = False,
    trace_max_steps: int | None = None,
) -> tuple[TrackState | List[TrackState] | None, List[TrackState]]:
    """
    Generic search with candidates returned for diagnostics.
    """
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
        dominant_dir_only=dominant_dir_only,
        debug=debug,
        debug_counts=debug_counts,
        progress=progress,
        progress_interval=progress_interval,
        trace_steps=trace_steps,
        trace_max_steps=trace_max_steps,
    )
    if shortcut is not None:
        return shortcut, candidates
    if not allow_complete_set:
        return None, candidates
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

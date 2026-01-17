"""
exhaustive_search.py

Run an exhaustive search over a Klein signature and report shortcuts.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterable, List, Optional, Tuple

from klein_signature_generator import (
    build_surface_from_signature,
    canonical_perm_under_symmetry,
    parse_klein_signature,
    surfaces_from_signature,
    unique_perms_under_symmetry,
)
from track_bfs import (
    collect_candidate_states,
    diagnose_simple_shortcut,
    search_shortcut_or_complete_set,
    search_shortcut_or_complete_set_with_candidates,
)

# ----------------------------
# Config
# ----------------------------

SIGNATURE = "I R V H I"
SURFACE = "annulus"  # "annulus" or "strip"
UNIQUE = True
EXCLUDE_ADJACENT_I = True
PREFIX_PRUNING = True
START_PREFIX_LENGTH = 2
REQUIRE_DY_NONZERO = True
LIMIT_INTERIOR_CROSSINGS = True
REQUIRE_DX_INFEASIBLE = True
REQUIRE_EVEN_TURNING = True
REQUIRE_EVEN_OR_PAIRS = True




# BFS bounds and minimization parameters.
MAX_NODES = 20000
MAX_CANDIDATES = 1_000
MINIMIZE_SEED = None
MAX_MINIMIZE_SIZE = 20
MAX_PORTS_PER_EDGE = 3
DEBUG_UNSOLVED = False
DEBUG_UNSOLVED_MAX = 20
DEBUG_TRACE_BFS = False
DEBUG_TRACE_BFS_MAX = 50
DEBUG_BFS_STEPS = False
DEBUG_BFS_STEPS_MAX = 5000
SHOW_PROGRESS = True
PROGRESS_INTERVAL = 200

# ANSI colors
_COLOR_RESET = "\033[0m"
_COLOR_START = "\033[36m"
_COLOR_FAIL = "\033[31m"
_COLOR_AMBER = "\033[33m"
_COLOR_TRACE = "\033[34m"


# ----------------------------
# Helpers
# ----------------------------


def _print_header(sig: str) -> None:
    print("=== Shortcut search over a Klein signature ===")
    print(f"Signature: {sig}")
    print(f"Surface: {SURFACE}")
    print(f"Unique: {UNIQUE}")
    print(f"Exclude adjacent I: {EXCLUDE_ADJACENT_I}")
    print(f"Prefix pruning: {PREFIX_PRUNING}")
    if PREFIX_PRUNING:
        print(f"Start prefix length: {START_PREFIX_LENGTH}")
    print(f"Require dy nonzero: {REQUIRE_DY_NONZERO}")
    print(f"Limit interior crossings: {LIMIT_INTERIOR_CROSSINGS}")
    print(f"Require dx infeasible: {REQUIRE_DX_INFEASIBLE}")
    print(f"Require even turning: {REQUIRE_EVEN_TURNING}")
    print(f"Require even OR pairs: {REQUIRE_EVEN_OR_PAIRS}")
    print(f"Max ports per edge: {MAX_PORTS_PER_EDGE}")
    print(f"Debug unsolved: {DEBUG_UNSOLVED}")
    if DEBUG_UNSOLVED:
        print(f"Debug unsolved max: {DEBUG_UNSOLVED_MAX}")
        print(f"Debug trace bfs: {DEBUG_TRACE_BFS}")
        if DEBUG_TRACE_BFS:
            print(f"Debug trace bfs max: {DEBUG_TRACE_BFS_MAX}")
    print(f"Debug bfs steps: {DEBUG_BFS_STEPS}")
    if DEBUG_BFS_STEPS:
        print(f"Debug bfs steps max: {DEBUG_BFS_STEPS_MAX}")
    print(f"Show progress: {SHOW_PROGRESS}")
    if SHOW_PROGRESS:
        print(f"Progress interval: {PROGRESS_INTERVAL}")
    print()


def _render_result(
    idx: int,
    surface,
    result: Optional[object] = None,
    candidates: Optional[List[object]] = None,
    debug_counts: Optional[dict] = None,
    *,
    label: Optional[str] = None,
) -> tuple[int, int, int, int]:
    """
    Render a single search result and return (simple, complete, unsolved, max_size).
    """
    simple_count = 0
    complete_count = 0
    unsolved_count = 0
    max_complete_size = 0

    print("=" * 80)
    if label is None:
        label = "Annulus" if SURFACE == "annulus" else "Strip"
    print(f"{label} [{idx}]")
    print(f"{_COLOR_START}{surface}{_COLOR_RESET}")

    if result is None:
        if DEBUG_UNSOLVED:
            if debug_counts is None:
                debug_counts = {}
            result, candidates = search_shortcut_or_complete_set_with_candidates(
                surface,
                max_nodes=MAX_NODES,
                max_candidates=MAX_CANDIDATES,
                max_ports_per_edge=MAX_PORTS_PER_EDGE,
                minimize_seed=MINIMIZE_SEED,
                max_minimize_size=MAX_MINIMIZE_SIZE,
                multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                require_even_turning=REQUIRE_EVEN_TURNING,
                require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                require_dy_nonzero=REQUIRE_DY_NONZERO,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                debug_counts=debug_counts,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=DEBUG_BFS_STEPS,
                trace_max_steps=DEBUG_BFS_STEPS_MAX,
            )
        else:
            result = search_shortcut_or_complete_set(
                surface,
                max_nodes=MAX_NODES,
                max_candidates=MAX_CANDIDATES,
                max_ports_per_edge=MAX_PORTS_PER_EDGE,
                minimize_seed=MINIMIZE_SEED,
                max_minimize_size=MAX_MINIMIZE_SIZE,
                multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                require_even_turning=REQUIRE_EVEN_TURNING,
                require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                require_dy_nonzero=REQUIRE_DY_NONZERO,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=DEBUG_BFS_STEPS,
                trace_max_steps=DEBUG_BFS_STEPS_MAX,
            )

    if result is None:
        if REQUIRE_DX_INFEASIBLE:
            msg = "No simple shortcut or complete set found."
        else:
            msg = "No simple shortcut found."
        print(f"{_COLOR_FAIL}{msg}{_COLOR_RESET}")
        if DEBUG_UNSOLVED:
            if candidates is None:
                debug_counts = {}
                candidates = collect_candidate_states(
                    surface,
                    max_nodes=MAX_NODES,
                    max_states=MAX_CANDIDATES,
                    multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                    require_even_turning=REQUIRE_EVEN_TURNING,
                    require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                    require_dy_nonzero=REQUIRE_DY_NONZERO,
                    debug=False,
                    debug_counts=debug_counts,
                    trace_steps=DEBUG_TRACE_BFS,
                    trace_step_color=_COLOR_TRACE,
                    trace_candidate_color=_COLOR_FAIL,
                    trace_reset=_COLOR_RESET,
                    trace_max_steps=DEBUG_TRACE_BFS_MAX,
                )
            for c_idx, st in enumerate(candidates[:DEBUG_UNSOLVED_MAX], start=1):
                print(f"{_COLOR_FAIL}Candidate [{c_idx}]:{_COLOR_RESET}")
                print(
                    f"{_COLOR_FAIL}dy: {st.dy}  dx: {st.dx_linear_form(pretty=True)}{_COLOR_RESET}"
                )
                print(f"{_COLOR_FAIL}{st.render()}{_COLOR_RESET}")
                print()
            multi_cross_count = 0
            for st in candidates:
                if _count_interior_edges_with_multiple_ports(st) > 0:
                    multi_cross_count += 1
            print(
                f"{_COLOR_FAIL}Candidates with interior edges crossed >1 time: "
                f"{multi_cross_count}{_COLOR_RESET}"
            )
            if debug_counts is not None:
                print(
                    f"{_COLOR_FAIL}Interior add attempts: "
                    f"{debug_counts.get('interior_add_attempts', 0)}{_COLOR_RESET}"
                )
                print(
                    f"{_COLOR_FAIL}Interior existing-port blocks: "
                    f"{debug_counts.get('interior_existing_candidates', 0)}{_COLOR_RESET}"
                )
                print(
                    f"{_COLOR_FAIL}Interior add success: "
                    f"{debug_counts.get('interior_add_success', 0)}{_COLOR_RESET}"
                )
                print(
                    f"{_COLOR_FAIL}Interior add fail (add_port_between): "
                    f"{debug_counts.get('interior_add_fail_add_port', 0)}{_COLOR_RESET}"
                )
                print(
                    f"{_COLOR_FAIL}Interior add fail (add_chord): "
                    f"{debug_counts.get('interior_add_fail_add_chord', 0)}{_COLOR_RESET}"
                )
                print(
                    f"{_COLOR_FAIL}Interior add fail (pair): "
                    f"{debug_counts.get('interior_add_fail_pair', 0)}{_COLOR_RESET}"
                )
        print()
        return (0, 0, 1, 0)

    if isinstance(result, list):
        print(
            f"{_COLOR_AMBER}Complete candidate set found "
            f"(size {len(result)}).{_COLOR_RESET}"
        )
        complete_count = 1
        max_complete_size = len(result)
        for r_idx, st in enumerate(result, start=1):
            print(f"  [{r_idx}] dy: {st.dy}  dx: {st.dx_linear_form(pretty=True)}")
            print(st.render())
            print()
        if MAX_PORTS_PER_EDGE is not None:
            max_ports = max(_max_ports_on_any_edge(st.surface) for st in result)
            if max_ports > MAX_PORTS_PER_EDGE:
                print(
                    f"{_COLOR_AMBER}Warning: max ports per edge observed "
                    f"{max_ports} > {MAX_PORTS_PER_EDGE}{_COLOR_RESET}"
                )
        if len(result) == 1:
            reason = diagnose_simple_shortcut(surface, result[0])
            print(f"{_COLOR_AMBER}Single-candidate diagnostic:{_COLOR_RESET} {reason}")
        print()
        return (0, complete_count, 0, max_complete_size)

    print("Simple shortcut found (dx sign).")
    print(result.render())
    print(f"dy: {result.dy}")
    print(f"dx: {result.dx_linear_form(pretty=True)}")
    if MAX_PORTS_PER_EDGE is not None:
        max_ports = _max_ports_on_any_edge(result.surface)
        if max_ports > MAX_PORTS_PER_EDGE:
            print(
                f"{_COLOR_AMBER}Warning: max ports per edge observed "
                f"{max_ports} > {MAX_PORTS_PER_EDGE}{_COLOR_RESET}"
            )
    print()
    simple_count = 1
    return (simple_count, 0, 0, 0)


def _print_diagnostics(
    total: int,
    simple_count: int,
    complete_count: int,
    unsolved_count: int,
    max_complete_size: int,
) -> None:
    print("=" * 80)
    print("=== Diagnostics ===")
    print(f"{_COLOR_START}Total cases:{_COLOR_RESET} {total}")
    print(f"{_COLOR_START}Solved by simple shortcut:{_COLOR_RESET} {simple_count}")
    print(f"{_COLOR_AMBER}Solved by complete candidate set:{_COLOR_RESET} {complete_count}")
    print(f"{_COLOR_FAIL}Unsolved:{_COLOR_RESET} {unsolved_count}")
    if complete_count:
        print(
            f"{_COLOR_AMBER}Max complete set size:{_COLOR_RESET} {max_complete_size}"
        )
    else:
        print(f"{_COLOR_AMBER}Max complete set size:{_COLOR_RESET} n/a")


def _insert_label_everywhere(base: Tuple[int, ...], new_label: int) -> Iterable[Tuple[int, ...]]:
    m = len(base)
    for pos in range(m + 1):
        yield base[:pos] + (new_label,) + base[pos:]


def _enumerate_stage_perms(m: int) -> List[Tuple[int, ...]]:
    if UNIQUE:
        return list(unique_perms_under_symmetry(m, surface=SURFACE))
    return list(permutations(range(m)))


def _perm_has_adjacent_I(sig, perm, *, surface: str) -> bool:
    n = len(sig)
    is_I = [s.name == "I" for s in sig]
    for i in range(n):
        j = (i + 1) % n
        if surface == "strip" and i == n - 1:
            break
        if is_I[perm[i]] and is_I[perm[j]]:
            return True
    return False


def _count_interior_edges_with_multiple_ports(st) -> int:
    count = 0
    seen: set[int] = set()
    for e in st.surface.all_edge_refs():
        if not st.surface.is_interior_edge(e):
            continue
        edge_obj = st.surface.square(e.i).edge(e.side)
        edge_id = id(edge_obj)
        if edge_id in seen:
            continue
        seen.add(edge_id)
        if len(list(edge_obj.ports())) > 1:
            count += 1
    return count


def _max_ports_on_any_edge(surface) -> int:
    max_ports = 0
    seen: set[int] = set()
    for e in surface.all_edge_refs():
        edge_obj = surface.square(e.i).edge(e.side)
        edge_id = id(edge_obj)
        if edge_id in seen:
            continue
        seen.add(edge_id)
        max_ports = max(max_ports, len(list(edge_obj.ports())))
    return max_ports


def _prefix_pruning_search(sig_full) -> tuple[int, int, int, int, int]:
    n = len(sig_full)
    if START_PREFIX_LENGTH < 1 or START_PREFIX_LENGTH > n:
        raise ValueError("START_PREFIX_LENGTH must satisfy 1 <= start <= len(signature)")

    candidates = _enumerate_stage_perms(START_PREFIX_LENGTH)
    basket: set[Tuple[int, ...]] = set()

    sig_prefix = sig_full[:START_PREFIX_LENGTH]
    print("=" * 80)
    print(f"Stage {START_PREFIX_LENGTH}/{n} prefix={list(sig_prefix)}")
    print(f"Frontier size: {len(candidates)}")
    for idx, perm in enumerate(candidates, start=1):
        surface_obj = build_surface_from_signature(sig_prefix, perm, surface=SURFACE)
        debug_counts: dict = {}
        if DEBUG_UNSOLVED:
            result, candidates = search_shortcut_or_complete_set_with_candidates(
                surface_obj,
                max_nodes=MAX_NODES,
                max_candidates=MAX_CANDIDATES,
                max_ports_per_edge=MAX_PORTS_PER_EDGE,
                minimize_seed=MINIMIZE_SEED,
                max_minimize_size=MAX_MINIMIZE_SIZE,
                multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                require_even_turning=REQUIRE_EVEN_TURNING,
                require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                require_dy_nonzero=REQUIRE_DY_NONZERO,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                debug_counts=debug_counts,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=DEBUG_BFS_STEPS,
                trace_max_steps=DEBUG_BFS_STEPS_MAX,
            )
            _render_result(
                idx,
                surface_obj,
                result,
                candidates=candidates,
                debug_counts=debug_counts,
                label=f"Stage {START_PREFIX_LENGTH}",
            )
        else:
            result = search_shortcut_or_complete_set(
                surface_obj,
                max_nodes=MAX_NODES,
                max_candidates=MAX_CANDIDATES,
                max_ports_per_edge=MAX_PORTS_PER_EDGE,
                minimize_seed=MINIMIZE_SEED,
                max_minimize_size=MAX_MINIMIZE_SIZE,
                multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                require_even_turning=REQUIRE_EVEN_TURNING,
                require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                require_dy_nonzero=REQUIRE_DY_NONZERO,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=DEBUG_BFS_STEPS,
                trace_max_steps=DEBUG_BFS_STEPS_MAX,
            )
            _render_result(
                idx,
                surface_obj,
                result,
                label=f"Stage {START_PREFIX_LENGTH}",
            )
        if result is None:
            basket.add(perm)
    print(f"Remaining bad prefixes: {len(basket)}")

    simple_total = 0
    complete_total = 0
    unsolved_total = 0
    max_complete_size = 0
    total_cases = 0

    for m in range(START_PREFIX_LENGTH, n):
        new_label = m
        sig_prefix = sig_full[: m + 1]
        candidates_set: set[Tuple[int, ...]] = set()
        for base in basket:
            for ext in _insert_label_everywhere(base, new_label):
                if UNIQUE:
                    ext = canonical_perm_under_symmetry(ext, surface=SURFACE, n=m + 1)
                candidates_set.add(ext)
        candidates = sorted(candidates_set)
        basket = set()

        is_final = (m + 1 == n)
        print("=" * 80)
        print(f"Stage {m + 1}/{n} prefix={list(sig_prefix)}")
        print(f"Frontier size: {len(candidates)}")
        for idx, perm in enumerate(candidates, start=1):
            if is_final and EXCLUDE_ADJACENT_I:
                if _perm_has_adjacent_I(sig_full, perm, surface=SURFACE):
                    continue
            surface_obj = build_surface_from_signature(sig_prefix, perm, surface=SURFACE)
            debug_counts = {}
            if DEBUG_UNSOLVED:
                result, candidates = search_shortcut_or_complete_set_with_candidates(
                    surface_obj,
                    max_nodes=MAX_NODES,
                    max_candidates=MAX_CANDIDATES,
                    max_ports_per_edge=MAX_PORTS_PER_EDGE,
                    minimize_seed=MINIMIZE_SEED,
                    max_minimize_size=MAX_MINIMIZE_SIZE,
                    multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                    require_even_turning=REQUIRE_EVEN_TURNING,
                    require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                    require_dy_nonzero=REQUIRE_DY_NONZERO,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    debug_counts=debug_counts,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=DEBUG_BFS_STEPS,
                    trace_max_steps=DEBUG_BFS_STEPS_MAX,
                )
            else:
                candidates = None
                result = search_shortcut_or_complete_set(
                    surface_obj,
                    max_nodes=MAX_NODES,
                    max_candidates=MAX_CANDIDATES,
                    max_ports_per_edge=MAX_PORTS_PER_EDGE,
                    minimize_seed=MINIMIZE_SEED,
                    max_minimize_size=MAX_MINIMIZE_SIZE,
                    multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                    require_even_turning=REQUIRE_EVEN_TURNING,
                    require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                    require_dy_nonzero=REQUIRE_DY_NONZERO,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=DEBUG_BFS_STEPS,
                    trace_max_steps=DEBUG_BFS_STEPS_MAX,
                )
            if is_final:
                total_cases += 1
                s, c, u, msize = _render_result(
                    total_cases,
                    surface_obj,
                    result,
                    candidates=candidates,
                    debug_counts=debug_counts,
                )
                simple_total += s
                complete_total += c
                unsolved_total += u
                if msize > max_complete_size:
                    max_complete_size = msize
            else:
                _render_result(
                    idx,
                    surface_obj,
                    result,
                    candidates=candidates,
                    debug_counts=debug_counts,
                    label=f"Stage {m + 1}",
                )
                if result is None:
                    basket.add(perm)
        print(f"Remaining bad prefixes: {len(basket)}")

    return simple_total, complete_total, unsolved_total, max_complete_size, total_cases


def main() -> None:
    _print_header(SIGNATURE)

    simple_total = 0
    complete_total = 0
    unsolved_total = 0
    max_complete_size = 0

    if not PREFIX_PRUNING:
        surfaces = surfaces_from_signature(
            SIGNATURE,
            surface=SURFACE,
            unique=UNIQUE,
            exclude_adjacent_I=EXCLUDE_ADJACENT_I,
        )
        for idx, surface in enumerate(surfaces, start=1):
            debug_counts = {}
            if DEBUG_UNSOLVED:
                result, candidates = search_shortcut_or_complete_set_with_candidates(
                    surface,
                    max_nodes=MAX_NODES,
                    max_candidates=MAX_CANDIDATES,
                    max_ports_per_edge=MAX_PORTS_PER_EDGE,
                    minimize_seed=MINIMIZE_SEED,
                    max_minimize_size=MAX_MINIMIZE_SIZE,
                    multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                    require_even_turning=REQUIRE_EVEN_TURNING,
                    require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                    require_dy_nonzero=REQUIRE_DY_NONZERO,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    debug_counts=debug_counts,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=DEBUG_BFS_STEPS,
                    trace_max_steps=DEBUG_BFS_STEPS_MAX,
                )
                s, c, u, m = _render_result(
                    idx,
                    surface,
                    result,
                    candidates=candidates,
                    debug_counts=debug_counts,
                )
            else:
                result = search_shortcut_or_complete_set(
                    surface,
                    max_nodes=MAX_NODES,
                    max_candidates=MAX_CANDIDATES,
                    max_ports_per_edge=MAX_PORTS_PER_EDGE,
                    minimize_seed=MINIMIZE_SEED,
                    max_minimize_size=MAX_MINIMIZE_SIZE,
                    multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
                    require_even_turning=REQUIRE_EVEN_TURNING,
                    require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
                    require_dy_nonzero=REQUIRE_DY_NONZERO,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=DEBUG_BFS_STEPS,
                    trace_max_steps=DEBUG_BFS_STEPS_MAX,
                )
                s, c, u, m = _render_result(idx, surface, result)
            simple_total += s
            complete_total += c
            unsolved_total += u
            if m > max_complete_size:
                max_complete_size = m
        total = len(surfaces)
    else:
        sig_full = parse_klein_signature(SIGNATURE)
        (
            simple_total,
            complete_total,
            unsolved_total,
            max_complete_size,
            total,
        ) = _prefix_pruning_search(sig_full)

    _print_diagnostics(
        total=total,
        simple_count=simple_total,
        complete_count=complete_total,
        unsolved_count=unsolved_total,
        max_complete_size=max_complete_size,
    )


if __name__ == "__main__":
    main()

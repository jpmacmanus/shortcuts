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
from directed_klein_signature_generator import (
    build_directed_surface_from_signature,
    canonical_perm_under_directed_symmetry,
    directed_surfaces_from_signature,
    unique_perms_under_directed_symmetry,
)
from track_bfs import (
    collect_candidate_states,
    diagnose_simple_shortcut,
    search_shortcut_or_complete_set,
    search_shortcut_or_complete_set_with_candidates,
    trace_shortcut_path,
)

# ----------------------------
# Config
# ----------------------------

# Search input
SIGNATURE = "I I"  # Klein signature to be searched over.
SURFACE = "annulus"        # "annulus" or "strip"
DIRECTED_MODE = False      # make ports directed, to model the 'small splittings' case.
                           # interior edges in the strip/directed case are not directed.

# Generation / symmetry controls
UNIQUE = True               # skips surfaces which differ by a symmetry.
EXCLUDE_ADJACENT_I = False   # skip cases which have trivial solutions due to adjacent I squares.
                            # ignored when DIRECTED_MODE=True.
PREFIX_PRUNING = True       # start search on smaller prefix cases and build up to desired case.
START_PREFIX_LENGTH = 1     # only used when PREFIX_PRUNING=True

# Acceptance constraints
REQUIRE_DY_NONZERO = True        # only accept tracks with dy != 0
REQUIRE_DX_INFEASIBLE = True     # require dx != 0:
                                 #      when raised, will sometimes return multiple candidates 
                                 #      to account for extra unmarked squares.
                                 #      at least one of these candidates is guaranteed to work for 
                                 #      any given assignment of weights.
REQUIRE_EVEN_TURNING = True      # only accept tracks with an even number of turns.
REQUIRE_EVEN_OR_PAIRS = True     # only accept orientable tracks (no Mobius band neighbourhood)
DOMINANT_DIR_ONLY = True        # only move along dominant x-direction (free first move)
LIMIT_INTERIOR_CROSSINGS = False  # when True, cap interior crossings to one per edge
REJECT_ALL_INTERIOR_USED = False  # reject solutions that use every interior edge
LONGCUT_MODE = True            # require all interior edges used, and some used >1

# BFS bounds and minimization parameters
MAX_NODES = 5000
MAX_CANDIDATES = 1_000
MAX_PORTS_PER_EDGE = 3
MINIMIZE_SEED = None
MAX_MINIMIZE_SIZE = 30

# Debug / output control
DEBUG_UNSOLVED = False
DEBUG_UNSOLVED_MAX = 30
DEBUG_TRACE_BFS = False
DEBUG_TRACE_BFS_MAX = 50
DEBUG_BFS_STEPS = False
DEBUG_BFS_STEPS_MAX = 5000
DEBUG_TRACE_ACCEPTED = True
DEBUG_EDGE_PAIRING = True
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
    print(f"Directed mode: {DIRECTED_MODE}")
    print(f"Exclude adjacent I: {EXCLUDE_ADJACENT_I and not DIRECTED_MODE}")
    print(f"Prefix pruning: {PREFIX_PRUNING}")
    if PREFIX_PRUNING:
        print(f"Start prefix length: {START_PREFIX_LENGTH}")
    print(f"Require dy nonzero: {REQUIRE_DY_NONZERO}")
    print(f"Limit interior crossings: {LIMIT_INTERIOR_CROSSINGS}")
    print(f"Require dx infeasible: {REQUIRE_DX_INFEASIBLE}")
    print(f"Require even turning: {REQUIRE_EVEN_TURNING}")
    print(f"Require even OR pairs: {REQUIRE_EVEN_OR_PAIRS}")
    print(f"Dominant dir only: {DOMINANT_DIR_ONLY}")
    print(f"Reject all interior used: {REJECT_ALL_INTERIOR_USED}")
    print(f"Longcut mode: {LONGCUT_MODE}")
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
    print(f"Debug trace accepted: {DEBUG_TRACE_ACCEPTED}")
    print(f"Debug edge pairing: {DEBUG_EDGE_PAIRING}")
    print(f"Show progress: {SHOW_PROGRESS}")
    if SHOW_PROGRESS:
        print(f"Progress interval: {PROGRESS_INTERVAL}")
    print()


def _print_case_header(
    idx: int,
    total: int,
    surface,
    *,
    label: Optional[str] = None,
) -> None:
    print()
    print("=" * 80)
    if label is None:
        label = "Annulus" if SURFACE == "annulus" else "Strip"
    print(f"{label} [{idx} / {total}]")
    print(f"{_COLOR_START}{surface}{_COLOR_RESET}")


def _render_result(
    idx: int,
    total: int,
    surface,
    result: Optional[object] = None,
    candidates: Optional[List[object]] = None,
    debug_counts: Optional[dict] = None,
    skip_search: bool = False,
    *,
    label: Optional[str] = None,
    show_header: bool = True,
) -> tuple[int, int, int, int]:
    """
    Render a single search result and return (simple, complete, unsolved, max_size).
    """
    simple_count = 0
    complete_count = 0
    unsolved_count = 0
    max_complete_size = 0

    if show_header:
        _print_case_header(idx, total, surface, label=label)

    def _trace_unsolved() -> None:
        if not DEBUG_BFS_STEPS:
            return
        search_shortcut_or_complete_set(
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
            reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
            require_all_interior_used=LONGCUT_MODE,
            require_some_interior_multiple=LONGCUT_MODE,
            dominant_dir_only=DOMINANT_DIR_ONLY,
            allow_complete_set=REQUIRE_DX_INFEASIBLE,
            debug=False,
            progress=False,
            trace_steps=True,
            trace_max_steps=DEBUG_BFS_STEPS_MAX,
        )

    def _trace_accepted() -> None:
        if not DEBUG_TRACE_ACCEPTED:
            return
        path_states = trace_shortcut_path(
            surface,
            max_nodes=MAX_NODES,
            max_candidates=MAX_CANDIDATES,
            max_ports_per_edge=MAX_PORTS_PER_EDGE,
            multiple_interior_edge_crossings=not LIMIT_INTERIOR_CROSSINGS,
            require_even_turning=REQUIRE_EVEN_TURNING,
            require_even_or_pairs=REQUIRE_EVEN_OR_PAIRS,
            require_dy_nonzero=REQUIRE_DY_NONZERO,
            reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
            require_all_interior_used=LONGCUT_MODE,
            require_some_interior_multiple=LONGCUT_MODE,
            dominant_dir_only=DOMINANT_DIR_ONLY,
            debug_edge_pairing=DEBUG_EDGE_PAIRING,
        )
        if not path_states:
            return
        print(f"{_COLOR_TRACE}Path length: {len(path_states)}{_COLOR_RESET}")
        for i, st_path in enumerate(path_states, start=1):
            sq_i, bp = st_path.cursor
            print(
                f"{_COLOR_TRACE}"
                f"Step {i}: cursor={bp.side.name.lower()}@{sq_i} "
                f"turns={st_path.turn_count} dy={st_path.dy} "
                f"or_pairs={st_path.or_pair_count} dom_x={st_path.dominant_x_dir}"
                f"{_COLOR_RESET}"
            )
            print(st_path.render())
            print()

    if result is None and not skip_search:
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
                reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                require_all_interior_used=LONGCUT_MODE,
                require_some_interior_multiple=LONGCUT_MODE,
                dominant_dir_only=DOMINANT_DIR_ONLY,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                debug_counts=debug_counts,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=False,
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
                reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                require_all_interior_used=LONGCUT_MODE,
                require_some_interior_multiple=LONGCUT_MODE,
                dominant_dir_only=DOMINANT_DIR_ONLY,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=False,
                trace_max_steps=DEBUG_BFS_STEPS_MAX,
            )

    if result is None:
        _trace_unsolved()
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
                    reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                    dominant_dir_only=DOMINANT_DIR_ONLY,
                    debug=False,
                    debug_counts=debug_counts,
                    trace_steps=DEBUG_TRACE_BFS,
                    trace_step_color=_COLOR_TRACE,
                    trace_candidate_color=_COLOR_FAIL,
                    trace_reset=_COLOR_RESET,
                    trace_max_steps=DEBUG_TRACE_BFS_MAX,
                )
            print(
                f"{_COLOR_FAIL}Total candidates found: {len(candidates)}{_COLOR_RESET}"
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
                print(
                    f"{_COLOR_FAIL}Direction block (current IN): "
                    f"{debug_counts.get('dir_block_current_in', 0)}{_COLOR_RESET}"
                )
                print(
                    f"{_COLOR_FAIL}Direction block (target OUT): "
                    f"{debug_counts.get('dir_block_out', 0)}{_COLOR_RESET}"
                )
                print(
                    f"{_COLOR_FAIL}Direction block (paired IN): "
                    f"{debug_counts.get('dir_block_pair_in', 0)}{_COLOR_RESET}"
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

    _trace_accepted()
    print("Simple shortcut found.")
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
    stage_rows: Optional[List[Tuple[str, int, int, int, int, int]]] = None,
) -> None:
    print("=" * 80)
    print("=== Diagnostics ===")
    if stage_rows:
        header = f"{'Stage':<8} {'Total':>5} {'Simple':>6} {'Complete':>8} {'Unsolved':>8} {'MaxSet':>7}"
        print(header)

        def _cell(text: str, color: str, width: int) -> str:
            padded = f"{text:>{width}}"
            return f"{color}{padded}{_COLOR_RESET}"

        for label, total_s, simple_s, complete_s, unsolved_s, max_s in stage_rows:
            simple_txt = _cell(str(simple_s), _COLOR_START, 6)
            complete_txt = _cell(str(complete_s), _COLOR_AMBER, 8)
            unsolved_txt = _cell(str(unsolved_s), _COLOR_FAIL, 8)
            max_val = str(max_s if complete_s else "n/a")
            max_txt = _cell(max_val, _COLOR_AMBER, 7)
            print(f"{label:<8} {total_s:>5} {simple_txt} {complete_txt} {unsolved_txt} {max_txt}")
        print("-" * 80)
        sum_total = sum(r[1] for r in stage_rows)
        sum_simple = sum(r[2] for r in stage_rows)
        sum_complete = sum(r[3] for r in stage_rows)
        sum_unsolved = sum(r[4] for r in stage_rows)
        simple_txt = _cell(str(sum_simple), _COLOR_START, 6)
        complete_txt = _cell(str(sum_complete), _COLOR_AMBER, 8)
        unsolved_txt = _cell(str(sum_unsolved), _COLOR_FAIL, 8)
        max_txt = " " * 7
        print(f"{'TOTAL':<8} {sum_total:>5} {simple_txt} {complete_txt} {unsolved_txt} {max_txt}")
    else:
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


def _build_surface(sig, perm):
    if DIRECTED_MODE:
        return build_directed_surface_from_signature(sig, perm, surface=SURFACE)
    return build_surface_from_signature(sig, perm, surface=SURFACE)


def _unique_perms(m: int) -> Iterable[Tuple[int, ...]]:
    if DIRECTED_MODE:
        return unique_perms_under_directed_symmetry(m, surface=SURFACE)
    return unique_perms_under_symmetry(m, surface=SURFACE)


def _canonical_perm(perm: Tuple[int, ...], *, n: int) -> Tuple[int, ...]:
    if DIRECTED_MODE:
        return canonical_perm_under_directed_symmetry(perm, surface=SURFACE, n=n)
    return canonical_perm_under_symmetry(perm, surface=SURFACE, n=n)


def _surfaces(sig):
    if DIRECTED_MODE:
        return directed_surfaces_from_signature(sig, surface=SURFACE, unique=UNIQUE)
    return surfaces_from_signature(sig, surface=SURFACE, unique=UNIQUE, exclude_adjacent_I=EXCLUDE_ADJACENT_I)


def _enumerate_stage_perms(m: int) -> List[Tuple[int, ...]]:
    if UNIQUE:
        return list(_unique_perms(m))
    return list(permutations(range(m)))


def _perm_has_adjacent_I(sig, perm, *, surface: str) -> bool:
    if DIRECTED_MODE:
        return False
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


def _prefix_pruning_search(sig_full) -> tuple[int, int, int, int, int, List[Tuple[str, int, int, int, int, int]]]:
    n = len(sig_full)
    if START_PREFIX_LENGTH < 1 or START_PREFIX_LENGTH > n:
        raise ValueError("START_PREFIX_LENGTH must satisfy 1 <= start <= len(signature)")

    candidates = _enumerate_stage_perms(START_PREFIX_LENGTH)
    basket: set[Tuple[int, ...]] = set()  # prefixes with no solution so far

    sig_prefix = sig_full[:START_PREFIX_LENGTH]
    stage_rows: List[Tuple[str, int, int, int, int, int]] = []
    simple_total = 0
    complete_total = 0
    unsolved_total = 0
    max_complete_size = 0
    total_cases = 0
    print("=" * 80)
    print(f"Stage {START_PREFIX_LENGTH}/{n} prefix={list(sig_prefix)}")
    print(f"Frontier size: {len(candidates)}")
    stage_simple = 0
    stage_complete = 0
    stage_unsolved = 0
    stage_max = 0
    for idx, perm in enumerate(candidates, start=1):
        surface_obj = _build_surface(sig_prefix, perm)
        debug_counts: dict = {}
        stage_total = len(candidates)
        _print_case_header(idx, stage_total, surface_obj, label=f"Stage {START_PREFIX_LENGTH}")
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
                reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                require_all_interior_used=LONGCUT_MODE,
                require_some_interior_multiple=LONGCUT_MODE,
                dominant_dir_only=DOMINANT_DIR_ONLY,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                debug_counts=debug_counts,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=False,
                trace_max_steps=DEBUG_BFS_STEPS_MAX,
            )
            s, c, u, msize = _render_result(
                idx,
                stage_total,
                surface_obj,
                result,
                candidates=candidates,
                debug_counts=debug_counts,
                label=f"Stage {START_PREFIX_LENGTH}",
                skip_search=True,
                show_header=False,
            )
            stage_simple += s
            stage_complete += c
            stage_unsolved += u
            stage_max = max(stage_max, msize)
            if START_PREFIX_LENGTH == n:
                total_cases += 1
                simple_total += s
                complete_total += c
                unsolved_total += u
                if msize > max_complete_size:
                    max_complete_size = msize
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
                reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                require_all_interior_used=LONGCUT_MODE,
                require_some_interior_multiple=LONGCUT_MODE,
                dominant_dir_only=DOMINANT_DIR_ONLY,
                allow_complete_set=REQUIRE_DX_INFEASIBLE,
                debug=False,
                progress=SHOW_PROGRESS,
                progress_interval=PROGRESS_INTERVAL,
                trace_steps=False,
                trace_max_steps=DEBUG_BFS_STEPS_MAX,
            )
            s, c, u, msize = _render_result(
                idx,
                stage_total,
                surface_obj,
                result,
                label=f"Stage {START_PREFIX_LENGTH}",
                skip_search=True,
                show_header=False,
            )
            stage_simple += s
            stage_complete += c
            stage_unsolved += u
            stage_max = max(stage_max, msize)
            if START_PREFIX_LENGTH == n:
                total_cases += 1
                simple_total += s
                complete_total += c
                unsolved_total += u
                if msize > max_complete_size:
                    max_complete_size = msize
        if result is None:
            basket.add(perm)
    print(f"Remaining bad prefixes: {len(basket)}")

    if START_PREFIX_LENGTH == n:
        stage_rows.append(
            (f"{START_PREFIX_LENGTH}/{n}", total_cases, stage_simple, stage_complete, stage_unsolved, stage_max)
        )
        return simple_total, complete_total, unsolved_total, max_complete_size, total_cases, stage_rows
    stage_rows.append(
        (f"{START_PREFIX_LENGTH}/{n}", stage_total, stage_simple, stage_complete, stage_unsolved, stage_max)
    )

    for m in range(START_PREFIX_LENGTH, n):
        stage_simple = 0
        stage_complete = 0
        stage_unsolved = 0
        stage_max = 0
        new_label = m
        sig_prefix = sig_full[: m + 1]
        candidates_set: set[Tuple[int, ...]] = set()  # next frontier
        for base in basket:
            for ext in _insert_label_everywhere(base, new_label):
                if UNIQUE:
                    ext = _canonical_perm(ext, n=m + 1)
                candidates_set.add(ext)
        candidates = sorted(candidates_set)
        basket = set()

        is_final = (m + 1 == n)
        filtered_out = 0
        if is_final and EXCLUDE_ADJACENT_I and not DIRECTED_MODE:
            filtered = [perm for perm in candidates if not _perm_has_adjacent_I(sig_full, perm, surface=SURFACE)]
            filtered_out = len(candidates) - len(filtered)
            candidates = filtered
            total_final = len(candidates)
        else:
            total_final = len(candidates)
        print("=" * 80)
        print(f"Stage {m + 1}/{n} prefix={list(sig_prefix)}")
        print(f"Frontier size: {len(candidates)}")
        if filtered_out:
            print(f"Filtered by adjacent-I rule: {filtered_out}")
        stage_total = len(candidates)
        for idx, perm in enumerate(candidates, start=1):
            surface_obj = _build_surface(sig_prefix, perm)
            debug_counts = {}
            _print_case_header(idx, stage_total, surface_obj, label=f"Stage {m + 1}")
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
                    reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                    require_all_interior_used=LONGCUT_MODE,
                    require_some_interior_multiple=LONGCUT_MODE,
                    dominant_dir_only=DOMINANT_DIR_ONLY,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    debug_counts=debug_counts,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=False,
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
                    reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                    require_all_interior_used=LONGCUT_MODE,
                    require_some_interior_multiple=LONGCUT_MODE,
                    dominant_dir_only=DOMINANT_DIR_ONLY,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=False,
                    trace_max_steps=DEBUG_BFS_STEPS_MAX,
                )
            if is_final:
                total_cases += 1
                s, c, u, msize = _render_result(
                    total_cases,
                    total_final,
                    surface_obj,
                    result,
                    candidates=candidates,
                    debug_counts=debug_counts,
                    skip_search=True,
                    show_header=False,
                )
                simple_total += s
                complete_total += c
                unsolved_total += u
                if msize > max_complete_size:
                    max_complete_size = msize
                stage_simple += s
                stage_complete += c
                stage_unsolved += u
                stage_max = max(stage_max, msize)
            else:
                s, c, u, msize = _render_result(
                    idx,
                    stage_total,
                    surface_obj,
                    result,
                    candidates=candidates,
                    debug_counts=debug_counts,
                    label=f"Stage {m + 1}",
                    skip_search=True,
                    show_header=False,
                )
                stage_simple += s
                stage_complete += c
                stage_unsolved += u
                stage_max = max(stage_max, msize)
                if result is None:
                    basket.add(perm)
        print(f"Remaining bad prefixes: {len(basket)}")
        stage_rows.append(
            (f"{m + 1}/{n}", total_final if is_final else stage_total, stage_simple, stage_complete, stage_unsolved, stage_max)
        )

    return simple_total, complete_total, unsolved_total, max_complete_size, total_cases, stage_rows


def main() -> None:
    _print_header(SIGNATURE)

    simple_total = 0
    complete_total = 0
    unsolved_total = 0
    max_complete_size = 0

    stage_rows: List[Tuple[str, int, int, int, int, int]] = []
    if not PREFIX_PRUNING:
        surfaces = _surfaces(SIGNATURE)
        for idx, surface in enumerate(surfaces, start=1):
            debug_counts = {}
            _print_case_header(idx, len(surfaces), surface)
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
                    reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                    require_all_interior_used=LONGCUT_MODE,
                    require_some_interior_multiple=LONGCUT_MODE,
                    dominant_dir_only=DOMINANT_DIR_ONLY,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    debug_counts=debug_counts,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=False,
                    trace_max_steps=DEBUG_BFS_STEPS_MAX,
                )
                s, c, u, m = _render_result(
                    idx,
                    len(surfaces),
                    surface,
                    result,
                    candidates=candidates,
                    debug_counts=debug_counts,
                    skip_search=True,
                    show_header=False,
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
                    reject_all_interior_used=REJECT_ALL_INTERIOR_USED,
                    require_all_interior_used=LONGCUT_MODE,
                    require_some_interior_multiple=LONGCUT_MODE,
                    dominant_dir_only=DOMINANT_DIR_ONLY,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    debug=False,
                    progress=SHOW_PROGRESS,
                    progress_interval=PROGRESS_INTERVAL,
                    trace_steps=False,
                    trace_max_steps=DEBUG_BFS_STEPS_MAX,
                )
                s, c, u, m = _render_result(
                    idx,
                    len(surfaces),
                    surface,
                    result,
                    skip_search=True,
                    show_header=False,
                )
            simple_total += s
            complete_total += c
            unsolved_total += u
            if m > max_complete_size:
                max_complete_size = m
        total = len(surfaces)
        stage_rows.append(("all", total, simple_total, complete_total, unsolved_total, max_complete_size))
    else:
        sig_full = parse_klein_signature(SIGNATURE)
        (
            simple_total,
            complete_total,
            unsolved_total,
            max_complete_size,
            total,
            stage_rows,
        ) = _prefix_pruning_search(sig_full)

    _print_diagnostics(
        total=total,
        simple_count=simple_total,
        complete_count=complete_total,
        unsolved_count=unsolved_total,
        max_complete_size=max_complete_size,
        stage_rows=stage_rows,
    )


if __name__ == "__main__":
    main()

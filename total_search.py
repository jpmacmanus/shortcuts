"""
total_search.py

Search over all Klein signatures of a given length N.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from directed_klein_signature_generator import (
    build_directed_surface_from_signature,
    canonical_perm_under_directed_symmetry,
    directed_surfaces_from_signature,
    unique_perms_under_directed_symmetry,
)
from klein_signature_generator import (
    H,
    I,
    R,
    V,
    KleinElt,
    KLEIN_BY_CHAR,
    build_surface_from_signature,
    canonical_perm_under_symmetry,
    parse_klein_signature,
    surfaces_from_signature,
    unique_perms_under_symmetry,
)
from track_bfs import (
    search_shortcut_or_complete_set,
)

# ----------------------------
# Config
# ----------------------------

# Search input
N = 8                 # length of Klein signatures to enumerate
SURFACE = "annulus"   # "annulus" or "strip"

# Directed edge modes (can be enabled independently).
DIRECTED_MARKED = True    # vertical directions on marked TOP/BOTTOM edges
DIRECTED_INTERIOR = True  # horizontal directions on interior edges (annulus only)

# Generation / symmetry controls
UNIQUE = True               # skips surfaces which differ by a symmetry.
EXCLUDE_ADJACENT_I = True  # skip cases with trivial solutions due to adjacent I squares.
PREFIX_PRUNING = False       # start search on smaller prefix cases and build up to desired case.
START_PREFIX_LENGTH = 1     # only used when PREFIX_PRUNING=True
INFIX_PRUNING = True        # skip signatures containing a solved contiguous subword.

# Acceptance constraints (same as exhaustive_search)
REQUIRE_DY_NONZERO = False
REQUIRE_DX_INFEASIBLE = False
REQUIRE_EVEN_TURNING = False
REQUIRE_EVEN_OR_PAIRS = True
DOMINANT_DIR_ONLY = False
LIMIT_INTERIOR_CROSSINGS = True
REJECT_ALL_INTERIOR_USED = True
LONGCUT_MODE = False
REQUIRE_ALL_MARKED_USED = False
REQUIRE_NONTRIVIAL = True

# BFS bounds and minimization parameters
MAX_NODES = 5000
MAX_CANDIDATES = 1000
MAX_PORTS_PER_EDGE = 3
MINIMIZE_SEED = None
MAX_MINIMIZE_SIZE = 30

# UI
SHOW_LAST_SOLUTION = True
REFRESH_INTERVAL = 0.2
SHOW_UNSOLVED_FINAL = True

# Optional markdown report
WRITE_MARKDOWN_REPORT = False
REPORT_DIR = "reports"
REPORT_PATH = ""  # Empty => auto name inside REPORT_DIR.

# ANSI
_RESET = "\033[0m"
_CYAN = "\033[36m"
_GREEN = "\033[32m"
_AMBER = "\033[33m"
_RED = "\033[31m"


# ----------------------------
# Klein-word helpers
# ----------------------------


_BITS_TO_CHAR = {(False, False): "I", (True, False): "H", (False, True): "V", (True, True): "R"}
_CHAR_ORDER = {c: i for i, c in enumerate("IHRV")}


def _apply_klein(word: str, g: KleinElt) -> str:
    out = []
    for c in word:
        k = KLEIN_BY_CHAR[c]
        out.append(_BITS_TO_CHAR[(k.swap_tb ^ g.swap_tb, k.flip_orient ^ g.flip_orient)])
    return "".join(out)


def _canonical_k4(word: str) -> str:
    def _key(w: str) -> tuple:
        return tuple(_CHAR_ORDER[c] for c in w)

    return min((_apply_klein(word, g) for g in (I, H, V, R)), key=_key)


def _all_signatures(n: int) -> Iterable[str]:
    for tup in product("IRVH", repeat=n):
        yield "".join(tup)


def _unique_signatures(n: int) -> List[str]:
    unique: dict[str, str] = {}
    for sig in _all_signatures(n):
        canon = _canonical_k4(sig)
        if canon not in unique:
            unique[canon] = canon
    return sorted(unique.values())


# ----------------------------
# Signature search
# ----------------------------


def _build_surface(sig: List[KleinElt], perm: Sequence[int]):
    if DIRECTED_MARKED or DIRECTED_INTERIOR:
        return build_directed_surface_from_signature(
            sig,
            perm,
            surface=SURFACE,
            directed_marked=DIRECTED_MARKED,
            directed_interior=DIRECTED_INTERIOR,
        )
    return build_surface_from_signature(sig, perm, surface=SURFACE)


def _unique_perms(m: int) -> Iterable[Tuple[int, ...]]:
    if DIRECTED_MARKED or DIRECTED_INTERIOR:
        return unique_perms_under_directed_symmetry(m, surface=SURFACE)
    return unique_perms_under_symmetry(m, surface=SURFACE)


def _canonical_perm(perm: Tuple[int, ...], *, n: int) -> Tuple[int, ...]:
    if DIRECTED_MARKED or DIRECTED_INTERIOR:
        return canonical_perm_under_directed_symmetry(perm, surface=SURFACE, n=n)
    return canonical_perm_under_symmetry(perm, surface=SURFACE, n=n)


def _surfaces(sig: str):
    if DIRECTED_MARKED or DIRECTED_INTERIOR:
        return directed_surfaces_from_signature(
            sig,
            surface=SURFACE,
            unique=UNIQUE,
            directed_marked=DIRECTED_MARKED,
            directed_interior=DIRECTED_INTERIOR,
        )
    return surfaces_from_signature(sig, surface=SURFACE, unique=UNIQUE, exclude_adjacent_I=EXCLUDE_ADJACENT_I)


def _insert_label_everywhere(base: Tuple[int, ...], new_label: int) -> Iterable[Tuple[int, ...]]:
    m = len(base)
    for pos in range(m + 1):
        yield base[:pos] + (new_label,) + base[pos:]


def _perm_has_adjacent_I(sig, perm, *, surface: str) -> bool:
    if DIRECTED_MARKED or DIRECTED_INTERIOR:
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


def _run_signature(
    sig_str: str,
    *,
    surface_hook=None,
    progress_hook=None,
) -> Tuple[bool, int, int, int, int, int, Optional[str], List[dict]]:
    """
    Run the search for a single signature.
    Returns (solved, simple, complete, unsolved, maxset, total_cases, last_solution_text).
    """
    sig_full = parse_klein_signature(sig_str)
    simple_total = 0
    complete_total = 0
    unsolved_total = 0
    max_complete_size = 0
    total_cases = 0
    last_solution: Optional[str] = None
    solution_items: List[dict] = []

    if not PREFIX_PRUNING:
        surfaces = list(_surfaces(sig_str))
        total_surfaces = len(surfaces)
        for surface_idx, surface in enumerate(surfaces, start=1):
            if surface_hook is not None:
                surface_hook(surface, surface_idx, total_surfaces)
            try:
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
                    require_all_marked_used=REQUIRE_ALL_MARKED_USED,
                    longcut_mode=LONGCUT_MODE,
                    dominant_dir_only=DOMINANT_DIR_ONLY,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    require_nontrivial=REQUIRE_NONTRIVIAL,
                    debug=False,
                    progress=True,
                    progress_hook=progress_hook,
                )
            except ValueError:
                result = None
            total_cases += 1
            kind, text = _format_result_text(result)
            if result is None:
                unsolved_total += 1
            elif isinstance(result, list) and REQUIRE_ALL_MARKED_USED:
                if any(isinstance(item, list) for item in result):
                    complete_total += 1
                    max_complete_size = max(
                        max_complete_size,
                        max(len(item) for item in result if isinstance(item, list)),
                    )
                else:
                    simple_total += 1
                if SHOW_LAST_SOLUTION:
                    last_solution = text
                solution_items.append({"kind": kind, "text": text, "idx": total_cases})
            elif isinstance(result, list):
                complete_total += 1
                max_complete_size = max(max_complete_size, len(result))
                if SHOW_LAST_SOLUTION:
                    last_solution = text
                solution_items.append({"kind": kind, "text": text, "idx": total_cases})
            else:
                simple_total += 1
                if SHOW_LAST_SOLUTION:
                    last_solution = text
                solution_items.append({"kind": kind, "text": text, "idx": total_cases})

        solved = (unsolved_total == 0) and (simple_total + complete_total > 0)
        return (
            solved,
            simple_total,
            complete_total,
            unsolved_total,
            max_complete_size,
            total_cases,
            last_solution,
            solution_items,
        )

    # Prefix pruning
    n = len(sig_full)
    if START_PREFIX_LENGTH < 1 or START_PREFIX_LENGTH > n:
        raise ValueError("START_PREFIX_LENGTH must satisfy 1 <= start <= len(signature)")

    candidates = list(_unique_perms(START_PREFIX_LENGTH))
    basket: set[Tuple[int, ...]] = set()

    for m in range(START_PREFIX_LENGTH, n + 1):
        sig_prefix = sig_full[:m]
        is_final = (m == n)
        if m == START_PREFIX_LENGTH:
            stage_candidates = candidates
        else:
            new_label = m - 1
            candidates_set: set[Tuple[int, ...]] = set()
            for base in basket:
                for ext in _insert_label_everywhere(base, new_label):
                    if UNIQUE:
                        ext = _canonical_perm(ext, n=m)
                    candidates_set.add(ext)
            stage_candidates = sorted(candidates_set)
            basket = set()

        for surface_idx, perm in enumerate(stage_candidates, start=1):
            if is_final and EXCLUDE_ADJACENT_I and not (DIRECTED_MARKED or DIRECTED_INTERIOR):
                if _perm_has_adjacent_I(sig_full, perm, surface=SURFACE):
                    continue

            surface_obj = _build_surface(sig_prefix, perm)
            if surface_hook is not None:
                surface_hook(surface_obj, surface_idx, len(stage_candidates))
            try:
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
                    require_all_marked_used=REQUIRE_ALL_MARKED_USED,
                    longcut_mode=LONGCUT_MODE,
                    dominant_dir_only=DOMINANT_DIR_ONLY,
                    allow_complete_set=REQUIRE_DX_INFEASIBLE,
                    require_nontrivial=REQUIRE_NONTRIVIAL,
                    debug=False,
                    progress=True,
                    progress_hook=progress_hook,
                )
            except ValueError:
                result = None

            if is_final:
                total_cases += 1
                kind, text = _format_result_text(result)
                if result is None:
                    unsolved_total += 1
                elif isinstance(result, list) and REQUIRE_ALL_MARKED_USED:
                    if any(isinstance(item, list) for item in result):
                        complete_total += 1
                        max_complete_size = max(
                            max_complete_size,
                            max(len(item) for item in result if isinstance(item, list)),
                        )
                    else:
                        simple_total += 1
                    if SHOW_LAST_SOLUTION:
                        last_solution = text
                    solution_items.append({"kind": kind, "text": text, "idx": total_cases})
                elif isinstance(result, list):
                    complete_total += 1
                    max_complete_size = max(max_complete_size, len(result))
                    if SHOW_LAST_SOLUTION:
                        last_solution = text
                    solution_items.append({"kind": kind, "text": text, "idx": total_cases})
                else:
                    simple_total += 1
                    if SHOW_LAST_SOLUTION:
                        last_solution = text
                    solution_items.append({"kind": kind, "text": text, "idx": total_cases})
            else:
                if result is None:
                    basket.add(perm)

    solved = (unsolved_total == 0) and (simple_total + complete_total > 0)
    return (
        solved,
        simple_total,
        complete_total,
        unsolved_total,
        max_complete_size,
        total_cases,
        last_solution,
        solution_items,
    )


# ----------------------------
# UI
# ----------------------------


def _render_surface_text(surface) -> str:
    if hasattr(surface, "render"):
        return surface.render()
    return str(surface)


def _bar(current: int, total: int, width: int = 24) -> str:
    if total <= 0:
        return "-" * width
    filled = int(round((current / total) * width))
    filled = min(width, max(0, filled))
    return "#" * filled + "-" * (width - filled)


def _render_covering_set(items: List[object]) -> str:
    lines: List[str] = [f"Covering set found (size {len(items)})."]
    for idx, item in enumerate(items, start=1):
        if isinstance(item, list):
            lines.append(f"  [{idx}] Complete candidate set (size {len(item)})")
            for c_idx, st in enumerate(item, start=1):
                lines.append(f"    ({c_idx}) dy: {st.dy}  dx: {st.dx_linear_form(pretty=True)}")
                lines.append(st.render())
        else:
            st = item
            lines.append(f"  [{idx}] Simple shortcut")
            lines.append(st.render())
            lines.append(f"dy: {st.dy}")
            lines.append(f"dx: {st.dx_linear_form(pretty=True)}")
    return "\n".join(lines)


def _format_result_text(result) -> Tuple[str, str]:
    if result is None:
        return "unsolved", "No solution found."
    if isinstance(result, list) and REQUIRE_ALL_MARKED_USED:
        return "covering", _render_covering_set(result)
    if isinstance(result, list):
        lines: List[str] = [f"Complete candidate set found (size {len(result)})."]
        for idx, st in enumerate(result, start=1):
            lines.append(f"  [{idx}] dy: {st.dy}  dx: {st.dx_linear_form(pretty=True)}")
            lines.append(st.render())
        return "complete", "\n".join(lines)
    return "simple", "\n".join(
        [
            "Simple shortcut found.",
            result.render(),
            f"dy: {result.dy}",
            f"dx: {result.dx_linear_form(pretty=True)}",
        ]
    )


def _conclusion(simple: int, complete: int, uns: int) -> str:
    if uns > 0:
        return "Unsolved"
    if simple > 0 and complete > 0:
        return "Simple + Complete"
    if simple > 0:
        return "Simple"
    if complete > 0:
        return "Complete"
    return "No result"


def _write_markdown_report(
    *,
    total_sigs: int,
    skipped: int,
    sig_solved: int,
    sig_simple_only: int,
    sig_complete_only: int,
    sig_both: int,
    sig_unsolved: int,
    total_cases: int,
    case_simple: int,
    case_complete: int,
    case_unsolved: int,
    max_complete_size: int,
    elapsed: float,
    sig_rate: float,
    case_rate: float,
    considered_rows: List[dict],
    solutions: List[dict],
    unsolved_final: List[Tuple[str, int, int]],
) -> Path:
    report_dir = Path(REPORT_DIR)
    report_dir.mkdir(parents=True, exist_ok=True)

    if REPORT_PATH.strip():
        out = Path(REPORT_PATH.strip())
        if not out.is_absolute():
            out = report_dir / out
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = report_dir / f"total_search_report_{SURFACE}_N{N}_{stamp}.md"

    lines: List[str] = []
    lines.append("# Total Search Report")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Surface: `{SURFACE}`")
    lines.append(f"- N: `{N}`")
    lines.append(f"- Directed marked: `{DIRECTED_MARKED}`")
    lines.append(f"- Directed interior: `{DIRECTED_INTERIOR}`")
    lines.append(f"- Infix pruning: `{INFIX_PRUNING}`")
    lines.append(f"- Prefix pruning: `{PREFIX_PRUNING}`")
    lines.append(f"- Total signatures (enumerated): `{total_sigs}`")
    lines.append(f"- Skipped by infix: `{skipped}`")
    lines.append(f"- Solved signatures: `{sig_solved}`")
    lines.append(f"- Simple-only signatures: `{sig_simple_only}`")
    lines.append(f"- Complete-only signatures: `{sig_complete_only}`")
    lines.append(f"- Both simple+complete signatures: `{sig_both}`")
    lines.append(f"- Unsolved signatures: `{sig_unsolved}`")
    lines.append(f"- Total cases evaluated: `{total_cases}`")
    lines.append(f"- Simple cases: `{case_simple}`")
    lines.append(f"- Complete cases: `{case_complete}`")
    lines.append(f"- Unsolved cases: `{case_unsolved}`")
    lines.append(f"- Max complete set size: `{max_complete_size if max_complete_size else 'n/a'}`")
    lines.append(f"- Elapsed: `{elapsed:.1f}s`")
    lines.append(f"- Signatures/sec: `{sig_rate:.2f}`")
    lines.append(f"- Cases/sec: `{case_rate:.2f}`")
    lines.append("")

    if SHOW_UNSOLVED_FINAL:
        lines.append(f"- Unsolved signatures at length {N}: `{len(unsolved_final)}`")
        lines.append("")

    lines.append("## Considered Signatures")
    lines.append("")
    lines.append("| Signature | Length | Cases | Simple | Complete | Unsolved | Conclusion |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in sorted(considered_rows, key=lambda r: (r["length"], r["signature"])):
        lines.append(
            f"| `{row['signature']}` | {row['length']} | {row['cases']} | "
            f"{row['simple']} | {row['complete']} | {row['unsolved']} | {row['conclusion']} |"
        )
    lines.append("")

    if SHOW_UNSOLVED_FINAL and unsolved_final:
        lines.append(f"## Unsolved Signatures At Length {N}")
        lines.append("")
        lines.append("| Signature | Simple | Complete |")
        lines.append("|---|---:|---:|")
        for sig, simple, complete in sorted(unsolved_final):
            lines.append(f"| `{sig}` | {simple} | {complete} |")
        lines.append("")

    lines.append("## Solutions")
    lines.append("")
    if not solutions:
        lines.append("No solutions were found.")
    else:
        for i, sol in enumerate(sorted(solutions, key=lambda s: (s["signature"], s["idx"])), start=1):
            lines.append(f"### {i}. `{sol['signature']}` ({sol['kind']})")
            lines.append("")
            lines.append("```text")
            lines.append(sol["text"])
            lines.append("```")
            lines.append("")

    out.write_text("\n".join(lines) + "\n")
    return out


def _render_ui(
    *,
    current_sig: str,
    idx: int,
    total: int,
    length: int,
    length_idx: int,
    length_total: int,
    sig_solved: int,
    sig_complete_only: int,
    sig_simple_only: int,
    sig_both: int,
    sig_unsolved: int,
    skipped: int,
    total_cases: int,
    case_simple: int,
    case_complete: int,
    case_unsolved: int,
    max_complete_size: int,
    current_surface: Optional[str],
    current_surface_idx: int,
    current_surface_total: int,
    current_progress: Optional[str],
    elapsed: float,
    eta: Optional[float],
    last_solution: Optional[str],
) -> None:
    sys.stdout.write("\033[H\033[J")
    pct = (idx / total) * 100 if total else 0.0
    sig_rate = (idx / elapsed) if elapsed > 0 else 0.0
    case_rate = (total_cases / elapsed) if elapsed > 0 else 0.0
    print(f"{_CYAN}=== Total search ==={_RESET}")
    print(f"Surface: {SURFACE}  N={N}  Alphabet=IRVH")
    print(f"Directed marked: {DIRECTED_MARKED}  Directed interior: {DIRECTED_INTERIOR}")
    print(f"Infix pruning: {INFIX_PRUNING}  Prefix pruning: {PREFIX_PRUNING}")
    print()
    print(
        f"Length: {length}  Signature: {current_sig} ({length_idx}/{length_total})"
        f"  Global: {idx}/{total} ({pct:.1f}%)"
    )
    completed = sig_solved + sig_unsolved + skipped
    print(f"Completed: {completed}  Skipped: {skipped}")
    print(
        "Solved signatures:"
        f" {sig_solved} (simple-only={sig_simple_only}, complete-only={sig_complete_only}, both={sig_both})"
    )
    print(f"Unsolved signatures: {sig_unsolved}")
    print()
    print(
        f"Cases: total={total_cases} simple={case_simple} complete={case_complete} unsolved={case_unsolved}"
    )
    print(f"Max complete set size: {max_complete_size if max_complete_size else 'n/a'}")
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Signatures/sec: {sig_rate:.2f}  Cases/sec: {case_rate:.2f}")
    if eta is not None:
        print(f"ETA: {eta:.1f}s")
    print()
    if current_surface:
        bar = _bar(current_surface_idx, current_surface_total)
        print(f"Surface {current_surface_idx}/{current_surface_total} [{bar}]")
        print(current_surface)
    if current_progress:
        print(current_progress)
    if SHOW_LAST_SOLUTION and last_solution:
        print(f"{_GREEN}Last solution:{_RESET}")
        print(last_solution)


def main() -> None:
    directed_mode = DIRECTED_MARKED or DIRECTED_INTERIOR
    if EXCLUDE_ADJACENT_I and directed_mode:
        pass

    length_start = 1 if INFIX_PRUNING else N
    unique_by_length: dict[int, List[str]] = {}
    for length in range(length_start, N + 1):
        unique_by_length[length] = _unique_signatures(length)
    total_sigs = sum(len(sigs) for sigs in unique_by_length.values())
    start = time.time()
    last_refresh = 0.0

    sig_solved = 0
    sig_simple_only = 0
    sig_complete_only = 0
    sig_both = 0
    sig_unsolved = 0
    skipped = 0
    last_solution: Optional[str] = None

    solved_by_length: dict[int, set[str]] = {}
    total_cases = 0
    case_simple = 0
    case_complete = 0
    case_unsolved = 0
    max_complete_size = 0
    current_surface_text: Optional[str] = None
    current_surface_idx = 0
    current_surface_total = 0
    current_progress: Optional[str] = None
    current_sig = ""
    current_length = 0
    current_length_idx = 0
    current_length_total = 0
    unsolved_final: List[Tuple[str, int, int]] = []
    unsolved_final_cases = 0
    considered_rows: List[dict] = []
    solution_rows: List[dict] = []

    def _surface_hook(surface, surface_idx: int, surface_total: int) -> None:
        nonlocal current_surface_text, current_surface_idx, current_surface_total
        current_surface_text = _render_surface_text(surface)
        current_surface_idx = surface_idx
        current_surface_total = surface_total

    def _progress_hook(
        *,
        start_idx: int,
        starts_total: int,
        expanded: int,
        max_nodes: int,
        queue_len: int,
        seen_len: int,
        msg: str,
    ) -> None:
        nonlocal current_progress, last_refresh
        current_progress = (
            f"start {start_idx}/{starts_total} "
            f"[{_bar(expanded, max_nodes)}] "
            f"{expanded}/{max_nodes} queue {queue_len} seen {seen_len}"
        )
        now = time.time()
        if now - last_refresh >= REFRESH_INTERVAL:
            eta = None
            if global_idx > 0:
                elapsed = now - start
                eta = (elapsed / global_idx) * (total_sigs - global_idx)
            _render_ui(
                current_sig=current_sig,
                idx=global_idx,
                total=total_sigs,
                length=current_length,
                length_idx=current_length_idx,
                length_total=current_length_total,
                sig_solved=sig_solved,
                sig_complete_only=sig_complete_only,
                sig_simple_only=sig_simple_only,
                sig_both=sig_both,
                sig_unsolved=sig_unsolved,
                skipped=skipped,
                total_cases=total_cases,
                case_simple=case_simple,
                case_complete=case_complete,
                case_unsolved=case_unsolved,
                max_complete_size=max_complete_size,
                current_surface=current_surface_text,
                current_surface_idx=current_surface_idx,
                current_surface_total=current_surface_total,
                current_progress=current_progress,
                elapsed=now - start,
                eta=eta,
                last_solution=last_solution,
            )
            last_refresh = now

    global_idx = 0
    for length in range(length_start, N + 1):
        signatures = unique_by_length[length]
        length_total = len(signatures)
        for length_idx, sig in enumerate(signatures, start=1):
            global_idx += 1
            current_sig = sig
            current_length = length
            current_length_idx = length_idx
            current_length_total = length_total
            if INFIX_PRUNING and solved_by_length:
                skip = False
                for i in range(length):
                    for j in range(i + 1, length + 1):
                        if j - i == length or j - i == 1:
                            continue
                        sub_len = j - i
                        solved_set = solved_by_length.get(sub_len)
                        if not solved_set:
                            continue
                        sub = sig[i:j]
                        if _canonical_k4(sub) in solved_set:
                            skip = True
                            break
                    if skip:
                        break
                if skip:
                    skipped += 1
                    now = time.time()
                    if now - last_refresh >= REFRESH_INTERVAL:
                        eta = None
                        if global_idx > 0:
                            elapsed = now - start
                            eta = (elapsed / global_idx) * (total_sigs - global_idx)
                        _render_ui(
                            current_sig=sig,
                            idx=global_idx,
                            total=total_sigs,
                            length=length,
                            length_idx=length_idx,
                            length_total=length_total,
                            sig_solved=sig_solved,
                            sig_complete_only=sig_complete_only,
                            sig_simple_only=sig_simple_only,
                            sig_both=sig_both,
                            sig_unsolved=sig_unsolved,
                            skipped=skipped,
                            total_cases=total_cases,
                            case_simple=case_simple,
                            case_complete=case_complete,
                            case_unsolved=case_unsolved,
                            max_complete_size=max_complete_size,
                            current_surface=current_surface_text,
                            current_surface_idx=current_surface_idx,
                            current_surface_total=current_surface_total,
                            current_progress=current_progress,
                            elapsed=now - start,
                            eta=eta,
                            last_solution=last_solution,
                        )
                        last_refresh = now
                    continue

            (
                solved,
                simple,
                complete,
                uns,
                _maxset,
                _total_cases,
                last,
                sig_solutions,
            ) = _run_signature(
                sig,
                surface_hook=_surface_hook,
                progress_hook=_progress_hook,
            )
            total_cases += _total_cases
            case_simple += simple
            case_complete += complete
            case_unsolved += uns
            if _maxset:
                max_complete_size = max(max_complete_size, _maxset)
            considered_rows.append(
                {
                    "signature": sig,
                    "length": length,
                    "cases": _total_cases,
                    "simple": simple,
                    "complete": complete,
                    "unsolved": uns,
                    "conclusion": _conclusion(simple, complete, uns),
                }
            )
            for item in sig_solutions:
                solution_rows.append(
                    {
                        "signature": sig,
                        "idx": item["idx"],
                        "kind": item["kind"],
                        "text": item["text"],
                    }
                )

            if uns > 0:
                sig_unsolved += 1
                if length == N:
                    unsolved_final.append((sig, simple, complete))
                    unsolved_final_cases += uns
            else:
                if simple > 0 and complete > 0:
                    sig_both += 1
                elif simple > 0:
                    sig_simple_only += 1
                elif complete > 0:
                    sig_complete_only += 1
                if simple > 0 or complete > 0:
                    sig_solved += 1

            if last:
                last_solution = last

            if solved and INFIX_PRUNING:
                solved_set = solved_by_length.setdefault(length, set())
                solved_set.add(_canonical_k4(sig))

            now = time.time()
            eta = None
            if global_idx > 0:
                elapsed = now - start
                eta = (elapsed / global_idx) * (total_sigs - global_idx)
            if now - last_refresh >= REFRESH_INTERVAL or global_idx == total_sigs:
                _render_ui(
                    current_sig=sig,
                    idx=global_idx,
                    total=total_sigs,
                    length=length,
                    length_idx=length_idx,
                    length_total=length_total,
                    sig_solved=sig_solved,
                    sig_complete_only=sig_complete_only,
                    sig_simple_only=sig_simple_only,
                    sig_both=sig_both,
                    sig_unsolved=sig_unsolved,
                    skipped=skipped,
                    total_cases=total_cases,
                    case_simple=case_simple,
                    case_complete=case_complete,
                    case_unsolved=case_unsolved,
                    max_complete_size=max_complete_size,
                    current_surface=current_surface_text,
                    current_surface_idx=current_surface_idx,
                    current_surface_total=current_surface_total,
                    current_progress=current_progress,
                    elapsed=now - start,
                    eta=eta,
                    last_solution=last_solution,
                )
                last_refresh = now

    elapsed = time.time() - start
    sig_rate = (total_sigs / elapsed) if elapsed > 0 else 0.0
    case_rate = (total_cases / elapsed) if elapsed > 0 else 0.0
    print()
    print(f"{_CYAN}=== Summary ==={_RESET}")
    print(f"Surface: {SURFACE}  N={N}  Alphabet=IRVH")
    print(f"Directed marked: {DIRECTED_MARKED}  Directed interior: {DIRECTED_INTERIOR}")
    print(f"Infix pruning: {INFIX_PRUNING}  Prefix pruning: {PREFIX_PRUNING}")
    print()
    print(f"Total signatures: {total_sigs}")
    print(f"Skipped by infix: {skipped}")
    print(f"Solved signatures: {sig_solved}")
    print(f"  Simple-only: {sig_simple_only}")
    print(f"  Complete-only: {sig_complete_only}")
    print(f"  Both simple+complete: {sig_both}")
    print(f"Unsolved signatures: {sig_unsolved}")
    print()
    print(f"Total cases evaluated: {total_cases}")
    print(f"  Simple cases: {case_simple}")
    print(f"  Complete cases: {case_complete}")
    print(f"  Unsolved cases: {case_unsolved}")
    print(f"Max complete set size: {max_complete_size if max_complete_size else 'n/a'}")
    if SHOW_UNSOLVED_FINAL and unsolved_final:
        print()
        print(f"Unsolved signatures at length {N}:")
        for sig, simple, complete in unsolved_final:
            print(f"  {sig}  (simple={simple}, complete={complete})")
    if SHOW_UNSOLVED_FINAL:
        print()
        print(f"Unsolved signatures at length {N}: {len(unsolved_final)}")
    print()
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Signatures/sec: {sig_rate:.2f}")
    print(f"Cases/sec: {case_rate:.2f}")

    if WRITE_MARKDOWN_REPORT:
        out = _write_markdown_report(
            total_sigs=total_sigs,
            skipped=skipped,
            sig_solved=sig_solved,
            sig_simple_only=sig_simple_only,
            sig_complete_only=sig_complete_only,
            sig_both=sig_both,
            sig_unsolved=sig_unsolved,
            total_cases=total_cases,
            case_simple=case_simple,
            case_complete=case_complete,
            case_unsolved=case_unsolved,
            max_complete_size=max_complete_size,
            elapsed=elapsed,
            sig_rate=sig_rate,
            case_rate=case_rate,
            considered_rows=considered_rows,
            solutions=solution_rows,
            unsolved_final=unsolved_final,
        )
        print()
        print(f"Markdown report written to: {out}")


if __name__ == "__main__":
    main()

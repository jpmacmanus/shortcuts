#!/usr/bin/env python3
"""
exhaustive_ks_search.py

Exhaustively enumerate marked annuli/strips with a given Klein signature,
then run find_shortcut on each and print results.

Enhancement:
- If no shortcut is found for an example, optionally print a blank schematic
  (no chords) for inspection. Pair-ID tags and marked identifications are still shown.

Options:
- DEDUPLICATE: enumerate only distinct examples up to surface symmetries
- SIMPLE_ONLY: only accept "simple" shortcuts (dx == lambda)
- PRETTY_PRINT_ON_SUCCESS: pretty-print witnesses using ascii_pretty.py
- PRINT_BLANK_ON_FAILURE: pretty-print the blank surface when no shortcut is found
"""

from __future__ import annotations

import inspect
from itertools import permutations
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

from search_shortcuts import find_shortcut
from ascii_pretty import render_trackstate, RenderConfig

from klein_signature_generator import (
    parse_klein_signature,
    build_surface_from_signature,
    all_unique_surfaces_with_signature,
)

from chord_diagram import BoundaryPoint, PortKind
from annulus import Side
from track_state import initial_state


# ----------------------------
# User-configurable variables
# ----------------------------

SURFACE_TYPE = "annulus"          # "annulus" or "strip"
N = 8
KLEIN_SIGNATURE = "I R V H I R V H"

DEDUPLICATE = True
LIMIT_EXAMPLES: Optional[int] = None

MAX_STEPS: Optional[int] = None
MAX_NODES: int = 2000

SIMPLE_ONLY = True

PRETTY_PRINT_ON_SUCCESS = True
PRINT_BLANK_ON_FAILURE = True

PRETTY_CONFIG = RenderConfig(width=22, height=11, pad_between=3)
PRINT_WITNESS_WORD = True


# ----------------------------
# Helpers
# ----------------------------

def _filtered_kwargs(fn, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def _call_find_shortcut(surface_obj) -> Any:
    desired = {
        "max_steps": MAX_STEPS,
        "max_nodes": MAX_NODES,
        "simple_only": SIMPLE_ONLY,
    }
    kwargs = _filtered_kwargs(find_shortcut, desired)
    return find_shortcut(surface_obj, **kwargs)


def _factorial(n: int) -> int:
    out = 1
    for k in range(2, n + 1):
        out *= k
    return out


def _format_perm(perm: Sequence[int]) -> str:
    return "[" + ",".join(map(str, perm)) + "]"


def _try_get_format_shortcut():
    try:
        from search_shortcuts import format_shortcut  # type: ignore
        return format_shortcut
    except Exception:
        return None


def _blank_trackstate_for_surface(surface_obj):
    """
    Create an empty TrackState just to render the surface markings.
    Prefer starting on a marked TOP edge, else marked BOTTOM, else LEFT.
    """
    for i in range(surface_obj.N):
        if surface_obj.is_marked(surface_obj.edge(Side.TOP, i)):
            return initial_state(i, BoundaryPoint(PortKind.TOP, 1))
    for i in range(surface_obj.N):
        if surface_obj.is_marked(surface_obj.edge(Side.BOTTOM, i)):
            return initial_state(i, BoundaryPoint(PortKind.BOTTOM, 1))
    # fallback: always valid
    return initial_state(0, BoundaryPoint(PortKind.LEFT, 1))


def _enumerate_surfaces() -> Iterable[Tuple[object, Optional[Tuple[int, ...]]]]:
    sig = parse_klein_signature(KLEIN_SIGNATURE)

    if DEDUPLICATE:
        examples = all_unique_surfaces_with_signature(sig, surface=SURFACE_TYPE)
        if LIMIT_EXAMPLES is not None:
            examples = examples[:LIMIT_EXAMPLES]
        for A in examples:
            yield A, None
    else:
        count = 0
        for perm in permutations(range(N)):
            A = build_surface_from_signature(sig, perm, surface=SURFACE_TYPE)
            yield A, perm
            count += 1
            if LIMIT_EXAMPLES is not None and count >= LIMIT_EXAMPLES:
                break


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    sig = parse_klein_signature(KLEIN_SIGNATURE)
    if len(sig) != N:
        raise ValueError(f"KLEIN_SIGNATURE has length {len(sig)} but N={N}")

    if SURFACE_TYPE not in ("annulus", "strip"):
        raise ValueError("SURFACE_TYPE must be 'annulus' or 'strip'")

    fmt = _try_get_format_shortcut()

    print(f"Surface type: {SURFACE_TYPE}")
    print(f"Length N:     {N}")
    print(f"Signature:    {KLEIN_SIGNATURE}")
    print(f"DEDUPLICATE:  {DEDUPLICATE}")
    if not DEDUPLICATE:
        approx_total = _factorial(N)
        if LIMIT_EXAMPLES is not None:
            approx_total = min(approx_total, LIMIT_EXAMPLES)
        print(f"Examples (max): {approx_total}")
    else:
        if LIMIT_EXAMPLES is not None:
            print(f"Examples (max): {LIMIT_EXAMPLES}")
        else:
            print("Examples:     (deduplicated; count computed during run)")
    print("Search bounds:")
    print(f"  MAX_STEPS={MAX_STEPS}")
    print(f"  MAX_NODES={MAX_NODES}")
    print(f"  SIMPLE_ONLY={SIMPLE_ONLY}  (dx == lambda)")
    print(f"Pretty printing on success: {PRETTY_PRINT_ON_SUCCESS}")
    print(f"Print blank on failure:     {PRINT_BLANK_ON_FAILURE}")
    print("")

    tested = 0
    found = 0

    for surface_obj, perm in _enumerate_surfaces():
        tested += 1

        try:
            res = _call_find_shortcut(surface_obj)
        except Exception as e:
            if perm is None:
                print(f"[{tested}] ERROR: {type(e).__name__}: {e}")
            else:
                print(f"[{tested}] perm={_format_perm(perm)}  ERROR: {type(e).__name__}: {e}")
            # Still allow inspection if requested
            if PRINT_BLANK_ON_FAILURE:
                st0 = _blank_trackstate_for_surface(surface_obj)
                print("")
                print(render_trackstate(st0, surface_obj, cfg=PRETTY_CONFIG))
                print("")
            continue

        ok = (res is not None)
        if ok:
            found += 1

        if perm is None:
            print(f"[{tested}] shortcut={ok}")
        else:
            print(f"[{tested}] perm={_format_perm(perm)}  shortcut={ok}")

        if ok:
            # Summary line
            if fmt is not None:
                try:
                    print(" ", fmt(res))
                except Exception:
                    pass
            else:
                attrs = []
                for k in ("lam", "dx", "dy"):
                    if hasattr(res, k):
                        attrs.append(f"{k}={getattr(res, k)}")
                if attrs:
                    print("  shortcut:", ", ".join(attrs))

            if PRINT_WITNESS_WORD and hasattr(res, "word"):
                try:
                    w = getattr(res, "word")
                    parts = [getattr(x, "value", str(x)).upper() for x in w]
                    print("  word:", " ".join(parts))
                except Exception:
                    pass

            if PRETTY_PRINT_ON_SUCCESS and hasattr(res, "st"):
                try:
                    print("")
                    print(render_trackstate(res.st, surface_obj, cfg=PRETTY_CONFIG))
                    print("")
                except Exception as e:
                    print(f"  (pretty print failed: {type(e).__name__}: {e})")
        else:
            if PRINT_BLANK_ON_FAILURE:
                try:
                    st0 = _blank_trackstate_for_surface(surface_obj)
                    print("")
                    print(render_trackstate(st0, surface_obj, cfg=PRETTY_CONFIG))
                    print("")
                except Exception as e:
                    print(f"  (blank pretty print failed: {type(e).__name__}: {e})")

        # Optional early stop:
        # if ok: break

    print("")
    print(f"Done. Tested {tested} examples. Shortcuts found in {found}.")


if __name__ == "__main__":
    main()

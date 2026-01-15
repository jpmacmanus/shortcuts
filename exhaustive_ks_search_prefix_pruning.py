#!/usr/bin/env python3
"""
exhaustive_ks_search_prefix_pruning.py

Prefix-pruned exhaustive search for shortcuts over permutations of a fixed Klein signature.

Idea implemented (as per your description):
- Let S = (s0, ..., s_{N-1}).
- For m = 2,3,...,N, consider the prefix S_m = (s0, ..., s_{m-1}).
- Maintain a basket B_m of "bad cases": permutations (up to surface symmetry) of length m
  for which NO shortcut is found (within the configured search bounds).
- Initial stage m=2: enumerate all (deduped) permutations of {0,1}, test each, and store the bad ones.
- Inductive step m -> m+1:
    Generate candidates by taking each bad permutation in B_m and inserting the new label m
    into every position (0..m). Deduplicate candidates up to surface symmetry.
    Test each candidate; store the failures as the new basket B_{m+1}.
- Terminate at m=N. The final basket is the set of bad permutations for the full signature.

This script is “standalone” in the sense of the project; it uses the canonical modules:
- klein_signature_generator.parse_klein_signature
- klein_signature_generator.build_surface_from_signature
- search_shortcuts.find_shortcut
- ascii_pretty.render_trackstate (optional printing)

Configuration is via the variables in the next section.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from klein_signature_generator import parse_klein_signature, build_surface_from_signature

from search_shortcuts import find_shortcut
from ascii_pretty import render_trackstate, RenderConfig

from annulus import Side
from chord_diagram import BoundaryPoint, PortKind
from track_state import initial_state


# ----------------------------
# Configuration
# ----------------------------

SURFACE_TYPE = "strip"  # "annulus" or "strip"

# Full signature S = (s0, ..., s_{N-1})
# Accepts: "I R V H I ..." or "IRVHI..." etc.
KLEIN_SIGNATURE = "I V R H I"

# Search bounds (passed through to find_shortcut where supported)
MAX_STEPS: Optional[int] = None
MAX_NODES: int = 20000

# Require simple shortcuts? (dx == lambda). 
# NOTE THAT PREFIX PRUNING AS IS DONE HERE IS ONLY LOGICALLY SOUND FOR SIMPLE SHORTCUTS
SIMPLE_ONLY: bool = True 

# Prefix search controls
START_PREFIX_LENGTH: int = 2      # must be >=2
STOP_EARLY_IF_BASKET_EMPTY: bool = True

# Symmetry deduplication per stage:
# - annulus: cyclic shifts + reversal
# - strip: reversal
DEDUPLICATE_BY_SURFACE_SYMMETRY: bool = True

# Output controls
PRINT_PROGRESS: bool = True
PRINT_EACH_BAD_CASE: bool = True          # prints permutation representatives that fail at each stage
PRINT_EACH_SUCCESS: bool = True           # prints successes (can be huge)
PRETTY_PRINT_SUCCESS: bool = True
PRETTY_PRINT_FAILURE_BLANK: bool = True   # print blank surface when no shortcut found
PRETTY_CFG = RenderConfig(width=22, height=11, pad_between=3)


# ----------------------------
# Symmetry canonicalisation on permutations
# ----------------------------

def _rotations(seq: Tuple[int, ...]) -> Iterable[Tuple[int, ...]]:
    n = len(seq)
    for k in range(n):
        yield seq[k:] + seq[:k]


def canonical_perm_under_surface_symmetry(perm: Tuple[int, ...], surface: str) -> Tuple[int, ...]:
    """
    Canonical representative of a permutation under surface symmetries acting on POSITIONS:

    Annulus: dihedral action on the linear representation of the cycle = all rotations and reversals.
    Strip: reversal only.
    """
    if not DEDUPLICATE_BY_SURFACE_SYMMETRY:
        return perm

    if surface == "strip":
        rev = tuple(reversed(perm))
        return min(perm, rev)

    if surface == "annulus":
        best = None
        for r in _rotations(perm):
            cand1 = r
            cand2 = tuple(reversed(r))
            if best is None or cand1 < best:
                best = cand1
            if cand2 < best:
                best = cand2
        return best  # type: ignore

    raise ValueError("surface must be 'annulus' or 'strip'")


# ----------------------------
# Prefix-pruning mechanics
# ----------------------------

def insert_label_everywhere(base: Tuple[int, ...], new_label: int) -> Iterable[Tuple[int, ...]]:
    """
    Insert new_label into base at every position.
    base is a permutation of labels {0,...,m-1}; new_label is m.
    """
    m = len(base)
    for pos in range(m + 1):
        yield base[:pos] + (new_label,) + base[pos:]


def enumerate_stage_m_perms(m: int, surface: str) -> List[Tuple[int, ...]]:
    """
    Enumerate all permutations of {0,...,m-1}, optionally deduped by surface symmetries.
    Only used for the initial stage (or if you choose to start later).
    """
    reps: Set[Tuple[int, ...]] = set()
    for perm in permutations(range(m)):
        p = tuple(perm)
        reps.add(canonical_perm_under_surface_symmetry(p, surface))
    return sorted(reps)


def blank_trackstate_for_surface(surface_obj) -> object:
    """
    Construct an empty TrackState (no chords) anchored on a marked edge so
    ascii_pretty can render marked-pair tags; fallback to LEFT if needed.
    """
    for i in range(surface_obj.N):
        if surface_obj.is_marked(surface_obj.edge(Side.TOP, i)):
            return initial_state(i, BoundaryPoint(PortKind.TOP, 1))
    for i in range(surface_obj.N):
        if surface_obj.is_marked(surface_obj.edge(Side.BOTTOM, i)):
            return initial_state(i, BoundaryPoint(PortKind.BOTTOM, 1))
    return initial_state(0, BoundaryPoint(PortKind.LEFT, 1))


@dataclass
class StageStats:
    tested: int = 0
    good: int = 0
    bad: int = 0


def run_stage(
    sig_prefix,
    candidates: Sequence[Tuple[int, ...]],
    surface: str,
) -> Tuple[Set[Tuple[int, ...]], StageStats]:
    """
    Test each candidate permutation for the prefix signature.
    Returns (new_bad_basket, stats).
    """
    bad: Set[Tuple[int, ...]] = set()
    stats = StageStats()

    for perm in candidates:
        stats.tested += 1
        surface_obj = build_surface_from_signature(sig_prefix, perm, surface=surface)

        kwargs = {
            "max_steps": MAX_STEPS,
            "max_nodes": MAX_NODES,
            "simple_only": SIMPLE_ONLY,
        }
        # find_shortcut is expected to accept these; if it doesn't, it will throw early.
        res = find_shortcut(surface_obj, **kwargs)

        if res is None:
            stats.bad += 1
            bad.add(perm)

            if PRINT_EACH_BAD_CASE:
                print(f"  BAD perm={perm}")

            if PRETTY_PRINT_FAILURE_BLANK:
                st0 = blank_trackstate_for_surface(surface_obj)
                print(render_trackstate(st0, surface_obj, cfg=PRETTY_CFG))
                print()
        else:
            stats.good += 1
            if PRINT_EACH_SUCCESS:
                lam = getattr(res, "lam", None)
                dx = getattr(res, "dx", None)
                dy = getattr(res, "dy", None)
                print(f"  GOOD perm={perm}  lambda={lam} dx={dx} dy={dy} steps={len(res.word)}")
            if PRETTY_PRINT_SUCCESS:
                try:
                    print(render_trackstate(res.st, surface_obj, cfg=PRETTY_CFG))
                    print()
                except Exception as e:
                    print(f"  (pretty print failed: {type(e).__name__}: {e})")

    return bad, stats


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    sig_full = parse_klein_signature(KLEIN_SIGNATURE)
    N = len(sig_full)

    if SURFACE_TYPE not in ("annulus", "strip"):
        raise ValueError("SURFACE_TYPE must be 'annulus' or 'strip'")
    if START_PREFIX_LENGTH < 2 or START_PREFIX_LENGTH > N:
        raise ValueError("START_PREFIX_LENGTH must satisfy 2 <= START_PREFIX_LENGTH <= len(signature)")

    if PRINT_PROGRESS:
        print(f"Surface type: {SURFACE_TYPE}")
        print(f"Signature length N: {N}")
        print(f"Signature: {KLEIN_SIGNATURE}")
        print(f"Prefix start length: {START_PREFIX_LENGTH}")
        print(f"DEDUPLICATE_BY_SURFACE_SYMMETRY: {DEDUPLICATE_BY_SURFACE_SYMMETRY}")
        print(f"Search bounds: MAX_STEPS={MAX_STEPS}, MAX_NODES={MAX_NODES}, SIMPLE_ONLY={SIMPLE_ONLY}")
        print()

    # Initialise basket at m = START_PREFIX_LENGTH by brute enumeration at that stage.
    m0 = START_PREFIX_LENGTH
    sig0 = sig_full[:m0]

    if PRINT_PROGRESS:
        print(f"Stage m={m0}: enumerating all (deduped) perms of 0..{m0-1}")

    initial_candidates = enumerate_stage_m_perms(m0, SURFACE_TYPE)
    basket, stats = run_stage(sig0, initial_candidates, SURFACE_TYPE)

    if PRINT_PROGRESS:
        print(f"Stage m={m0} done: tested={stats.tested} good={stats.good} bad={stats.bad}")
        print(f"  basket size = {len(basket)}")
        print()

    if STOP_EARLY_IF_BASKET_EMPTY and not basket:
        if PRINT_PROGRESS:
            print("Basket is empty. All extensions are automatically good; terminating early.")
        return

    # Inductively extend
    for m in range(m0, N):
        # We have basket of perms on labels 0..m-1 failing for signature prefix length m.
        # Now extend to m+1 by inserting new label m.
        new_label = m
        sig_prefix = sig_full[: m + 1]

        if PRINT_PROGRESS:
            print(f"Stage m={m+1}: extending basket of size {len(basket)} by inserting label {new_label}")

        candidates_set: Set[Tuple[int, ...]] = set()
        for base in basket:
            for ext in insert_label_everywhere(base, new_label):
                rep = canonical_perm_under_surface_symmetry(ext, SURFACE_TYPE)
                candidates_set.add(rep)

        candidates = sorted(candidates_set)

        if PRINT_PROGRESS:
            print(f"  generated candidates (after dedup) = {len(candidates)}")

        basket, stats = run_stage(sig_prefix, candidates, SURFACE_TYPE)

        if PRINT_PROGRESS:
            print(f"Stage m={m+1} done: tested={stats.tested} good={stats.good} bad={stats.bad}")
            print(f"  basket size = {len(basket)}")
            print()

        if STOP_EARLY_IF_BASKET_EMPTY and not basket:
            if PRINT_PROGRESS:
                print("Basket is empty. All further extensions are automatically good; terminating early.")
            return

    # Final report
    if PRINT_PROGRESS:
        print("Completed full length.")
        print(f"Final basket size (bad cases at full length): {len(basket)}")
        if len(basket) <= 50:
            print("Bad representatives:")
            for p in sorted(basket):
                print(" ", p)


if __name__ == "__main__":
    main()

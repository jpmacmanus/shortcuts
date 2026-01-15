"""
klein_signature_generator.py

Generate marked annuli/strips from a Klein signature, then generate further
examples by permuting squares as *tiles* (markings move with squares and keep
their OP/OR status).

Pipeline (as per your latest clarification):
  1) Build the standard chain annulus/strip for the signature S on Q0..Q_{N-1}.
  2) Permute squares (tiles) by reordering the chain positions.
     The marked boundary identifications are transported with the squares,
     and orientation (OP/OR) is preserved.
  3) Reindex squares by their new chain positions 0..N-1 to fit MarkedAnnulus/MarkedStrip.

Deduplication:
- Annulus: quotient by dihedral action on positions (size 2N), so typically N!/(2N).
- Strip: quotient by reversal on positions (size 2), so typically N!/2.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterable, List, Sequence, Tuple

from annulus import MarkedAnnulus, MarkedStrip, Side, BoundaryEdge


# ----------------------------
# Klein group representation
# ----------------------------

class KleinElt:
    """
    Element of the Klein four-group, represented by two bits:
      - swap_tb: swaps TOP/BOTTOM on that square
      - flip_orient: contributes to OP/OR parity across a marked identification
    """
    __slots__ = ("name", "swap_tb", "flip_orient")

    def __init__(self, name: str, swap_tb: bool, flip_orient: bool):
        self.name = name
        self.swap_tb = swap_tb
        self.flip_orient = flip_orient

    def __repr__(self) -> str:
        return self.name


# Convention:
# H = mirror in horizontal axis => swaps TOP/BOTTOM
# V = mirror in vertical axis   => toggles orientation-reversal bit
I = KleinElt("I", swap_tb=False, flip_orient=False)
H = KleinElt("H", swap_tb=True,  flip_orient=False)
V = KleinElt("V", swap_tb=False, flip_orient=True)
R = KleinElt("R", swap_tb=True,  flip_orient=True)

KLEIN_BY_CHAR = {"I": I, "H": H, "V": V, "R": R}


# ----------------------------
# Signature parsing
# ----------------------------

def parse_klein_signature(sig) -> List[KleinElt]:
    """
    Accepts:
      - "I,H,V,R"
      - "IHVR"
      - "I H V R"
      - ["I","H","V","R"]
    """
    if isinstance(sig, str):
        s = sig.replace(",", " ").strip()
        chars = s.split() if " " in s else list(s)
    else:
        chars = list(sig)

    out: List[KleinElt] = []
    for c in chars:
        c = str(c).strip().upper()
        if c not in KLEIN_BY_CHAR:
            raise ValueError(f"Bad Klein symbol {c!r}; expected one of I,H,V,R")
        out.append(KLEIN_BY_CHAR[c])
    return out


# ----------------------------
# Step 1: build the standard surface for a signature
# ----------------------------

def build_standard_surface(signature: List[KleinElt], surface: str = "annulus"):
    """
    Build the *standard chain* surface on squares Q0..Q_{N-1} for the given signature,
    with marked pairs between consecutive squares (i -> i+1), after applying the
    local symmetry on each square *before* any permutation.

    This is the object whose markings we will then transport under permutations.
    """
    N = len(signature)
    if surface == "annulus":
        A = MarkedAnnulus(N)
    elif surface == "strip":
        A = MarkedStrip(N)
    else:
        raise ValueError("surface must be 'annulus' or 'strip'")

    # Standard chain: for i=0..N-2, mark Top(Q_i) <-> Bottom(Q_{i+1}) initially OP,
    # then apply local square symmetries to determine which side is used and whether OP/OR flips.
    for i in range(N - 1):
        sym_i = signature[i]
        sym_j = signature[i + 1]

        side_i = Side.BOTTOM if sym_i.swap_tb else Side.TOP
        side_j = Side.TOP if sym_j.swap_tb else Side.BOTTOM

        orientation_reversing = bool(sym_i.flip_orient ^ sym_j.flip_orient)

        e1 = A.edge(side_i, i)
        e2 = A.edge(side_j, i + 1)
        A.add_marked_pair(e1, e2, orientation_reversing=orientation_reversing)

    return A


# ----------------------------
# Step 2: transport markings under a permutation of tiles
# ----------------------------

def permute_surface_tiles(A0, perm: Sequence[int], surface: str):
    """
    Given a base surface A0 on indices 0..N-1, and a permutation perm where:
      new position k contains old square perm[k],
    produce a new surface A1 whose indices are the new positions 0..N-1, and whose
    marked boundary identifications are transported with the squares.

    Transport rule:
      BoundaryEdge(side, old_i)  -> BoundaryEdge(side, newpos_of_old[old_i])
    and orientation_reversing is preserved.
    """
    N = A0.N
    if surface == "annulus":
        A1 = MarkedAnnulus(N)
    elif surface == "strip":
        A1 = MarkedStrip(N)
    else:
        raise ValueError("surface must be 'annulus' or 'strip'")

    # newpos_of_old[old] = new position
    newpos_of_old = [0] * N
    for newpos, old in enumerate(perm):
        newpos_of_old[old] = newpos

    # Copy marked pairs once each
    for p in A0.all_pairs():
        a = p.a
        b = p.b
        a1 = A1.edge(a.side, newpos_of_old[a.i])
        b1 = A1.edge(b.side, newpos_of_old[b.i])
        A1.add_marked_pair(a1, b1, orientation_reversing=p.orientation_reversing)

    return A1


# ----------------------------
# Public builder: signature + permutation
# ----------------------------

def build_surface_from_signature(signature: List[KleinElt], perm: Sequence[int], surface: str = "annulus"):
    """
    Public API (used by your exhaustive search):

    - Build the standard surface for the signature.
    - Permute tiles by perm, transporting markings and preserving OP/OR.
    """
    A0 = build_standard_surface(signature, surface=surface)
    return permute_surface_tiles(A0, perm, surface=surface)


def all_surfaces_with_signature(signature, surface: str = "annulus"):
    """
    Backwards-compatible behaviour: returns ALL N! surfaces from all permutations.
    """
    sig = parse_klein_signature(signature)
    N = len(sig)
    return [build_surface_from_signature(sig, perm, surface=surface) for perm in permutations(range(N))]


# ----------------------------
# Deduplication up to surface symmetries (permutation-level on positions)
# ----------------------------

def _act_on_positions(perm: Sequence[int], f) -> Tuple[int, ...]:
    # Apply symmetry f to positions: (perm ∘ f^{-1}) on the list representation.
    # Implemented by building the inverse image positions.
    N = len(perm)
    inv = [0] * N
    for i, v in enumerate(perm):
        inv[v] = i
    # We want new_perm[p] = perm[f^{-1}(p)].
    # f is given as a map on positions, so compute f^{-1} by brute force (N is small).
    finv = [0] * N
    for p in range(N):
        finv[f(p)] = p
    return tuple(perm[finv[p]] for p in range(N))


def _annulus_position_symmetries(N: int):
    # rotations on positions
    for k in range(N):
        yield (lambda p, k=k: (p + k) % N)
    # reflections on positions
    for k in range(N):
        yield (lambda p, k=k: (-p + k) % N)


def _strip_position_symmetries(N: int):
    yield (lambda p: p)
    yield (lambda p, N=N: (N - 1 - p))


def canonical_perm_under_symmetry(perm: Sequence[int], *, surface: str, N: int) -> Tuple[int, ...]:
    """
    Canonical representative of a permutation under symmetries of the underlying surface
    acting on POSITIONS.
    """
    perm = tuple(perm)
    if surface == "annulus":
        syms = _annulus_position_symmetries(N)
    elif surface == "strip":
        syms = _strip_position_symmetries(N)
    else:
        raise ValueError("surface must be 'annulus' or 'strip'")

    best = None
    for f in syms:
        img = _act_on_positions(perm, f)
        if best is None or img < best:
            best = img
    return best  # type: ignore


def unique_perms_under_symmetry(N: int, surface: str) -> Iterable[Tuple[int, ...]]:
    seen = set()
    for perm in permutations(range(N)):
        key = canonical_perm_under_symmetry(perm, surface=surface, N=N)
        if key in seen:
            continue
        seen.add(key)
        yield tuple(perm)


def all_unique_surfaces_with_signature(signature, surface: str = "annulus"):
    """
    Return distinct marked annuli/strips for the given Klein signature, modulo
    surface symmetries acting on positions.
    """
    sig = parse_klein_signature(signature)
    N = len(sig)
    out = []
    for perm in unique_perms_under_symmetry(N, surface=surface):
        out.append(build_surface_from_signature(sig, perm, surface=surface))
    return out

"""
klein_signature_generator.py

Generate marked annuli/strips from a Klein signature, and optionally
enumerate all permutations (with or without symmetry deduplication).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, List, Sequence, Tuple

from marked_strips import MarkedAnnulus, MarkedStrip
from square import Side
from strips import Annulus, SquareStrip


# ----------------------------
# Klein group representation
# ----------------------------


@dataclass(frozen=True)
class KleinElt:
    """
    Element of the Klein four-group, represented by two bits:
      - swap_tb: swaps TOP/BOTTOM on that square
      - flip_orient: contributes to OP/OR parity across a marked identification
    """

    name: str
    swap_tb: bool
    flip_orient: bool

    def __repr__(self) -> str:
        return self.name


# Convention:
# H = mirror in horizontal axis => swaps TOP/BOTTOM
# V = mirror in vertical axis   => toggles orientation-reversal bit
I = KleinElt("I", swap_tb=False, flip_orient=False)
H = KleinElt("H", swap_tb=True, flip_orient=False)
V = KleinElt("V", swap_tb=False, flip_orient=True)
R = KleinElt("R", swap_tb=True, flip_orient=True)

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


def build_standard_surface(signature: List[KleinElt], *, surface: str = "annulus"):
    """
    Build the standard chain surface on Q0..Q_{N-1} for the given signature,
    with marked pairs between consecutive squares (i -> i+1).
    """
    n = len(signature)
    if surface == "annulus":
        base = MarkedAnnulus(annulus=Annulus.build(n))
    elif surface == "strip":
        base = MarkedStrip(strip=SquareStrip.build(n))
    else:
        raise ValueError("surface must be 'annulus' or 'strip'")

    for i in range(n - 1):
        sym_i = signature[i]
        sym_j = signature[i + 1]

        side_i = Side.BOTTOM if sym_i.swap_tb else Side.TOP
        side_j = Side.TOP if sym_j.swap_tb else Side.BOTTOM

        orientation_reversing = bool(sym_i.flip_orient ^ sym_j.flip_orient)
        e1 = base.edge(side_i, i)
        e2 = base.edge(side_j, i + 1)
        base.add_marked_pair(e1, e2, orientation_reversing=orientation_reversing)

    return base


# ----------------------------
# Step 2: transport markings under a permutation of tiles
# ----------------------------


def permute_surface_tiles(base, perm: Sequence[int], *, surface: str):
    """
    Given a base surface and a permutation perm where:
      new position k contains old square perm[k],
    produce a new surface whose indices are the new positions 0..N-1, and whose
    marked boundary identifications are transported with the squares.
    """
    n = base.N
    if surface == "annulus":
        out = MarkedAnnulus(annulus=Annulus.build(n))
    elif surface == "strip":
        out = MarkedStrip(strip=SquareStrip.build(n))
    else:
        raise ValueError("surface must be 'annulus' or 'strip'")

    newpos_of_old = [0] * n
    for newpos, old in enumerate(perm):
        newpos_of_old[old] = newpos

    for p in base.all_pairs():
        a = p.a
        b = p.b
        a1 = out.edge(a.side, newpos_of_old[a.i])
        b1 = out.edge(b.side, newpos_of_old[b.i])
        out.add_marked_pair(a1, b1, orientation_reversing=p.orientation_reversing)

    return out


def build_surface_from_signature(signature: List[KleinElt], perm: Sequence[int], *, surface: str = "annulus"):
    """
    Build the standard surface, then permute tiles by perm.
    """
    base = build_standard_surface(signature, surface=surface)
    return permute_surface_tiles(base, perm, surface=surface)


# ----------------------------
# Deduplication up to surface symmetries
# ----------------------------


def _act_on_positions(perm: Sequence[int], f) -> Tuple[int, ...]:
    n = len(perm)
    inv = [0] * n
    for i, v in enumerate(perm):
        inv[v] = i
    finv = [0] * n
    for p in range(n):
        finv[f(p)] = p
    return tuple(perm[finv[p]] for p in range(n))


def _annulus_position_symmetries(n: int):
    for k in range(n):
        yield (lambda p, k=k: (p + k) % n)
    for k in range(n):
        yield (lambda p, k=k: (-p + k) % n)


def _strip_position_symmetries(n: int):
    yield (lambda p: p)
    yield (lambda p, n=n: (n - 1 - p))


def canonical_perm_under_symmetry(perm: Sequence[int], *, surface: str, n: int) -> Tuple[int, ...]:
    perm = tuple(perm)
    if surface == "annulus":
        syms = _annulus_position_symmetries(n)
    elif surface == "strip":
        syms = _strip_position_symmetries(n)
    else:
        raise ValueError("surface must be 'annulus' or 'strip'")

    best = None
    for f in syms:
        img = _act_on_positions(perm, f)
        if best is None or img < best:
            best = img
    return best  # type: ignore[return-value]


def unique_perms_under_symmetry(n: int, surface: str) -> Iterable[Tuple[int, ...]]:
    seen = set()
    for perm in permutations(range(n)):
        key = canonical_perm_under_symmetry(perm, surface=surface, n=n)
        if key in seen:
            continue
        seen.add(key)
        yield tuple(perm)


# ----------------------------
# Public helpers
# ----------------------------


def surfaces_from_signature(
    signature,
    *,
    surface: str = "annulus",
    unique: bool = True,
    exclude_adjacent_I: bool = False,
):
    """
    Return a list of marked annuli/strips for a Klein signature.
    """
    sig = parse_klein_signature(signature)
    n = len(sig)

    if unique:
        perms = unique_perms_under_symmetry(n, surface=surface)
    else:
        perms = permutations(range(n))

    out = []
    for perm in perms:
        if exclude_adjacent_I and _perm_has_adjacent_I(sig, perm, surface=surface):
            continue
        out.append(build_surface_from_signature(sig, perm, surface=surface))
    return out


def all_unique_surfaces_with_signature(
    signature, *, surface: str = "annulus", exclude_adjacent_I: bool = False
):
    """
    Backwards-compatible name for unique surface generation.
    """
    return surfaces_from_signature(
        signature, surface=surface, unique=True, exclude_adjacent_I=exclude_adjacent_I
    )


def _perm_has_adjacent_I(sig, perm, *, surface: str) -> bool:
    """
    Return True if the permutation places any two I's adjacent.
    """
    n = len(sig)
    is_I = [s.name == "I" for s in sig]

    for i in range(n):
        j = (i + 1) % n
        if surface == "strip" and i == n - 1:
            break
        if is_I[perm[i]] and is_I[perm[j]]:
            return True
    return False

"""
directed_klein_signature_generator.py

Directed variants of Klein signature surface generation.
"""

from __future__ import annotations

from itertools import permutations
from typing import Iterable, List, Sequence, Tuple

from directed_marked_strips import DirectedMarkedAnnulus, DirectedMarkedStrip, EdgeDir
from klein_signature_generator import (
    KleinElt,
    parse_klein_signature,
)
from marked_strips import BoundaryEdge, EdgeRef, MarkedAnnulus, MarkedStrip
from square import Side
from strips import Annulus, SquareStrip


def build_standard_directed_surface(signature: List[KleinElt], *, surface: str = "annulus"):
    """
    Build the standard directed surface for a signature.
    """
    n = len(signature)
    if surface == "annulus":
        base = MarkedAnnulus(annulus=Annulus.build(n))
        directed = DirectedMarkedAnnulus(base=base)
    elif surface == "strip":
        base = MarkedStrip(strip=SquareStrip.build(n))
        directed = DirectedMarkedStrip(base=base)
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

    # Marked boundary edges inherit direction; unmarked boundary edges stay undirected.
    for i, sym in enumerate(signature):
        for side in (Side.TOP, Side.BOTTOM):
            e_ref = EdgeRef(side, i)
            if not base.is_marked_edge(e_ref):
                continue
            if sym.swap_tb:
                directed.set_edge_direction(
                    EdgeRef(Side.TOP, i), EdgeDir.OUT
                )
                directed.set_edge_direction(
                    EdgeRef(Side.BOTTOM, i), EdgeDir.IN
                )
            else:
                directed.set_edge_direction(
                    EdgeRef(Side.TOP, i), EdgeDir.IN
                )
                directed.set_edge_direction(
                    EdgeRef(Side.BOTTOM, i), EdgeDir.OUT
                )

    # Interior edge directions: annulus directed, strip undirected.
    if surface == "annulus":
        for i in range(n):
            directed.set_edge_direction(EdgeRef(Side.RIGHT, i), EdgeDir.OUT)
            directed.set_edge_direction(EdgeRef(Side.LEFT, i), EdgeDir.IN)

    return directed


def permute_directed_surface_tiles(base, perm: Sequence[int], *, surface: str):
    """
    Permute a directed surface by tile permutation.
    """
    n = base.N
    if surface == "annulus":
        out_base = MarkedAnnulus(annulus=Annulus.build(n))
        out = DirectedMarkedAnnulus(base=out_base)
    elif surface == "strip":
        out_base = MarkedStrip(strip=SquareStrip.build(n))
        out = DirectedMarkedStrip(base=out_base)
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
        out_base.add_marked_pair(a1, b1, orientation_reversing=p.orientation_reversing)

    for e in base.all_edge_refs():
        d = base.edge_direction(e)
        new_e = EdgeRef(e.side, newpos_of_old[e.i])
        out.set_edge_direction(new_e, d)

    return out


def build_directed_surface_from_signature(
    signature: List[KleinElt], perm: Sequence[int], *, surface: str = "annulus"
):
    base = build_standard_directed_surface(signature, surface=surface)
    return permute_directed_surface_tiles(base, perm, surface=surface)


def _act_on_positions(perm: Sequence[int], f) -> Tuple[int, ...]:
    n = len(perm)
    inv = [0] * n
    for i, v in enumerate(perm):
        inv[v] = i
    finv = [0] * n
    for p in range(n):
        finv[f(p)] = p
    return tuple(perm[finv[p]] for p in range(n))


def _directed_annulus_position_symmetries(n: int):
    for k in range(n):
        yield (lambda p, k=k: (p + k) % n)


def _directed_strip_position_symmetries(n: int):
    yield (lambda p: p)
    yield (lambda p, n=n: (n - 1 - p))


def canonical_perm_under_directed_symmetry(
    perm: Sequence[int], *, surface: str, n: int
) -> Tuple[int, ...]:
    perm = tuple(perm)
    if surface == "annulus":
        syms = _directed_annulus_position_symmetries(n)
    elif surface == "strip":
        syms = _directed_strip_position_symmetries(n)
    else:
        raise ValueError("surface must be 'annulus' or 'strip'")

    best = None
    for f in syms:
        img = _act_on_positions(perm, f)
        if best is None or img < best:
            best = img
    return best  # type: ignore[return-value]


def unique_perms_under_directed_symmetry(n: int, surface: str) -> Iterable[Tuple[int, ...]]:
    seen = set()
    for perm in permutations(range(n)):
        key = canonical_perm_under_directed_symmetry(perm, surface=surface, n=n)
        if key in seen:
            continue
        seen.add(key)
        yield tuple(perm)


def directed_surfaces_from_signature(
    signature,
    *,
    surface: str = "annulus",
    unique: bool = True,
):
    """
    Return a list of directed marked annuli/strips for a Klein signature.
    """
    sig = parse_klein_signature(signature)
    n = len(sig)

    if unique:
        perms = unique_perms_under_directed_symmetry(n, surface=surface)
    else:
        perms = permutations(range(n))

    out = []
    for perm in perms:
        out.append(build_directed_surface_from_signature(sig, perm, surface=surface))
    return out

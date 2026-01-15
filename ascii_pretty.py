# ascii_pretty.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Iterable, Any
import inspect

from annulus import MarkedAnnulus, Side, BoundaryEdge
from chord_diagram import BoundaryPoint, PortKind, SquareChordSet
from track_state import TrackState

_LABELS = "123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _label_for(k: int) -> str:
    return _LABELS[k] if k < len(_LABELS) else _LABELS[k % len(_LABELS)]


@dataclass(frozen=True)
class RenderConfig:
    width: int = 22
    height: int = 11
    pad_between: int = 3


def _inner_w(cfg: RenderConfig) -> int:
    return cfg.width - 2


def _distribute(n: int, span: int) -> List[int]:
    if n <= 0:
        return []
    if n == 1:
        return [span // 2]
    step = (span - 1) / (n - 1)
    xs = [int(round(i * step)) for i in range(n)]
    out: List[int] = []
    for x in xs:
        out.append(x if not out else max(x, out[-1] + 1))
    # clamp
    out = [min(max(0, x), span - 1) for x in out]
    return out


def _square_canvas(cfg: RenderConfig) -> List[List[str]]:
    W, H = cfg.width, cfg.height
    c = [[" " for _ in range(W)] for _ in range(H)]

    c[0][0] = "┌"
    c[0][W - 1] = "┐"
    c[H - 1][0] = "└"
    c[H - 1][W - 1] = "┘"
    for x in range(1, W - 1):
        c[0][x] = "─"
        c[H - 1][x] = "─"
    for y in range(1, H - 1):
        c[y][0] = "│"
        c[y][W - 1] = "│"
    return c


def _merge_line(cur: str, new: str) -> str:
    if cur == " ":
        return new
    if cur == new:
        return cur
    return "┼"


def _plot_interior(c, r, x, ch):
    if 0 < r < len(c) - 1 and 0 < x < len(c[0]) - 1:
        c[r][x] = _merge_line(c[r][x], ch)


def _plot_boundary(c, r, x, ch):
    if 0 <= r < len(c) and 0 <= x < len(c[0]):
        if c[r][x] not in "┌┐└┘":
            c[r][x] = ch


def _draw_polyline(c, pts):
    for (r1, x1), (r2, x2) in zip(pts, pts[1:]):
        if r1 == r2:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                _plot_interior(c, r1, x, "─")
        elif x1 == x2:
            for r in range(min(r1, r2), max(r1, r2) + 1):
                _plot_interior(c, r, x1, "│")


# ----------------------------
# Robust SquareChordSet handling (fixes your crash)
# ----------------------------

def _empty_square_chordset() -> SquareChordSet:
    """
    Construct an "empty" SquareChordSet in a way that is robust to
    constructor signature differences across your edits.
    """
    sig = inspect.signature(SquareChordSet)
    params = list(sig.parameters.values())

    # Prefer keywords if available.
    kw = {}
    for name in ("top_slots", "top", "t"):
        if name in sig.parameters:
            kw[name] = 0
            break
    for name in ("bottom_slots", "bottom", "b"):
        if name in sig.parameters:
            kw[name] = 0
            break
    for name in ("chords", "edges", "pairs", "chord_pairs"):
        if name in sig.parameters:
            kw[name] = set()
            break

    if kw:
        try:
            return SquareChordSet(**kw)  # type: ignore
        except TypeError:
            pass

    # Fallback: try the common positional form (top_slots, bottom_slots, chords)
    try:
        return SquareChordSet(0, 0, set())  # type: ignore
    except TypeError:
        pass

    # Last resort: (top_slots, bottom_slots)
    return SquareChordSet(0, 0)  # type: ignore


def _get_chords(sq: SquareChordSet) -> List[Tuple[BoundaryPoint, BoundaryPoint]]:
    """
    Robustly obtain chord pairs from a SquareChordSet.
    In some intermediate refactors, sq.chords may be absent or mis-set; we tolerate that.
    """
    # Most common: sq.chords is an iterable of (BoundaryPoint, BoundaryPoint)
    if hasattr(sq, "chords"):
        v = getattr(sq, "chords")
        if v is None:
            return []
        if isinstance(v, int):
            return []
        if isinstance(v, dict):
            # chord maps sometimes store chords as keys
            return list(v.keys())
        try:
            return list(v)
        except TypeError:
            return []

    # Alternative attribute names
    for name in ("pairs", "chord_pairs", "edges"):
        if hasattr(sq, name):
            v = getattr(sq, name)
            if v is None or isinstance(v, int):
                return []
            if isinstance(v, dict):
                return list(v.keys())
            try:
                return list(v)
            except TypeError:
                return []

    return []


def _get_top_slots(sq: SquareChordSet) -> int:
    for name in ("top_slots", "top"):
        if hasattr(sq, name):
            v = getattr(sq, name)
            return int(v) if isinstance(v, int) else int(v or 0)
    return 0


def _get_bottom_slots(sq: SquareChordSet) -> int:
    for name in ("bottom_slots", "bottom"):
        if hasattr(sq, name):
            v = getattr(sq, name)
            return int(v) if isinstance(v, int) else int(v or 0)
    return 0


def _uses_left(sq: SquareChordSet) -> bool:
    if hasattr(sq, "uses_left") and callable(getattr(sq, "uses_left")):
        return bool(sq.uses_left())
    # fallback: inspect chords
    for a, b in _get_chords(sq):
        if a.kind == PortKind.LEFT or b.kind == PortKind.LEFT:
            return True
    return False


def _uses_right(sq: SquareChordSet) -> bool:
    if hasattr(sq, "uses_right") and callable(getattr(sq, "uses_right")):
        return bool(sq.uses_right())
    for a, b in _get_chords(sq):
        if a.kind == PortKind.RIGHT or b.kind == PortKind.RIGHT:
            return True
    return False


# ----------------------------
# Boundary positions
# ----------------------------

def _boundary_positions(sq: SquareChordSet, cfg: RenderConfig):
    W, H = cfg.width, cfg.height
    iw = _inner_w(cfg)

    top_slots = _get_top_slots(sq)
    bottom_slots = _get_bottom_slots(sq)

    pos = {}
    for k, x in enumerate(_distribute(top_slots, iw), start=1):
        pos[BoundaryPoint(PortKind.TOP, k)] = (0, 1 + x)
    for k, x in enumerate(_distribute(bottom_slots, iw), start=1):
        pos[BoundaryPoint(PortKind.BOTTOM, k)] = (H - 1, 1 + x)

    mid = H // 2
    if _uses_left(sq):
        pos[BoundaryPoint(PortKind.LEFT, 1)] = (mid, 0)
    if _uses_right(sq):
        pos[BoundaryPoint(PortKind.RIGHT, 1)] = (mid, W - 1)
    return pos


# ----------------------------
# Marked-edge annotations
# ----------------------------

def _pair_annotations(ann: MarkedAnnulus):
    edge_tag: Dict[BoundaryEdge, str] = {}
    lines = ["Marked boundary identifications (ID· = OP, ID↻ = OR):"]

    pairs = ann.all_pairs()
    for i, p in enumerate(pairs):
        lab = _label_for(i)
        glyph = "↻" if p.orientation_reversing else "·"
        edge_tag[p.a] = lab + glyph
        edge_tag[p.b] = lab + glyph
        lines.append(f"  {lab}: {p.a} ↔ {p.b}   ({'OR' if p.orientation_reversing else 'OP'})")

    if len(pairs) == 0:
        lines.append("  (none)")
    return edge_tag, lines


def _render_square(
    i: int,
    sq: SquareChordSet,
    cfg: RenderConfig,
    current: Optional[BoundaryPoint],
    edge_tag: Dict[BoundaryEdge, str],
):
    c = _square_canvas(cfg)
    pos = _boundary_positions(sq, cfg)

    # Pair-ID tags on boundary edges
    for side in (Side.TOP, Side.BOTTOM):
        e = BoundaryEdge(side, i)
        if e in edge_tag:
            tag = edge_tag[e]
            r = 0 if side == Side.TOP else cfg.height - 1
            for j, ch in enumerate(tag):
                _plot_boundary(c, r, 2 + j, ch)

    # Current position marker
    if current is not None and current in pos:
        r, x = pos[current]
        _plot_boundary(c, r, x, "@")

    # Chords
    chords = _get_chords(sq)
    # Deterministic ordering
    chords = sorted(chords, key=lambda ab: (ab[0].kind.value, ab[0].slot, ab[1].kind.value, ab[1].slot))

    for idx, (a, b) in enumerate(chords):
        lab = _label_for(idx)

        if a in pos:
            ra, xa = pos[a]
            _plot_boundary(c, ra, xa, lab)
        if b in pos:
            rb, xb = pos[b]
            _plot_boundary(c, rb, xb, lab)

        # route chord via a central horizontal “bus” to keep things readable
        H, W = cfg.height, cfg.width

        def nudge_in(p: BoundaryPoint) -> Tuple[int, int]:
            if p.kind == PortKind.TOP:
                r, x = pos[p]
                return (1, x)
            if p.kind == PortKind.BOTTOM:
                r, x = pos[p]
                return (H - 2, x)
            if p.kind == PortKind.LEFT:
                r, x = pos[p]
                return (r, 1)
            if p.kind == PortKind.RIGHT:
                r, x = pos[p]
                return (r, W - 2)
            r, x = pos[p]
            return (r, x)

        if a not in pos or b not in pos:
            continue

        pa = nudge_in(a)
        pb = nudge_in(b)
        bus_r = H // 2

        pts = [pa, (bus_r, pa[1]), (bus_r, pb[1]), pb]
        _draw_polyline(c, pts)

    return ["".join(row) for row in c]


def render_trackstate(
    st: TrackState,
    ann: Optional[MarkedAnnulus] = None,
    *,
    cfg: Optional[RenderConfig] = None,
) -> str:
    """
    Public interface preserved.

    Produces:
      - small header line with lambda and current position
      - marked-pair block (with OP/OR)
      - row of squares with:
          * pair IDs placed on top/bottom edges
          * chord endpoints labelled 1,2,3,...
          * chord routing drawn inside squares
    """
    if cfg is None:
        cfg = RenderConfig()

    square_map = dict(st.squares)

    if ann is not None:
        N = ann.N
    else:
        N = max(square_map.keys(), default=-1) + 1

    edge_tag: Dict[BoundaryEdge, str] = {}
    pair_block: List[str] = []
    if ann is not None:
        edge_tag, pair_block = _pair_annotations(ann)

    blocks = []
    for i in range(N):
        sq = square_map.get(i)
        if sq is None:
            sq = _empty_square_chordset()
        cur = st.pos.point if st.pos.square == i else None
        blocks.append(_render_square(i, sq, cfg, cur, edge_tag))

    spacer = " " * cfg.pad_between
    out: List[str] = []
    out.append(f"TrackState: lambda={st.lam}, pos={st.pos}")

    if pair_block:
        out.append("")
        out.extend(pair_block)
        out.append("")

    for r in range(cfg.height):
        out.append(spacer.join(block[r] for block in blocks))

    return "\n".join(out)

"""
marked_strips.py

Marked strips and annuli: a strip/annulus with a disjoint set of identified
TOP/BOTTOM boundary edges, each marked as orientation-preserving (OP) or
orientation-reversing (OR).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import string
from typing import Dict, Iterable, List, Tuple

from edge import Port
from square import Side
from strips import Annulus, SquareStrip


@dataclass(frozen=True, order=True)
class BoundaryEdge:
    """A boundary edge specified by side (TOP/BOTTOM) and square index."""

    side: Side
    i: int

    def __str__(self) -> str:
        return f"{self.side.value}{self.i}"


@dataclass(frozen=True, order=True)
class EdgeRef:
    """Any edge specified by side and square index."""

    side: Side
    i: int

    def __str__(self) -> str:
        return f"{self.side.value}{self.i}"


@dataclass(frozen=True)
class MarkedPair:
    """Pairing of two boundary edges with orientation flag."""

    a: BoundaryEdge
    b: BoundaryEdge
    orientation_reversing: bool = False  # True = OR, False = OP

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("MarkedPair endpoints must be distinct")

    def other(self, e: BoundaryEdge) -> BoundaryEdge:
        if e == self.a:
            return self.b
        if e == self.b:
            return self.a
        raise KeyError(f"Edge {e} is not an endpoint of this pair")

    def normalized_key(self) -> Tuple[BoundaryEdge, BoundaryEdge]:
        return tuple(sorted((self.a, self.b)))


@dataclass
class _MarkedBase:
    _pairs: Dict[BoundaryEdge, MarkedPair] = field(default_factory=dict, init=False)

    def _validate_edge(self, e: BoundaryEdge) -> None:
        if e.side not in (Side.TOP, Side.BOTTOM):
            raise ValueError("Only TOP/BOTTOM boundary edges may be marked")
        if not (0 <= e.i < self.N):
            raise ValueError(f"Square index {e.i} out of range for N={self.N}")

    def add_marked_pair(
        self,
        a: BoundaryEdge,
        b: BoundaryEdge,
        *,
        orientation_reversing: bool = False,
        overwrite: bool = False,
    ) -> None:
        self._validate_edge(a)
        self._validate_edge(b)
        if a == b:
            raise ValueError("Cannot pair an edge with itself")

        if not overwrite:
            if a in self._pairs:
                raise ValueError(f"Edge {a} already marked")
            if b in self._pairs:
                raise ValueError(f"Edge {b} already marked")
        else:
            for e in (a, b):
                if e in self._pairs:
                    old = self._pairs[e]
                    del self._pairs[old.a]
                    del self._pairs[old.b]

        p = MarkedPair(a=a, b=b, orientation_reversing=orientation_reversing)
        self._pairs[a] = p
        self._pairs[b] = p

    def remove_marked_edge(self, e: BoundaryEdge) -> None:
        if e not in self._pairs:
            return
        p = self._pairs[e]
        del self._pairs[p.a]
        del self._pairs[p.b]

    def is_marked(self, e: BoundaryEdge) -> bool:
        return e in self._pairs

    def pair_info(self, e: BoundaryEdge) -> Tuple[BoundaryEdge, bool]:
        p = self._pairs[e]
        return (p.other(e), p.orientation_reversing)

    def all_pairs(self) -> List[MarkedPair]:
        seen = set()
        out: List[MarkedPair] = []
        for p in self._pairs.values():
            k = p.normalized_key()
            if k not in seen:
                seen.add(k)
                out.append(p)
        out.sort(key=lambda mp: (mp.normalized_key()[0], mp.normalized_key()[1]))
        return out

    def marked_edges(self) -> List[BoundaryEdge]:
        return sorted(self._pairs.keys())

    def marked_squares(self) -> List[int]:
        return sorted({e.i for e in self._pairs.keys()})

    def _validate_edge_ref(self, e: EdgeRef) -> None:
        if e.side not in (Side.TOP, Side.BOTTOM, Side.LEFT, Side.RIGHT):
            raise ValueError("Unknown side for edge reference")
        if not (0 <= e.i < self.N):
            raise ValueError(f"Square index {e.i} out of range for N={self.N}")

    def _pair_labels(self) -> Dict[Tuple[BoundaryEdge, BoundaryEdge], str]:
        pairs = self.all_pairs()
        labels: Dict[Tuple[BoundaryEdge, BoundaryEdge], str] = {}

        def _label_for(idx: int) -> str:
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            if idx < 26:
                return alphabet[idx]
            a = idx // 26 - 1
            b = idx % 26
            return alphabet[a] + alphabet[b]

        for i, p in enumerate(pairs):
            labels[p.normalized_key()] = _label_for(i)
        return labels

    def _edge_label(self, e: BoundaryEdge, labels: Dict[Tuple[BoundaryEdge, BoundaryEdge], str]) -> str:
        if e not in self._pairs:
            return "--"
        p = self._pairs[e]
        key = p.normalized_key()
        tag = labels[key]
        return f"{tag}{'↻' if p.orientation_reversing else ''}"

    def _edge_marker(self, e: BoundaryEdge) -> str | None:
        # Direction hint for marked pairs (OP same direction, OR opposite).
        if e not in self._pairs:
            return None
        p = self._pairs[e]
        if p.orientation_reversing:
            return ">" if e == p.a else "<"
        return ">"

    def _render_square_ascii(
        self,
        square,
        *,
        width: int,
        height: int,
        padding: int,
        port_symbols: Dict[Port, str],
        interior_sides: set[Side],
        boundary_left: str,
        boundary_right: str,
        marker_top: str | None,
        marker_bottom: str | None,
        right_top_to_bottom: bool,
        edge_dir_lookup=None,
        square_index: int | None = None,
    ) -> List[str]:
        if width < 5 or height < 5:
            raise ValueError("width and height must be at least 5")
        if padding < 0:
            raise ValueError("padding must be nonnegative")

        grid = [[" " for _ in range(width)] for _ in range(height)]

        grid[0][0] = "+"
        grid[0][width - 1] = "+"
        grid[height - 1][0] = "+"
        grid[height - 1][width - 1] = "+"
        for x in range(1, width - 1):
            grid[0][x] = "-"
            grid[height - 1][x] = "-"
        for y in range(1, height - 1):
            grid[y][0] = "|"
            grid[y][width - 1] = "|"

        def _edge_dir_value(side: Side) -> str | None:
            if edge_dir_lookup is None or square_index is None:
                return None
            d = edge_dir_lookup(EdgeRef(side, square_index))
            return getattr(d, "value", d)

        def _place_edge_arrow(side: Side, row: int, col: int, on_char: str) -> None:
            d = _edge_dir_value(side)
            if d is None or d == "undirected":
                return
            if grid[row][col] != on_char:
                return
            if side == Side.TOP:
                grid[row][col] = "v" if d == "in" else "^"
            elif side == Side.BOTTOM:
                grid[row][col] = "^" if d == "in" else "v"
            elif side == Side.LEFT:
                grid[row][col] = ">" if d == "in" else "<"
            elif side == Side.RIGHT:
                grid[row][col] = "<" if d == "in" else ">"

        mid_x = width // 2
        _place_edge_arrow(Side.TOP, 0, width - 2, "-")
        _place_edge_arrow(Side.BOTTOM, height - 1, width - 2, "-")

        if boundary_left == "double":
            for y in range(1, height - 1):
                grid[y][0] = "‖"
        elif boundary_left == "caret":
            mid = height // 2
            for y in (mid - 1, mid, mid + 1):
                if 0 < y < height - 1:
                    grid[y][0] = "V"

        if boundary_right == "double":
            for y in range(1, height - 1):
                grid[y][width - 1] = "‖"
        elif boundary_right == "caret":
            mid = height // 2
            for y in (mid - 1, mid, mid + 1):
                if 0 < y < height - 1:
                    grid[y][width - 1] = "V"
        _place_edge_arrow(Side.LEFT, height - 2, 0, "|")
        _place_edge_arrow(Side.RIGHT, height - 2, width - 1, "|")

        inner_w = width - 2
        inner_h = height - 2

        def _positions_along(length: int, count: int, pad: int) -> List[int]:
            if count <= 0:
                return []
            inner_len = max(1, length - 2 * pad)
            if count == 1:
                return [1 + pad + (inner_len - 1) // 2]
            return [
                int(round(1 + pad + i * (inner_len - 1) / (count - 1)))
                for i in range(count)
            ]

        # TOP: right -> left
        top_ports = list(square.top.ports())
        xs = _positions_along(inner_w, len(top_ports), padding)
        xs = list(reversed(xs))
        for p, x in zip(top_ports, xs):
            grid[0][x] = port_symbols.get(p, "o")[:1]

        # LEFT: top -> bottom
        left_ports = list(square.left.ports())
        ys = _positions_along(inner_h, len(left_ports), padding)
        for p, y in zip(left_ports, ys):
            if Side.LEFT in interior_sides:
                continue
            grid[y][0] = port_symbols.get(p, "o")[:1]

        # BOTTOM: left -> right
        bottom_ports = list(square.bottom.ports())
        xs = _positions_along(inner_w, len(bottom_ports), padding)
        for p, x in zip(bottom_ports, xs):
            grid[height - 1][x] = port_symbols.get(p, "o")[:1]

        # RIGHT: typically bottom -> top, but use top -> bottom to align shared edges
        right_ports = list(square.right.ports())
        ys = _positions_along(inner_h, len(right_ports), padding)
        if not right_top_to_bottom:
            ys = list(reversed(ys))
        for p, y in zip(right_ports, ys):
            if Side.RIGHT in interior_sides:
                continue
            grid[y][width - 1] = port_symbols.get(p, "o")[:1]

        def _place_marker_on_edge(row: int, marker: str | None) -> None:
            if not marker:
                return
            candidates = [
                x
                for x in range(1, width - 1)
                if grid[row][x] == "-" and x != width - 2
            ]
            if not candidates:
                return
            best = min(candidates, key=lambda x: abs(x - mid_x))
            grid[row][best] = marker

        _place_marker_on_edge(0, marker_top)
        _place_marker_on_edge(height - 1, marker_bottom)

        return ["".join(row) for row in grid]

    def _render_ascii(self, squares: List, *, wrap: bool, width: int = 25, height: int = 11, padding: int = 3) -> str:
        labels = self._pair_labels()
        top_labels = [self._edge_label(self.edge(Side.TOP, i), labels) for i in range(self.N)]
        bottom_labels = [self._edge_label(self.edge(Side.BOTTOM, i), labels) for i in range(self.N)]

        port_symbols = self._port_symbol_map()

        def _blank_if_none(label: str) -> str:
            return "" if label == "--" else label

        def _merge_segments(segments: List[str]) -> str:
            # Overlap boundary columns to remove spacing between squares.
            if not segments:
                return ""
            out = segments[0]
            for seg in segments[1:]:
                out += seg[1:]
            return out

        label_segments_top = [_blank_if_none(lab).center(width) for lab in top_labels]
        label_segments_bottom = [_blank_if_none(lab).center(width) for lab in bottom_labels]
        label_line_top = _merge_segments(label_segments_top)
        label_line_bottom = _merge_segments(label_segments_bottom)

        square_lines = []
        for idx, sq in enumerate(squares):
            right_top_to_bottom = True  # align shared interior edges in strip/annulus view
            interior_sides: set[Side] = set()
            if self.is_interior_edge(EdgeRef(Side.LEFT, idx)):
                interior_sides.add(Side.LEFT)
            if self.is_interior_edge(EdgeRef(Side.RIGHT, idx)):
                interior_sides.add(Side.RIGHT)
            marker_top = self._edge_marker(BoundaryEdge(Side.TOP, idx))
            marker_bottom = self._edge_marker(BoundaryEdge(Side.BOTTOM, idx))
            boundary_left = "none"
            boundary_right = "none"
            if not wrap and idx == 0:
                boundary_left = "double"
            if not wrap and idx == self.N - 1:
                boundary_right = "double"
            if wrap and idx == 0:
                boundary_left = "caret"
            if wrap and idx == self.N - 1:
                boundary_right = "caret"
            square_lines.append(
                self._render_square_ascii(
                    sq,
                    width=width,
                    height=height,
                    padding=padding,
                    port_symbols=port_symbols,
                    interior_sides=interior_sides,
                    boundary_left=boundary_left,
                    boundary_right=boundary_right,
                    marker_top=marker_top,
                    marker_bottom=marker_bottom,
                    right_top_to_bottom=right_top_to_bottom,
                    edge_dir_lookup=getattr(self, "edge_direction", None),
                    square_index=idx,
                )
            )

        combined: List[str] = []
        combined.append(f"        {label_line_top}")
        for row in range(height):
            line = square_lines[0][row]
            for i in range(1, self.N):
                line += square_lines[i][row][1:]
            combined.append("        " + line)
        combined.append(f"        {label_line_bottom}")
        return "\n".join(combined)


@dataclass
class MarkedStrip(_MarkedBase):
    strip: SquareStrip

    @property
    def N(self) -> int:
        return len(self.strip)

    def edge(self, side: Side, i: int) -> BoundaryEdge:
        return BoundaryEdge(side=side, i=i)

    def square(self, i: int):
        return self.strip.square(i)

    def edge_ref(self, side: Side, i: int) -> EdgeRef:
        return EdgeRef(side=side, i=i)

    def _edge_obj(self, e: EdgeRef):
        return self.square(e.i).edge(e.side)

    def _interior_pair(self, e: EdgeRef) -> EdgeRef | None:
        if e.side == Side.RIGHT:
            if e.i >= self.N - 1:
                return None
            return EdgeRef(Side.LEFT, e.i + 1)
        if e.side == Side.LEFT:
            if e.i <= 0:
                return None
            return EdgeRef(Side.RIGHT, e.i - 1)
        return None

    def is_marked_edge(self, e: EdgeRef) -> bool:
        return e.side in (Side.TOP, Side.BOTTOM) and self.is_marked(BoundaryEdge(e.side, e.i))

    def is_interior_edge(self, e: EdgeRef) -> bool:
        return e.side in (Side.LEFT, Side.RIGHT) and self._interior_pair(e) is not None

    def is_boundary_edge(self, e: EdgeRef) -> bool:
        return not self.is_interior_edge(e) and not self.is_marked_edge(e)

    def _pair_reversed_marked(self, a: BoundaryEdge, b: BoundaryEdge, is_rev: bool) -> bool:
        top_bottom = (a.side != b.side)
        return top_bottom ^ bool(is_rev)

    def _insert_index(self, ports: List[Port], left: Port | None, right: Port | None) -> int | None:
        if left is None and right is None:
            return len(ports)
        if left is not None and left not in ports:
            return None
        if right is not None and right not in ports:
            return None
        if left is not None and right is not None:
            li = ports.index(left)
            ri = ports.index(right)
            if ri != li + 1:
                return None
            return ri
        if left is not None:
            return ports.index(left) + 1
        return ports.index(right)

    def _relabel_edge(self, e: EdgeRef) -> None:
        ports = list(self._edge_obj(e).ports())
        for idx, p in enumerate(ports, start=1):
            object.__setattr__(p, "label", f"{e.side.short()}{idx}")

    def _port_symbol_map(self) -> Dict[Port, str]:
        """
        Assign a unique symbol to each paired port group.
        """
        # Symbols encode identifications across pairs for visualization.
        palette = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
        sym_idx = 0
        out: Dict[Port, str] = {}

        def _next_symbol() -> str:
            nonlocal sym_idx
            if sym_idx < len(palette):
                s = palette[sym_idx]
            else:
                s = "?"
            sym_idx += 1
            return s

        # Interior pairs (RIGHT i) <-> (LEFT i+1)
        for i in range(self.N - 1):
            ea = EdgeRef(Side.RIGHT, i)
            eb = EdgeRef(Side.LEFT, i + 1)
            ports_a = list(self._edge_obj(ea).ports())
            ports_b = list(self._edge_obj(eb).ports())
            if len(ports_a) != len(ports_b):
                continue
            if self._edge_obj(ea) is self._edge_obj(eb):
                for pa in ports_a:
                    if pa in out:
                        continue
                    sym = _next_symbol()
                    out[pa] = sym
                continue
            for idx, pa in enumerate(ports_a):
                pb = ports_b[len(ports_b) - 1 - idx]
                if pa in out or pb in out:
                    continue
                sym = _next_symbol()
                out[pa] = sym
                out[pb] = sym

        # Marked pairs
        for p in self.all_pairs():
            ea = EdgeRef(p.a.side, p.a.i)
            eb = EdgeRef(p.b.side, p.b.i)
            ports_a = list(self._edge_obj(ea).ports())
            ports_b = list(self._edge_obj(eb).ports())
            if len(ports_a) != len(ports_b):
                continue
            reversed_order = self._pair_reversed_marked(p.a, p.b, p.orientation_reversing)
            for idx, pa in enumerate(ports_a):
                j = (len(ports_b) - 1 - idx) if reversed_order else idx
                pb = ports_b[j]
                if pa in out or pb in out:
                    continue
                sym = _next_symbol()
                out[pa] = sym
                out[pb] = sym

        return out

    def all_edge_refs(self) -> List[EdgeRef]:
        out: List[EdgeRef] = []
        for i in range(self.N):
            for side in (Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT):
                out.append(EdgeRef(side, i))
        return out

    def paired_port(self, e: EdgeRef, port: Port) -> Port | None:
        """
        Return the port paired with (e, port) under the identification rules.
        """
        self._validate_edge_ref(e)
        edge_a = self._edge_obj(e)
        ports_a = list(edge_a.ports())
        if port not in ports_a:
            return None
        idx = ports_a.index(port)

        if self.is_interior_edge(e):
            e2 = self._interior_pair(e)
            if e2 is None:
                return None
            edge_b = self._edge_obj(e2)
            ports_b = list(edge_b.ports())
            if len(ports_b) != len(ports_a):
                return None
            if edge_b is edge_a:
                return ports_a[idx]
            return ports_b[len(ports_b) - 1 - idx]

        if self.is_marked_edge(e):
            be = BoundaryEdge(e.side, e.i)
            other, is_rev = self.pair_info(be)
            e2 = EdgeRef(other.side, other.i)
            edge_b = self._edge_obj(e2)
            ports_b = list(edge_b.ports())
            if len(ports_b) != len(ports_a):
                return None
            reversed_order = self._pair_reversed_marked(be, other, is_rev)
            j = (len(ports_b) - 1 - idx) if reversed_order else idx
            return ports_b[j]

        return None

    def paired_boundary_point(self, e: EdgeRef, port: Port) -> Tuple[EdgeRef, Port] | None:
        """
        Return the paired (edge, port) for a boundary point, or None if unpaired.
        """
        self._validate_edge_ref(e)
        if self.is_interior_edge(e):
            e2 = self._interior_pair(e)
            if e2 is None:
                return None
            p2 = self.paired_port(e, port)
            if p2 is None:
                return None
            return (e2, p2)

        if self.is_marked_edge(e):
            be = BoundaryEdge(e.side, e.i)
            other, _is_rev = self.pair_info(be)
            e2 = EdgeRef(other.side, other.i)
            p2 = self.paired_port(e, port)
            if p2 is None:
                return None
            return (e2, p2)

        return None

    def validate_ports(self) -> bool:
        """
        Validate port constraints. Returns True if all constraints hold.
        """
        ok = True
        for e in self.all_edge_refs():
            edge = self._edge_obj(e)
            count = len(edge.ports())
            if self.is_boundary_edge(e):
                ok = ok and (count == 0)
            else:
                ok = ok and (count >= 1)

        # Marked pairs: equal counts
        for p in self.all_pairs():
            ea = EdgeRef(p.a.side, p.a.i)
            eb = EdgeRef(p.b.side, p.b.i)
            if len(self._edge_obj(ea).ports()) != len(self._edge_obj(eb).ports()):
                ok = False

        # Interior pairs: equal counts
        for i in range(self.N):
            e = EdgeRef(Side.RIGHT, i)
            e2 = self._interior_pair(e)
            if e2 is None:
                continue
            if len(self._edge_obj(e).ports()) != len(self._edge_obj(e2).ports()):
                ok = False

        return ok
    def add_port_between(
        self,
        e: EdgeRef,
        *,
        left: Port | None = None,
        right: Port | None = None,
    ) -> Tuple[Port, Port] | None:
        """
        Add a new port on edge e (between left/right), updating its paired edge.

        Returns (new_port_on_e, new_port_on_pair) on success, or None on failure.
        """
        self._validate_edge_ref(e)
        if self.is_boundary_edge(e):
            return None

        edge_a = self._edge_obj(e)
        ports_a = list(edge_a.ports())
        idx = self._insert_index(ports_a, left, right)
        if idx is None:
            return None

        new_a = edge_a.add_port(label="new", index=idx)

        if self.is_interior_edge(e):
            e2 = self._interior_pair(e)
            if e2 is None:
                return None
            edge_b = self._edge_obj(e2)
            if edge_b is edge_a:
                new_b = new_a
            else:
                ports_b = list(edge_b.ports())
                if len(ports_b) != len(ports_a):
                    edge_a.remove_port(new_a)
                    return None
                idx_b = len(ports_b) - idx
                new_b = edge_b.add_port(label="new", index=idx_b)
            self._relabel_edge(e)
            self._relabel_edge(e2)
            return new_a, new_b

        # Marked edge
        be = BoundaryEdge(e.side, e.i)
        other, is_rev = self.pair_info(be)
        e2 = EdgeRef(other.side, other.i)
        edge_b = self._edge_obj(e2)
        ports_b = list(edge_b.ports())
        if len(ports_b) != len(ports_a):
            edge_a.remove_port(new_a)
            return None

        reversed_order = self._pair_reversed_marked(be, other, is_rev)
        idx_b = (len(ports_b) - idx) if reversed_order else idx
        new_b = edge_b.add_port(label="new", index=idx_b)
        self._relabel_edge(e)
        self._relabel_edge(e2)
        return new_a, new_b

    def __str__(self) -> str:
        header = f"MarkedStrip(N={self.N})"
        return header + "\n" + self._render_ascii(self.strip.squares, wrap=False)


@dataclass
class MarkedAnnulus(_MarkedBase):
    annulus: Annulus

    @property
    def N(self) -> int:
        return len(self.annulus)

    def edge(self, side: Side, i: int) -> BoundaryEdge:
        return BoundaryEdge(side=side, i=i)

    def square(self, i: int):
        return self.annulus.square(i)

    def edge_ref(self, side: Side, i: int) -> EdgeRef:
        return EdgeRef(side=side, i=i)

    def _edge_obj(self, e: EdgeRef):
        return self.square(e.i).edge(e.side)

    def _interior_pair(self, e: EdgeRef) -> EdgeRef | None:
        if e.side == Side.RIGHT:
            return EdgeRef(Side.LEFT, (e.i + 1) % self.N)
        if e.side == Side.LEFT:
            return EdgeRef(Side.RIGHT, (e.i - 1) % self.N)
        return None

    def is_marked_edge(self, e: EdgeRef) -> bool:
        return e.side in (Side.TOP, Side.BOTTOM) and self.is_marked(BoundaryEdge(e.side, e.i))

    def is_interior_edge(self, e: EdgeRef) -> bool:
        return e.side in (Side.LEFT, Side.RIGHT)

    def is_boundary_edge(self, e: EdgeRef) -> bool:
        return not self.is_interior_edge(e) and not self.is_marked_edge(e)

    def _pair_reversed_marked(self, a: BoundaryEdge, b: BoundaryEdge, is_rev: bool) -> bool:
        top_bottom = (a.side != b.side)
        return top_bottom ^ bool(is_rev)

    def _insert_index(self, ports: List[Port], left: Port | None, right: Port | None) -> int | None:
        if left is None and right is None:
            return len(ports)
        if left is not None and left not in ports:
            return None
        if right is not None and right not in ports:
            return None
        if left is not None and right is not None:
            li = ports.index(left)
            ri = ports.index(right)
            if ri != li + 1:
                return None
            return ri
        if left is not None:
            return ports.index(left) + 1
        return ports.index(right)

    def _relabel_edge(self, e: EdgeRef) -> None:
        ports = list(self._edge_obj(e).ports())
        for idx, p in enumerate(ports, start=1):
            object.__setattr__(p, "label", f"{e.side.short()}{idx}")

    def _port_symbol_map(self) -> Dict[Port, str]:
        """
        Assign a unique symbol to each paired port group.
        """
        palette = list(string.digits + string.ascii_uppercase + string.ascii_lowercase)
        sym_idx = 0
        out: Dict[Port, str] = {}

        def _next_symbol() -> str:
            nonlocal sym_idx
            if sym_idx < len(palette):
                s = palette[sym_idx]
            else:
                s = "?"
            sym_idx += 1
            return s

        # Interior pairs (RIGHT i) <-> (LEFT i+1 mod N)
        for i in range(self.N):
            ea = EdgeRef(Side.RIGHT, i)
            eb = EdgeRef(Side.LEFT, (i + 1) % self.N)
            ports_a = list(self._edge_obj(ea).ports())
            ports_b = list(self._edge_obj(eb).ports())
            if len(ports_a) != len(ports_b):
                continue
            if self._edge_obj(ea) is self._edge_obj(eb):
                for pa in ports_a:
                    if pa in out:
                        continue
                    sym = _next_symbol()
                    out[pa] = sym
                continue
            for idx, pa in enumerate(ports_a):
                pb = ports_b[len(ports_b) - 1 - idx]
                if pa in out or pb in out:
                    continue
                sym = _next_symbol()
                out[pa] = sym
                out[pb] = sym

        # Marked pairs
        for p in self.all_pairs():
            ea = EdgeRef(p.a.side, p.a.i)
            eb = EdgeRef(p.b.side, p.b.i)
            ports_a = list(self._edge_obj(ea).ports())
            ports_b = list(self._edge_obj(eb).ports())
            if len(ports_a) != len(ports_b):
                continue
            reversed_order = self._pair_reversed_marked(p.a, p.b, p.orientation_reversing)
            for idx, pa in enumerate(ports_a):
                j = (len(ports_b) - 1 - idx) if reversed_order else idx
                pb = ports_b[j]
                if pa in out or pb in out:
                    continue
                sym = _next_symbol()
                out[pa] = sym
                out[pb] = sym

        return out

    def all_edge_refs(self) -> List[EdgeRef]:
        out: List[EdgeRef] = []
        for i in range(self.N):
            for side in (Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT):
                out.append(EdgeRef(side, i))
        return out

    def paired_port(self, e: EdgeRef, port: Port) -> Port | None:
        """
        Return the port paired with (e, port) under the identification rules.
        """
        self._validate_edge_ref(e)
        edge_a = self._edge_obj(e)
        ports_a = list(edge_a.ports())
        if port not in ports_a:
            return None
        idx = ports_a.index(port)

        if self.is_interior_edge(e):
            e2 = self._interior_pair(e)
            if e2 is None:
                return None
            edge_b = self._edge_obj(e2)
            ports_b = list(edge_b.ports())
            if len(ports_b) != len(ports_a):
                return None
            if edge_b is edge_a:
                return ports_a[idx]
            return ports_b[len(ports_b) - 1 - idx]

        if self.is_marked_edge(e):
            be = BoundaryEdge(e.side, e.i)
            other, is_rev = self.pair_info(be)
            e2 = EdgeRef(other.side, other.i)
            edge_b = self._edge_obj(e2)
            ports_b = list(edge_b.ports())
            if len(ports_b) != len(ports_a):
                return None
            reversed_order = self._pair_reversed_marked(be, other, is_rev)
            j = (len(ports_b) - 1 - idx) if reversed_order else idx
            return ports_b[j]

        return None

    def paired_boundary_point(self, e: EdgeRef, port: Port) -> Tuple[EdgeRef, Port] | None:
        """
        Return the paired (edge, port) for a boundary point, or None if unpaired.
        """
        self._validate_edge_ref(e)
        if self.is_interior_edge(e):
            e2 = self._interior_pair(e)
            p2 = self.paired_port(e, port)
            if p2 is None:
                return None
            return (e2, p2)

        if self.is_marked_edge(e):
            be = BoundaryEdge(e.side, e.i)
            other, _is_rev = self.pair_info(be)
            e2 = EdgeRef(other.side, other.i)
            p2 = self.paired_port(e, port)
            if p2 is None:
                return None
            return (e2, p2)

        return None

    def validate_ports(self) -> bool:
        """
        Validate port constraints. Returns True if all constraints hold.
        """
        ok = True
        for e in self.all_edge_refs():
            edge = self._edge_obj(e)
            count = len(edge.ports())
            if self.is_boundary_edge(e):
                ok = ok and (count == 0)
            else:
                ok = ok and (count >= 1)

        for p in self.all_pairs():
            ea = EdgeRef(p.a.side, p.a.i)
            eb = EdgeRef(p.b.side, p.b.i)
            if len(self._edge_obj(ea).ports()) != len(self._edge_obj(eb).ports()):
                ok = False

        for i in range(self.N):
            e = EdgeRef(Side.RIGHT, i)
            e2 = self._interior_pair(e)
            if len(self._edge_obj(e).ports()) != len(self._edge_obj(e2).ports()):
                ok = False

        return ok
    def add_port_between(
        self,
        e: EdgeRef,
        *,
        left: Port | None = None,
        right: Port | None = None,
    ) -> Tuple[Port, Port] | None:
        """
        Add a new port on edge e (between left/right), updating its paired edge.

        Returns (new_port_on_e, new_port_on_pair) on success, or None on failure.
        """
        self._validate_edge_ref(e)
        if self.is_boundary_edge(e):
            return None

        edge_a = self._edge_obj(e)
        ports_a = list(edge_a.ports())
        idx = self._insert_index(ports_a, left, right)
        if idx is None:
            return None

        new_a = edge_a.add_port(label="new", index=idx)

        if self.is_interior_edge(e):
            e2 = self._interior_pair(e)
            if e2 is None:
                return None
            edge_b = self._edge_obj(e2)
            if edge_b is edge_a:
                new_b = new_a
            else:
                ports_b = list(edge_b.ports())
                if len(ports_b) != len(ports_a):
                    edge_a.remove_port(new_a)
                    return None
                idx_b = len(ports_b) - idx
                new_b = edge_b.add_port(label="new", index=idx_b)
            self._relabel_edge(e)
            self._relabel_edge(e2)
            return new_a, new_b

        # Marked edge
        be = BoundaryEdge(e.side, e.i)
        other, is_rev = self.pair_info(be)
        e2 = EdgeRef(other.side, other.i)
        edge_b = self._edge_obj(e2)
        ports_b = list(edge_b.ports())
        if len(ports_b) != len(ports_a):
            edge_a.remove_port(new_a)
            return None

        reversed_order = self._pair_reversed_marked(be, other, is_rev)
        idx_b = (len(ports_b) - idx) if reversed_order else idx
        new_b = edge_b.add_port(label="new", index=idx_b)
        self._relabel_edge(e)
        self._relabel_edge(e2)
        return new_a, new_b

    def __str__(self) -> str:
        header = f"MarkedAnnulus(N={self.N})"
        return header + "\n" + self._render_ascii(self.annulus.squares, wrap=True)

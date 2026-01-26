"""
pattern.py

Pattern: a choice of chord diagram on each square of a marked strip/annulus,
with compatibility across identified ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple, Union

from chord_diagram import ChordDiagram
from chords import BoundaryPoint
from marked_strips import EdgeRef, MarkedAnnulus, MarkedStrip
from square import Side


MarkedSurface = Union[MarkedStrip, MarkedAnnulus]

# Debug render flag: show labels for interior-edge ports inside the square.
DEBUG_INTERIOR_LABELS = False


def _used_points(diagram: ChordDiagram) -> Set[BoundaryPoint]:
    used: Set[BoundaryPoint] = set()
    for ch in diagram.chords:
        used.add(ch.a)
        used.add(ch.b)
    return used


@dataclass
class Pattern:
    """
    A pattern assigns a chord diagram to each square of a marked surface.

    Compatibility rule: if a boundary point is used in one square, then its
    paired boundary point must be used in the diagram on the paired square.
    """

    surface: MarkedSurface
    diagrams: Dict[int, ChordDiagram]

    @classmethod
    def from_diagrams(cls, surface: MarkedSurface, diagrams: Iterable[ChordDiagram]) -> "Pattern":
        mapping: Dict[int, ChordDiagram] = {}
        for i, d in enumerate(diagrams):
            mapping[i] = d
        return cls(surface=surface, diagrams=mapping)

    def diagram(self, i: int) -> ChordDiagram:
        return self.diagrams[i]

    def validate(self) -> bool:
        """
        Return True if all compatibility constraints are satisfied.
        """
        # Require a diagram for each square.
        for i in range(self.surface.N):
            if i not in self.diagrams:
                return False

        # Precompute used points per square.
        used_by_square: Dict[int, Set[BoundaryPoint]] = {}
        for i in range(self.surface.N):
            used_by_square[i] = _used_points(self.diagrams[i])

        # Build a lookup from (square, side, port) to used.
        used_lookup: Set[Tuple[int, Side, object]] = set()
        for i, pts in used_by_square.items():
            for bp in pts:
                used_lookup.add((i, bp.side, bp.port))

        # Check every used boundary point has its paired point used.
        for i, pts in used_by_square.items():
            for bp in pts:
                e = EdgeRef(bp.side, i)
                paired = self.surface.paired_boundary_point(e, bp.port)
                if paired is None:
                    return False
                e2, p2 = paired
                if (e2.i, e2.side, p2) not in used_lookup:
                    return False

        return True

    def render(self, width: int = 25, height: int = 11, padding: int = 3) -> str:
        """
        Render the pattern as ASCII, matching the marked strip/annulus style.
        """
        if width < 5 or height < 5:
            raise ValueError("width and height must be at least 5")
        if padding < 0:
            raise ValueError("padding must be nonnegative")

        labels = self.surface._pair_labels()  # type: ignore[attr-defined]
        top_labels = [self.surface._edge_label(self.surface.edge(Side.TOP, i), labels) for i in range(self.surface.N)]  # type: ignore[attr-defined]
        bottom_labels = [self.surface._edge_label(self.surface.edge(Side.BOTTOM, i), labels) for i in range(self.surface.N)]  # type: ignore[attr-defined]

        # Directed wrappers expose annulus/strip via attribute forwarding.
        is_annulus = hasattr(self.surface, "annulus") and not hasattr(self.surface, "strip")
        port_symbols = self._pair_symbol_map()

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

        def _line_points(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
            points: List[Tuple[int, int]] = []
            dx = abs(x1 - x0)
            dy = -abs(y1 - y0)
            sx = 1 if x0 < x1 else -1
            sy = 1 if y0 < y1 else -1
            err = dx + dy
            x, y = x0, y0
            while True:
                points.append((x, y))
                if x == x1 and y == y1:
                    break
                e2 = 2 * err
                if e2 >= dy:
                    err += dy
                    x += sx
                if e2 <= dx:
                    err += dx
                    y += sy
            return points

        def _render_square_with_chords(
            diagram: ChordDiagram,
            sq_index: int,
            *,
            boundary_left: str,
            boundary_right: str,
        ) -> List[str]:
            # Local square render with chord lines projected inward.
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
                if not hasattr(self.surface, "edge_direction"):
                    return None
                d = self.surface.edge_direction(EdgeRef(side, sq_index))  # type: ignore[attr-defined]
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

            marker_top = self.surface._edge_marker(self.surface.edge(Side.TOP, sq_index))  # type: ignore[attr-defined]
            marker_bottom = self.surface._edge_marker(self.surface.edge(Side.BOTTOM, sq_index))  # type: ignore[attr-defined]
            mid_x = width // 2
            if marker_top and grid[0][mid_x] == "-":
                grid[0][mid_x] = marker_top
            if marker_bottom and grid[height - 1][mid_x] == "-":
                grid[height - 1][mid_x] = marker_bottom
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

            port_positions: Dict[Tuple[Side, object], Tuple[int, int]] = {}

            # TOP: right -> left
            top_ports = list(diagram.square.top.ports())
            xs = _positions_along(inner_w, len(top_ports), padding)
            xs = list(reversed(xs))
            for p, x in zip(top_ports, xs):
                port_positions[(Side.TOP, p)] = (x, 0)

            # LEFT: top -> bottom
            left_ports = list(diagram.square.left.ports())
            ys = _positions_along(inner_h, len(left_ports), padding)
            for p, y in zip(left_ports, ys):
                port_positions[(Side.LEFT, p)] = (0, y)

            # BOTTOM: left -> right
            bottom_ports = list(diagram.square.bottom.ports())
            xs = _positions_along(inner_w, len(bottom_ports), padding)
            for p, x in zip(bottom_ports, xs):
                port_positions[(Side.BOTTOM, p)] = (x, height - 1)

            # RIGHT: bottom -> top (preserve edge orientation; aligns interior pair reversal)
            right_ports = list(diagram.square.right.ports())
            ys = _positions_along(inner_h, len(right_ports), padding)
            ys = list(reversed(ys))
            for p, y in zip(right_ports, ys):
                port_positions[(Side.RIGHT, p)] = (width - 1, y)

            for (side, port), (x, y) in port_positions.items():
                key = (sq_index, side, port)
                if self.surface.is_interior_edge(EdgeRef(side, sq_index)):
                    continue
                grid[y][x] = port_symbols.get(key, "o")[:1]

            def _inward_endpoint(side: Side, x: int, y: int) -> Tuple[int, int]:
                if side == Side.TOP:
                    return (x, 1)
                if side == Side.BOTTOM:
                    return (x, height - 2)
                if side == Side.LEFT:
                    return (1, y)
                if side == Side.RIGHT:
                    return (width - 2, y)
                raise ValueError("Unknown side")

            for ch in diagram.chords:
                ax, ay = port_positions[(ch.a.side, ch.a.port)]
                bx, by = port_positions[(ch.b.side, ch.b.port)]
                ax, ay = _inward_endpoint(ch.a.side, ax, ay)
                bx, by = _inward_endpoint(ch.b.side, bx, by)
                for x, y in _line_points(ax, ay, bx, by):
                    if 0 < x < width - 1 and 0 < y < height - 1:
                        if grid[y][x] == " ":
                            grid[y][x] = "*"

            if DEBUG_INTERIOR_LABELS:
                for (side, port), (x, y) in port_positions.items():
                    if not self.surface.is_interior_edge(EdgeRef(side, sq_index)):
                        continue
                    label = getattr(port, "label", "")
                    ch = label[-1] if label else "o"
                    if side == Side.LEFT:
                        grid[y][1] = ch
                    elif side == Side.RIGHT:
                        grid[y][width - 2] = ch

            return ["".join(row) for row in grid]

        diagrams_in_order = [self.diagrams[i] for i in range(self.surface.N)]
        square_lines = []
        for i, d in enumerate(diagrams_in_order):
            boundary_left = "none"
            boundary_right = "none"
            if is_annulus:
                if i == 0:
                    boundary_left = "caret"
                if i == self.surface.N - 1:
                    boundary_right = "caret"
            else:
                if i == 0:
                    boundary_left = "double"
                if i == self.surface.N - 1:
                    boundary_right = "double"
            square_lines.append(
                _render_square_with_chords(
                    d,
                    i,
                    boundary_left=boundary_left,
                    boundary_right=boundary_right,
                )
            )

        combined: List[str] = []
        combined.append(f"        {label_line_top}")
        for row in range(height):
            line = square_lines[0][row]
            for i in range(1, self.surface.N):
                line += square_lines[i][row][1:]
            combined.append("        " + line)
        combined.append(f"        {label_line_bottom}")
        return "\n".join(combined)

    def __str__(self) -> str:
        ok = self.validate()
        return f"Pattern(valid={ok}, squares={self.surface.N})"

    def _pair_symbol_map(self) -> Dict[Tuple[int, Side, object], str]:
        """
        Assign a unique symbol to each paired boundary point, keyed by (square, side, port).
        """
        palette = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
        sym_idx = 0
        out: Dict[Tuple[int, Side, object], str] = {}

        def _next_symbol() -> str:
            nonlocal sym_idx
            if sym_idx < len(palette):
                s = palette[sym_idx]
            else:
                s = "?"
            sym_idx += 1
            return s

        for i in range(self.surface.N):
            sq = self.diagrams[i].square
            for side, edge in sq.edges():
                for port in edge.ports():
                    key = (i, side, port)
                    if key in out:
                        continue
                    paired = self.surface.paired_boundary_point(EdgeRef(side, i), port)
                    if paired is None:
                        continue
                    e2, p2 = paired
                    key2 = (e2.i, e2.side, p2)
                    sym = _next_symbol()
                    out[key] = sym
                    out[key2] = sym

        return out

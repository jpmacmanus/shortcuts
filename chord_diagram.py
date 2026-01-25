"""
chord_diagram.py

ChordDiagram: a square equipped with a set of noncrossing chords.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Set, Tuple

from chords import BoundaryPoint, Chord, chords_cross_in_square
from edge import Port
from square import Side, Square


@dataclass
class ChordDiagram:
    """
    A square together with a collection of noncrossing chords.

    Constraints:
    - No two chords share a port.
    - No two chords cross.
    """

    square: Square
    chords: Set[Chord] = field(default_factory=set)

    def _used_ports(self) -> Set[BoundaryPoint]:
        used: Set[BoundaryPoint] = set()
        for ch in self.chords:
            used.add(ch.a)
            used.add(ch.b)
        return used

    def can_add_chord(self, chord: Chord) -> bool:
        """Return True if adding chord would preserve diagram constraints."""
        used = self._used_ports()
        used_ports = {bp.port for bp in used}
        if chord.a.port in used_ports or chord.b.port in used_ports:
            return False
        if chord.a in used or chord.b in used:
            return False
        for other in self.chords:
            if chords_cross_in_square(self.square, chord, other):
                return False
        return True

    def add_chord(self, chord: Chord) -> Chord | None:
        """
        Try to add a chord.

        Returns the chord on success, or None if it violates constraints.
        """
        if not self.can_add_chord(chord):
            return None
        self.chords.add(chord)
        return chord

    def remove_chord(self, chord: Chord) -> None:
        """Remove a chord if present."""
        self.chords.discard(chord)

    def reduce(self) -> None:
        """
        Remove any ports on the square's edges that are not used by chords.

        Port order among the remaining ports is preserved.
        Ports are relabeled consecutively per edge (no gaps).
        """
        used = self._used_ports()
        used_ports = {bp.port for bp in used}

        port_map: Dict[Port, Port] = {}

        for side, edge in self.square.edges():
            kept: List[Port] = [p for p in edge.ports() if p in used_ports]
            relabeled: List[Port] = []
            for idx, old in enumerate(kept, start=1):
                new_port = Port(label=f"{side.short()}{idx}")
                relabeled.append(new_port)
                port_map[old] = new_port
            edge._set_ports(relabeled)

        new_chords: Set[Chord] = set()
        for ch in self.chords:
            a = BoundaryPoint(ch.a.side, port_map[ch.a.port])
            b = BoundaryPoint(ch.b.side, port_map[ch.b.port])
            new_chords.add(Chord(a, b))
        self.chords = new_chords

    def _relabel_all_ports(self) -> None:
        """
        Relabel all ports consecutively per edge and rewrite chords accordingly.
        """
        port_map: Dict[Port, Port] = {}
        for side, edge in self.square.edges():
            relabeled: List[Port] = []
            for idx, old in enumerate(edge.ports(), start=1):
                new_port = Port(label=f"{side.short()}{idx}")
                relabeled.append(new_port)
                port_map[old] = new_port
            edge._set_ports(relabeled)

        new_chords: Set[Chord] = set()
        for ch in self.chords:
            a = BoundaryPoint(ch.a.side, port_map[ch.a.port])
            b = BoundaryPoint(ch.b.side, port_map[ch.b.port])
            new_chords.add(Chord(a, b))
        self.chords = new_chords

    def add_chord_between(
        self,
        *,
        from_point: BoundaryPoint,
        to_side: Side,
    ) -> Chord | None:
        """
        Try to add a chord from an unused port to a new port on the given side.

        The method inserts a new port on to_side (in any position that works),
        then adds the chord if it remains port-disjoint and noncrossing.
        Returns the new Chord on success, or None on failure.
        """
        if from_point.side == to_side:
            return None

        # Ensure from_point is on the square and unused.
        edge_from = self.square.edge(from_point.side)
        if from_point.port not in edge_from.ports():
            return None
        if from_point in self._used_ports():
            return None

        edge_to = self.square.edge(to_side)

        # Try each possible insertion position for the new port.
        for idx in range(len(edge_to.ports()) + 1):
            new_port = edge_to.add_port(label="new", index=idx)
            new_point = BoundaryPoint(to_side, new_port)
            try:
                chord = Chord(from_point, new_point)
                if self.can_add_chord(chord):
                    self.chords.add(chord)
                    self._relabel_all_ports()
                    return chord
            except Exception:
                pass
            edge_to.remove_port(new_port)

        return None

    def render(self, width: int = 21, height: int = 9, padding: int = 3) -> str:
        """
        Render the diagram as ASCII.

        width/height include the border; both must be >= 5 for readability.
        """
        if width < 5 or height < 5:
            raise ValueError("width and height must be at least 5")
        if padding < 0:
            raise ValueError("padding must be nonnegative")

        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Port positions follow edge orientation to keep crossings consistent.
        # Draw border.
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

        port_positions: Dict[Tuple[Side, object], Tuple[int, int]] = {}

        # TOP: right -> left
        top_ports = list(self.square.top.ports())
        xs = _positions_along(inner_w, len(top_ports), padding)
        xs = list(reversed(xs))
        for p, x in zip(top_ports, xs):
            port_positions[(Side.TOP, p)] = (x, 0)

        # LEFT: top -> bottom
        left_ports = list(self.square.left.ports())
        ys = _positions_along(inner_h, len(left_ports), padding)
        for p, y in zip(left_ports, ys):
            port_positions[(Side.LEFT, p)] = (0, y)

        # BOTTOM: left -> right
        bottom_ports = list(self.square.bottom.ports())
        xs = _positions_along(inner_w, len(bottom_ports), padding)
        for p, x in zip(bottom_ports, xs):
            port_positions[(Side.BOTTOM, p)] = (x, height - 1)

        # RIGHT: bottom -> top
        right_ports = list(self.square.right.ports())
        ys = _positions_along(inner_h, len(right_ports), padding)
        ys = list(reversed(ys))
        for p, y in zip(right_ports, ys):
            port_positions[(Side.RIGHT, p)] = (width - 1, y)

        # Draw ports.
        for (side, port), (x, y) in port_positions.items():
            _ = side  # side kept for clarity; not used here.
            _ = port
            grid[y][x] = "o"

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

        # Draw chords.
        for ch in self.chords:
            ax, ay = port_positions[(ch.a.side, ch.a.port)]
            bx, by = port_positions[(ch.b.side, ch.b.port)]
            ax, ay = _inward_endpoint(ch.a.side, ax, ay)
            bx, by = _inward_endpoint(ch.b.side, bx, by)
            for x, y in _line_points(ax, ay, bx, by):
                if 0 < x < width - 1 and 0 < y < height - 1:
                    if grid[y][x] == " ":
                        grid[y][x] = "*"

        return "\n".join("".join(row) for row in grid)

    def __str__(self) -> str:
        return self.render()

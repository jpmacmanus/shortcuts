"""
track_state.py

TrackState: core state container for BFS over patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Dict, List, Optional, Tuple, Union

from chord_diagram import ChordDiagram
from chords import BoundaryPoint, Chord
from edge import Port
from marked_strips import BoundaryEdge, EdgeRef, MarkedAnnulus, MarkedStrip
from square import Side


MarkedSurface = Union[MarkedStrip, MarkedAnnulus]
Cursor = Tuple[int, BoundaryPoint]  # (square index, boundary point)


def _edge_dir_value(surface: MarkedSurface, e: EdgeRef) -> str | None:
    if not hasattr(surface, "edge_direction"):
        return None
    d = surface.edge_direction(e)  # type: ignore[attr-defined]
    return getattr(d, "value", d)


def _dir_allows_in(val: str | None) -> bool:
    return val in (None, "in", "undirected")


def _dir_allows_out(val: str | None) -> bool:
    return val in (None, "out", "undirected")


def _paired_edge_ref(surface: MarkedSurface, e: EdgeRef) -> EdgeRef | None:
    if surface.is_interior_edge(e):
        return surface._interior_pair(e)  # type: ignore[attr-defined]
    if surface.is_marked_edge(e):
        be = BoundaryEdge(e.side, e.i)
        other, _is_rev = surface.pair_info(be)
        return EdgeRef(other.side, other.i)
    return None


@dataclass(frozen=True)
class TrackState:
    """
    Immutable state for BFS.

    - surface: marked strip/annulus with N squares
    - diagrams: chord diagram for each square (not necessarily a pattern)
    - cursor: current square index and boundary point
    - dy: integer displacement (can be negative)
    - dx: integer vector length N+1
    - lam: integer vector length N+1
    - dominant_x_dir: +1 (RIGHT) or -1 (LEFT)
    - dy_modifier: +1 or -1
    - or_pair_count: non-negative integer
    - turn_count: non-negative integer
    - last_hor_edge_top: last horizontal edge on TOP (True), BOTTOM (False), or None
    - last_vert_edge_left: last vertical edge on LEFT (True), RIGHT (False), or None
    - last_moved_dir: -1, 0, or 1 (0 means no horizontal move yet)
    """

    surface: MarkedSurface
    diagrams: Dict[int, ChordDiagram]
    cursor: Cursor
    dy: int
    dx: Tuple[int, ...]
    lam: Tuple[int, ...]
    dominant_x_dir: int
    dy_modifier: int
    or_pair_count: int
    turn_count: int
    last_hor_edge_top: Optional[bool]
    last_vert_edge_left: Optional[bool]
    last_moved_dir: int

    def __post_init__(self) -> None:
        n = self.surface.N
        if len(self.dx) != n + 1:
            raise ValueError("dx must have length N+1")
        if len(self.lam) != n + 1:
            raise ValueError("lam must have length N+1")
        if self.dominant_x_dir not in (-1, 1):
            raise ValueError("dominant_x_dir must be +1 or -1")
        if self.dy_modifier not in (-1, 1):
            raise ValueError("dy_modifier must be +1 or -1")
        if self.or_pair_count < 0:
            raise ValueError("or_pair_count must be non-negative")
        if self.turn_count < 0:
            raise ValueError("turn_count must be non-negative")
        if self.cursor[0] < 0 or self.cursor[0] >= n:
            raise ValueError("cursor square index out of range")

    def is_closed(self) -> bool:
        """
        Return True if the cursor's boundary point is used by a chord in that square.
        """
        square_i, bp = self.cursor
        diagram = self.diagrams.get(square_i)
        if diagram is None:
            return False
        for ch in diagram.chords:
            if ch.a == bp or ch.b == bp:
                return True
        return False

    def dx_linear_form(self, *, var_prefix: str = "x", pretty: bool = False) -> str:
        """
        Return dx[0] + dx[1]*x1 + ... as a readable linear expression.
        """
        if not self.dx:
            return "0"
        parts: List[str] = []
        const = self.dx[0]
        if const != 0:
            parts.append(str(const))
        for i, coef in enumerate(self.dx[1:], start=1):
            if coef == 0:
                continue
            sign = "+" if coef > 0 else "-"
            mag = abs(coef)
            if pretty:
                term = f"{var_prefix}_{i}"
                if mag != 1:
                    term = f"{mag}{term}"
            else:
                term = f"{mag}*{var_prefix}{i}" if mag != 1 else f"{var_prefix}{i}"
            if not parts:
                parts.append(f"-{term}" if coef < 0 else term)
            else:
                parts.append(f"{sign} {term}")
        return " ".join(parts) if parts else "0"

    def render(self, width: int = 25, height: int = 11, padding: int = 3) -> str:
        """
        Render the current track state with an '@' cursor marker.
        """
        # Lazy import to avoid circular dependency at load time.
        from pattern import Pattern

        pat = Pattern(surface=self.surface, diagrams=self.diagrams)
        base = pat.render(width=width, height=height, padding=padding)

        sq_i, bp = self.cursor
        # Build a cursor overlay for the square containing the cursor.
        lines = base.splitlines()
        if not lines:
            return base

        # Re-render just the cursor square to locate the '@' position.
        # This mirrors Pattern.render's square rendering for consistent placement.
        def _positions_along(length: int, count: int, pad: int) -> list[int]:
            if count <= 0:
                return []
            inner_len = max(1, length - 2 * pad)
            if count == 1:
                return [1 + pad + (inner_len - 1) // 2]
            return [
                int(round(1 + pad + i * (inner_len - 1) / (count - 1)))
                for i in range(count)
            ]

        # Figure out grid offset for square_i in the rendered output.
        # Layout is: optional wrap line, TOP label line, then height rows.
        row_offset = 1 if lines and lines[0].startswith("<->") else 0
        row_offset += 1  # TOP label line

        col_offset = len("        ") + sq_i * (width + 1)

        sq = self.surface.square(sq_i)
        inner_w = width - 2
        inner_h = height - 2

        if bp.side == Side.TOP:
            ports = list(sq.top.ports())
            xs = _positions_along(inner_w, len(ports), padding)
            xs = list(reversed(xs))
            x = xs[ports.index(bp.port)] + col_offset
            y = row_offset
        elif bp.side == Side.BOTTOM:
            ports = list(sq.bottom.ports())
            xs = _positions_along(inner_w, len(ports), padding)
            x = xs[ports.index(bp.port)] + col_offset
            y = row_offset + height - 1
        elif bp.side == Side.LEFT:
            ports = list(sq.left.ports())
            ys = _positions_along(inner_h, len(ports), padding)
            y = ys[ports.index(bp.port)] + row_offset
            x = col_offset
        else:
            ports = list(sq.right.ports())
            ys = _positions_along(inner_h, len(ports), padding)
            ys = list(reversed(ys))
            y = ys[ports.index(bp.port)] + row_offset
            x = col_offset + width - 1

        # Overlay '@' on the rendered output.
        if 0 <= y < len(lines):
            line = lines[y]
            if 0 <= x < len(line):
                lines[y] = line[:x] + "@" + line[x + 1 :]

        return "\n".join(lines)

    def moves(
        self,
        *,
        multiple_interior_edge_crossings: bool = True,
        one_dir_only: bool = False,
        dominant_dir_only: bool = False,
        max_ports_per_edge: int | None = None,
        debug_counts: Optional[dict] = None,
    ) -> List["TrackState"]:
        """
        Return a list of TrackStates reachable by adding one chord from the cursor.
        """
        if self.is_closed():
            return []

        square_i, bp = self.cursor
        if not _dir_allows_in(_edge_dir_value(self.surface, EdgeRef(bp.side, square_i))):
            if debug_counts is not None:
                debug_counts["dir_block_current_in"] = (
                    debug_counts.get("dir_block_current_in", 0) + 1
                )
            return []
        out: List[TrackState] = []
        sides = [Side.TOP, Side.RIGHT, Side.BOTTOM, Side.LEFT]
        sides = [s for s in sides if s != bp.side]

        for side in sides:
            if one_dir_only:
                effective_dir = self.last_moved_dir 
                if effective_dir == -1 and side == Side.RIGHT:
                    continue
                if effective_dir == 1 and side == Side.LEFT:
                    continue
            if dominant_dir_only:
                if self.dominant_x_dir == 1 and side == Side.LEFT:
                    continue
                if self.dominant_x_dir == -1 and side == Side.RIGHT:
                    continue
            e_ref = EdgeRef(side, square_i)
            edge = self.surface.square(square_i).edge(side)
            ports = list(edge.ports())
            if not _dir_allows_out(_edge_dir_value(self.surface, e_ref)):
                if debug_counts is not None:
                    debug_counts["dir_block_out"] = debug_counts.get("dir_block_out", 0) + 1
                continue
            paired_edge = _paired_edge_ref(self.surface, e_ref)
            if paired_edge is not None:
                if not _dir_allows_in(_edge_dir_value(self.surface, paired_edge)):
                    if debug_counts is not None:
                        debug_counts["dir_block_pair_in"] = (
                            debug_counts.get("dir_block_pair_in", 0) + 1
                        )
                    continue
            existing_candidates: List[Port] = []
            for p in ports:
                chord = Chord(BoundaryPoint(bp.side, bp.port), BoundaryPoint(side, p))
                if self.diagrams[square_i].can_add_chord(chord):
                    existing_candidates.append(p)

            if existing_candidates:
                if debug_counts is not None and self.surface.is_interior_edge(EdgeRef(side, square_i)):
                    debug_counts["interior_existing_candidates"] = (
                        debug_counts.get("interior_existing_candidates", 0) + 1
                    )
                for p in existing_candidates:
                    nxt = self._next_state_with_chord(
                        square_i,
                        bp,
                        side,
                        p,
                        dominant_dir_only=dominant_dir_only,
                    )
                    if nxt is not None:
                        out.append(nxt)
                continue

            # No existing valid port; try adding a new port on this edge.
            e_ref = EdgeRef(side, square_i)
            if self.surface.is_boundary_edge(e_ref):
                continue
            if max_ports_per_edge is not None and len(ports) >= max_ports_per_edge:
                continue
            # Cap also applies to the paired edge (marked or interior).
            if max_ports_per_edge is not None:
                paired_edge = None
                if self.surface.is_interior_edge(e_ref):
                    paired_edge = self.surface._interior_pair(e_ref)
                elif self.surface.is_marked_edge(e_ref):
                    be = BoundaryEdge(e_ref.side, e_ref.i)
                    other, _is_rev = self.surface.pair_info(be)
                    paired_edge = EdgeRef(other.side, other.i)
                if paired_edge is not None:
                    paired_ports = list(self.surface.square(paired_edge.i).edge(paired_edge.side).ports())
                    if len(paired_ports) >= max_ports_per_edge:
                        continue
            if debug_counts is not None and self.surface.is_interior_edge(e_ref):
                debug_counts["interior_add_attempts"] = (
                    debug_counts.get("interior_add_attempts", 0) + 1
                )
            if (not multiple_interior_edge_crossings) and self.surface.is_interior_edge(e_ref):
                if len(ports) >= 1:
                    continue
            # Try all insertion positions along the edge.
            if self.surface.is_interior_edge(e_ref):
                ports_local = self.surface._geom_order_ports(side, ports)
            else:
                ports_local = ports
            insertion_choices: list[tuple[Port | None, Port | None]] = []
            if not ports_local:
                insertion_choices.append((None, None))
            else:
                insertion_choices.append((None, ports_local[0]))  # before first
                for i in range(len(ports_local) - 1):
                    insertion_choices.append((ports_local[i], ports_local[i + 1]))
                insertion_choices.append((ports_local[-1], None))  # after last

            for left, right in insertion_choices:
                if self.surface.is_interior_edge(e_ref) and side == Side.RIGHT:
                    left, right = right, left
                nxt = self._next_state_with_new_port(
                    square_i,
                    bp,
                    e_ref,
                    left=left,
                    right=right,
                    dominant_dir_only=dominant_dir_only,
                    debug_counts=debug_counts,
                )
                if nxt is not None:
                    out.append(nxt)

        return out

    def _update_last_and_counts(
        self,
        *,
        surface: MarkedSurface,
        cursor: Cursor,
        entry_side: Side,
        entry_square: int,
        dominant_override: Optional[int] = None,
    ) -> Tuple[Optional[bool], Optional[bool], int, int, int, int, int, int]:
        # Apply bookkeeping when the path crosses into entry_side.
        last_h = self.last_hor_edge_top
        last_v = self.last_vert_edge_left
        turn = self.turn_count
        or_count = self.or_pair_count
        dy_modifier = self.dy_modifier
        dy = self.dy
        last_moved_dir = self.last_moved_dir
        dominant_x_dir = self.dominant_x_dir
        if dominant_override is not None:
            dominant_x_dir = dominant_override

        if entry_side in (Side.TOP, Side.BOTTOM):
            is_top = (entry_side == Side.TOP)
            if last_h is not None and last_h == is_top:
                turn += 1
                dy_modifier *= -1
            elif last_h is not None and last_h != is_top:
                dy += 1 * dy_modifier

        post_sq, post_bp = cursor
        if post_bp.side in (Side.TOP, Side.BOTTOM):
            last_h = (post_bp.side == Side.TOP)
            be = BoundaryEdge(post_bp.side, post_sq)
            if surface.is_marked(be):
                _other, is_rev = surface.pair_info(be)
                if is_rev:
                    or_count += 1
                    dominant_x_dir *= -1

        if entry_side in (Side.LEFT, Side.RIGHT):
            last_v = (entry_side == Side.LEFT)
            if entry_side == Side.RIGHT:
                last_moved_dir = -1
            else:
                last_moved_dir = 1

        return last_h, last_v, turn, or_count, dy, dy_modifier, last_moved_dir, dominant_x_dir

    @staticmethod
    def _interior_edge_index(
        surface: MarkedSurface,
        entry_side: Side,
        entry_square: int,
    ) -> Optional[int]:
        """
        Map an interior vertical edge to a stable index in 1..N.

        We index the edge shared by squares i-1 and i as index i (1-based).
        This means:
          - RIGHT edge of square i -> index i+1
          - LEFT edge of square i  -> index i (with square 0 -> index N)
        """
        e_ref = EdgeRef(entry_side, entry_square)
        if not surface.is_interior_edge(e_ref):
            return None
        if entry_side == Side.RIGHT:
            return entry_square + 1
        if entry_side == Side.LEFT:
            return entry_square if entry_square > 0 else surface.N
        return None

    def _next_state_with_chord(
        self,
        square_i: int,
        bp: BoundaryPoint,
        side: Side,
        port: Port,
        *,
        dominant_dir_only: bool = False,
    ) -> "TrackState | None":
        surface2, diagrams2, bp2, map_port = self._clone_with_mapped_cursor()
        port2 = map_port(side, square_i, port)
        chord = Chord(BoundaryPoint(bp2.side, bp2.port), BoundaryPoint(side, port2))
        if not diagrams2[square_i].add_chord(chord):
            return None

        paired = surface2.paired_boundary_point(EdgeRef(side, square_i), port2)
        if paired is None:
            return None
        e2, p2 = paired
        cursor2 = (e2.i, BoundaryPoint(e2.side, p2))
        return self._with_updated(
            surface2,
            diagrams2,
            cursor2,
            entry_side=side,
            entry_square=square_i,
            dominant_dir_only=dominant_dir_only,
        )

    def _next_state_with_new_port(
        self,
        square_i: int,
        bp: BoundaryPoint,
        e_ref: EdgeRef,
        *,
        left: Port | None = None,
        right: Port | None = None,
        dominant_dir_only: bool = False,
        debug_counts: Optional[dict] = None,
    ) -> "TrackState | None":
        surface2, diagrams2, bp2, map_port = self._clone_with_mapped_cursor()
        is_interior = surface2.is_interior_edge(e_ref)
        left2 = map_port(e_ref.side, e_ref.i, left) if left is not None else None
        right2 = map_port(e_ref.side, e_ref.i, right) if right is not None else None
        new_pair = surface2.add_port_between(e_ref, left=left2, right=right2)
        if new_pair is None:
            if debug_counts is not None and is_interior:
                debug_counts["interior_add_fail_add_port"] = (
                    debug_counts.get("interior_add_fail_add_port", 0) + 1
                )
            return None
        new_port, _paired_port = new_pair

        chord = Chord(BoundaryPoint(bp2.side, bp2.port), BoundaryPoint(e_ref.side, new_port))
        if not diagrams2[square_i].add_chord(chord):
            if debug_counts is not None and is_interior:
                debug_counts["interior_add_fail_add_chord"] = (
                    debug_counts.get("interior_add_fail_add_chord", 0) + 1
                )
            return None

        paired = surface2.paired_boundary_point(e_ref, new_port)
        if paired is None:
            if debug_counts is not None and is_interior:
                debug_counts["interior_add_fail_pair"] = (
                    debug_counts.get("interior_add_fail_pair", 0) + 1
                )
            return None
        e2, p2 = paired
        cursor2 = (e2.i, BoundaryPoint(e2.side, p2))
        if debug_counts is not None and is_interior:
            debug_counts["interior_add_success"] = (
                debug_counts.get("interior_add_success", 0) + 1
            )
        return self._with_updated(
            surface2,
            diagrams2,
            cursor2,
            entry_side=e_ref.side,
            entry_square=e_ref.i,
            dominant_dir_only=dominant_dir_only,
        )

    def _clone_with_mapped_cursor(self) -> Tuple[
        MarkedSurface, Dict[int, ChordDiagram], BoundaryPoint, callable
    ]:
        """
        Clone the surface and diagrams, returning the cloned cursor boundary point.
        """
        # Deep-copy the surface, then rebuild diagrams by port index.
        surface2 = copy.deepcopy(self.surface)
        diagrams2: Dict[int, ChordDiagram] = {}

        def _map_port(side: Side, sq_i: int, port: Port) -> Port:
            old_edge = self.surface.square(sq_i).edge(side)
            new_edge = surface2.square(sq_i).edge(side)
            ports_old = list(old_edge.ports())
            ports_new = list(new_edge.ports())
            idx = ports_old.index(port)
            return ports_new[idx]

        for i in range(self.surface.N):
            d_old = self.diagrams[i]
            d_new = ChordDiagram(square=surface2.square(i))
            for ch in d_old.chords:
                a = ch.a
                b = ch.b
                pa = _map_port(a.side, i, a.port)
                pb = _map_port(b.side, i, b.port)
                d_new.add_chord(Chord(BoundaryPoint(a.side, pa), BoundaryPoint(b.side, pb)))
            diagrams2[i] = d_new

        sq_i, bp = self.cursor
        mapped_port = _map_port(bp.side, sq_i, bp.port)
        return surface2, diagrams2, BoundaryPoint(bp.side, mapped_port), _map_port

    def _with_updated(
        self,
        surface: MarkedSurface,
        diagrams: Dict[int, ChordDiagram],
        cursor: Cursor,
        *,
        entry_side: Side,
        entry_square: int,
        dominant_dir_only: bool = False,
    ) -> "TrackState":
        dx = list(self.dx)
        lam = list(self.lam)
        dominant_override = None
        if dominant_dir_only and self.last_moved_dir == 0 and entry_side in (Side.LEFT, Side.RIGHT):
            dominant_override = 1 if entry_side == Side.RIGHT else -1
        if entry_side in (Side.LEFT, Side.RIGHT):
            idx = self._interior_edge_index(surface, entry_side, entry_square)
            if idx is not None:
                # Use the dominant direction at the time of the crossing.
                prev_dom = dominant_override if dominant_override is not None else self.dominant_x_dir
                moves_with_dominant = (
                    (entry_side == Side.RIGHT and prev_dom == 1)
                    or (entry_side == Side.LEFT and prev_dom == -1)
                )
                delta = 1 if moves_with_dominant else -1
                dx[0] += delta
                dx[idx] += delta
                lam[0] += 1
                lam[idx] += 1
        last_h, last_v, turn, or_count, dy, dy_modifier, last_moved_dir, dominant_x_dir = (
            self._update_last_and_counts(
                surface=surface,
                cursor=cursor,
                entry_side=entry_side,
                entry_square=entry_square,
                dominant_override=dominant_override,
            )
        )
        return TrackState(
            surface=surface,
            diagrams=diagrams,
            cursor=cursor,
            dy=dy,
            dx=tuple(dx),
            lam=tuple(lam),
            dominant_x_dir=dominant_x_dir,
            dy_modifier=dy_modifier,
            or_pair_count=or_count,
            turn_count=turn,
            last_hor_edge_top=last_h,
            last_vert_edge_left=last_v,
            last_moved_dir=last_moved_dir,
        )

    @classmethod
    def initialize(
        cls,
        surface: MarkedSurface,
        *,
        start_edge: EdgeRef,
        dy: int = 0,
        dominant_x_dir: int = 1,
        dy_modifier: int = 1,
        or_pair_count: int = 0,
        turn_count: int = 0,
        last_hor_edge_top: Optional[bool] = None,
        last_vert_edge_left: Optional[bool] = None,
        last_moved_dir: int = 0,
    ) -> "TrackState | None":
        """
        Initialize with exactly one port on every non-boundary edge and no chords.

        If the start_edge has no port (i.e., it is a boundary edge), return None.
        """
        n = surface.N

        # Set ports: boundary edges empty, all other edges exactly one port.
        for e in surface.all_edge_refs():
            edge_obj = surface.square(e.i).edge(e.side)
            if surface.is_boundary_edge(e):
                edge_obj._set_ports([])
            else:
                edge_obj._set_ports([Port(label=f"{e.side.short()}1")])

        # Validate start edge.
        if surface.is_boundary_edge(start_edge):
            return None
        if not _dir_allows_in(_edge_dir_value(surface, start_edge)):
            return None
        start_ports = surface.square(start_edge.i).edge(start_edge.side).ports()
        if not start_ports:
            return None

        cursor = (start_edge.i, BoundaryPoint(start_edge.side, start_ports[0]))
        diagrams = {i: ChordDiagram(square=surface.square(i)) for i in range(n)}

        if start_edge.side in (Side.TOP, Side.BOTTOM):
            last_hor_edge_top = (start_edge.side == Side.TOP)
        if start_edge.side in (Side.LEFT, Side.RIGHT):
            last_vert_edge_left = (start_edge.side == Side.LEFT)

        if start_edge.side in (Side.TOP, Side.BOTTOM):
            be = BoundaryEdge(start_edge.side, start_edge.i)
            if surface.is_marked(be):
                _other, is_rev = surface.pair_info(be)
                if is_rev:
                    dominant_x_dir *= -1

        return cls(
            surface=surface,
            diagrams=diagrams,
            cursor=cursor,
            dy=dy,
            dx=tuple(0 for _ in range(n + 1)),
            lam=tuple(0 for _ in range(n + 1)),
            dominant_x_dir=dominant_x_dir,
            dy_modifier=dy_modifier,
            or_pair_count=or_pair_count,
            turn_count=turn_count,
            last_hor_edge_top=last_hor_edge_top,
            last_vert_edge_left=last_vert_edge_left,
            last_moved_dir=last_moved_dir,
        )

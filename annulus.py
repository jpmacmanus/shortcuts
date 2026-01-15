# annulus.py
"""
Data model for Marked Annuli (Section 5 setting).

A marked annulus is encoded as:
- N squares arranged cyclically: 0..N-1
- Two boundary components: TOP and BOTTOM
- Boundary edges identified as (side, i) for i in 0..N-1
- A partial matching on boundary edges ("marked edges"), where each marked edge
  is paired with exactly one other boundary edge.
- Each marked pair carries a boolean flag: orientation_reversing.

This file provides:
- Pretty-printable, validated data structures
- Convenience helpers for construction and inspection
- A small self-test / demo when run directly:
    python annulus.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Tuple


class Side(str, Enum):
    TOP = "top"
    BOTTOM = "bottom"

    def short(self) -> str:
        return "T" if self is Side.TOP else "B"


@dataclass(frozen=True, order=True)
class BoundaryEdge:
    """
    A boundary edge of the annulus.
    side: TOP or BOTTOM
    i: square index in 0..N-1 (edge on boundary of square i on that side)
    """
    side: Side
    i: int

    def __str__(self) -> str:
        return f"{self.side.short()}{self.i}"

    def __repr__(self) -> str:
        return f"BoundaryEdge(side={self.side.value!r}, i={self.i})"


@dataclass(frozen=True)
class MarkedPair:
    """
    A pairing of two distinct boundary edges, with an orientation reversal flag.
    We store pairs as unordered for identity, but retain both endpoints.
    """
    a: BoundaryEdge
    b: BoundaryEdge
    orientation_reversing: bool = False

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise ValueError("MarkedPair endpoints must be distinct.")

    def endpoints(self) -> Tuple[BoundaryEdge, BoundaryEdge]:
        return (self.a, self.b)

    def other(self, e: BoundaryEdge) -> BoundaryEdge:
        if e == self.a:
            return self.b
        if e == self.b:
            return self.a
        raise KeyError(f"Edge {e} is not an endpoint of this pair.")

    def normalized_key(self) -> Tuple[BoundaryEdge, BoundaryEdge]:
        return tuple(sorted((self.a, self.b)))

    def __str__(self) -> str:
        flag = "rev" if self.orientation_reversing else "pres"
        return f"({self.a} ↔ {self.b}, {flag})"

    def __repr__(self) -> str:
        return (
            f"MarkedPair(a={self.a!r}, b={self.b!r}, "
            f"orientation_reversing={self.orientation_reversing})"
        )


@dataclass
class MarkedAnnulus:
    """
    Main container: N squares, and a set of marked boundary edge pairings.

    Internal representation uses endpoint->pair lookup for O(1) queries.
    """
    N: int
    _pairs: Dict[BoundaryEdge, MarkedPair] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.N, int) or self.N <= 0:
            raise ValueError("N must be a positive integer.")
        # validate any preloaded pairs
        self._validate_pair_dict()

    # -----------------------------
    # Surface adjacency (annulus-specific)
    # -----------------------------
    def vertical_edge_count(self) -> int:
        """Number of vertical edges between squares."""
        return self.N

    def next_square_horizontal(self, square_i: int, *, go_right: bool) -> int:
        """Neighbor square index after crossing a vertical edge."""
        if go_right:
            return (square_i + 1) % self.N
        return (square_i - 1) % self.N

    def vertical_edge_id(self, square_i: int, *, go_right: bool) -> int:
        """
        Identify a vertical edge (between squares) by an integer.

        Convention (matches the previous implementation):
          - Exiting square i to the right uses vertical edge i (between i and i+1).
          - Exiting square i to the left uses vertical edge i-1 (between i-1 and i), modulo N.
        """
        if go_right:
            return square_i
        return (square_i - 1) % self.N

    # -----------------------------
    # Core queries
    # -----------------------------
    def is_valid_edge(self, e: BoundaryEdge) -> bool:
        return 0 <= e.i < self.N

    def is_marked(self, e: BoundaryEdge) -> bool:
        return e in self._pairs

    def paired_edge(self, e: BoundaryEdge) -> BoundaryEdge:
        return self._pairs[e].other(e)

    def pair_info(self, e: BoundaryEdge) -> Tuple[BoundaryEdge, bool]:
        """
        Return (paired_edge, orientation_reversing) for marked edge e.
        """
        p = self._pairs[e]
        return (p.other(e), p.orientation_reversing)

    def all_pairs(self) -> List[MarkedPair]:
        """
        Return each marked pair exactly once.
        """
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
        """
        Squares incident to at least one marked boundary edge.
        """
        return sorted({e.i for e in self._pairs.keys()})

    def unmarked_squares(self) -> List[int]:
        ms = set(self.marked_squares())
        return [i for i in range(self.N) if i not in ms]

    def unmarked_segments(self) -> List[Tuple[int, int]]:
        """
        Return cyclic segments of consecutive unmarked squares between marked squares.

        Output format: list of (start_index, length), in cyclic order, where
        each segment is a maximal run of unmarked squares. If there are no marked
        squares, returns [(0, N)].

        Note: This is a purely combinatorial cyclic decomposition of indices; it is
        commonly the data you want when implementing §5.4-style "segment length" logic.
        """
        ms = set(self.marked_squares())
        if not ms:
            return [(0, self.N)]

        segs: List[Tuple[int, int]] = []
        i = 0
        visited = 0
        while visited < self.N:
            if i in ms:
                i = (i + 1) % self.N
                visited += 1
                continue
            # start of an unmarked run
            start = i
            length = 0
            while length < self.N and (i not in ms):
                length += 1
                i = (i + 1) % self.N
                visited += 1
                if visited >= self.N:
                    break
            segs.append((start, length))

        # Normalize cyclic order: rotate so smallest start index appears first for stable printing.
        if segs:
            k = min(range(len(segs)), key=lambda idx: segs[idx][0])
            segs = segs[k:] + segs[:k]
        return segs

    # -----------------------------
    # Construction helpers
    # -----------------------------
    def edge(self, side: Side, i: int) -> BoundaryEdge:
        e = BoundaryEdge(side=side, i=i)
        if not self.is_valid_edge(e):
            raise IndexError(f"Square index {i} out of range for N={self.N}.")
        return e

    def add_marked_pair(
        self,
        a: BoundaryEdge,
        b: BoundaryEdge,
        *,
        orientation_reversing: bool = False,
        overwrite: bool = False,
    ) -> None:
        """
        Add a marked pair (a,b). By default, raises if either endpoint is already marked.
        If overwrite=True, existing pairings at a or b are removed first.
        """
        self._validate_edge(a)
        self._validate_edge(b)
        if a == b:
            raise ValueError("Cannot pair an edge with itself.")

        if not overwrite:
            if a in self._pairs:
                raise ValueError(f"Edge {a} already marked (paired with {self.paired_edge(a)}).")
            if b in self._pairs:
                raise ValueError(f"Edge {b} already marked (paired with {self.paired_edge(b)}).")
        else:
            # remove any existing pairs involving a or b
            for e in (a, b):
                if e in self._pairs:
                    old = self._pairs[e]
                    del self._pairs[old.a]
                    del self._pairs[old.b]

        p = MarkedPair(a=a, b=b, orientation_reversing=orientation_reversing)
        self._pairs[a] = p
        self._pairs[b] = p

    def remove_marked_edge(self, e: BoundaryEdge) -> None:
        """
        Remove the marked pair containing e (if any).
        """
        if e not in self._pairs:
            return
        p = self._pairs[e]
        del self._pairs[p.a]
        del self._pairs[p.b]

    # -----------------------------
    # Validation
    # -----------------------------
    def validate(self) -> None:
        """
        Full validation. Raises ValueError on any inconsistency.
        """
        if self.N <= 0:
            raise ValueError("N must be positive.")
        self._validate_pair_dict()

    def _validate_edge(self, e: BoundaryEdge) -> None:
        if not self.is_valid_edge(e):
            raise ValueError(f"Invalid boundary edge {e}: index out of range for N={self.N}.")

    def _validate_pair_dict(self) -> None:
        """
        Ensure:
        - all edges are in range
        - each stored mapping is symmetric
        - no dangling endpoints
        """
        # range checks
        for e, p in self._pairs.items():
            self._validate_edge(e)
            self._validate_edge(p.a)
            self._validate_edge(p.b)

        # symmetry/dangling checks
        for e, p in self._pairs.items():
            if e != p.a and e != p.b:
                raise ValueError(f"Inconsistent pair dict: key {e} is not an endpoint of stored pair {p}.")
            oa, ob = p.a, p.b
            if oa not in self._pairs or ob not in self._pairs:
                raise ValueError(f"Dangling marked pair {p}: both endpoints must be present in dict.")
            if self._pairs[oa] is not p or self._pairs[ob] is not p:
                raise ValueError(f"Asymmetric marked pair storage for {p}.")

    # -----------------------------
    # Pretty printing / exporting
    # -----------------------------
    def __str__(self) -> str:
        lines = [f"MarkedAnnulus(N={self.N})"]
        pairs = self.all_pairs()
        if pairs:
            lines.append(f"  marked pairs ({len(pairs)}):")
            for p in pairs:
                lines.append(f"    - {p}")
        else:
            lines.append("  marked pairs: none")
        ms = self.marked_squares()
        lines.append(f"  marked squares: {ms if ms else 'none'}")
        segs = self.unmarked_segments()
        lines.append(f"  unmarked segments: {segs}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"MarkedAnnulus(N={self.N}, pairs={self.all_pairs()!r})"

    def to_dict(self) -> dict:
        """
        JSON-serializable-ish representation.
        """
        return {
            "N": self.N,
            "pairs": [
                {
                    "a": {"side": p.a.side.value, "i": p.a.i},
                    "b": {"side": p.b.side.value, "i": p.b.i},
                    "orientation_reversing": p.orientation_reversing,
                }
                for p in self.all_pairs()
            ],
        }


@dataclass
class MarkedStrip(MarkedAnnulus):
    """
    Marked strip: N squares arranged linearly (non-cyclic).

    The marking model (TOP/BOTTOM boundary edges paired with orientation flag)
    is identical to MarkedAnnulus. The only structural difference is the
    horizontal adjacency: there is no wrap-around between Q0 and Q(N-1).

    Consequences for track search:
      - Exiting LEFT from Q0 or RIGHT from Q(N-1) is impossible (hits boundary).
      - There are N-1 vertical edges.
    """

    def vertical_edge_count(self) -> int:
        return max(0, self.N - 1)

    def next_square_horizontal(self, square_i: int, *, go_right: bool) -> int:
        if go_right:
            if square_i >= self.N - 1:
                raise ValueError("Cannot exit RIGHT from last square in a strip.")
            return square_i + 1
        else:
            if square_i <= 0:
                raise ValueError("Cannot exit LEFT from first square in a strip.")
            return square_i - 1

    def vertical_edge_id(self, square_i: int, *, go_right: bool) -> int:
        """
        Identify a vertical edge (between squares) by an integer 0..N-2.

        Convention:
          - Exiting square i to the right uses vertical edge i (between i and i+1), for i=0..N-2.
          - Exiting square i to the left uses vertical edge i-1 (between i-1 and i), for i=1..N-1.
        """
        if go_right:
            if square_i >= self.N - 1:
                raise ValueError("No vertical edge to the right of the last square.")
            return square_i
        else:
            if square_i <= 0:
                raise ValueError("No vertical edge to the left of the first square.")
            return square_i - 1


# -----------------------------
# Demo / tests when run directly
# -----------------------------
def _demo_basic() -> None:
    print("=== Demo: basic construction ===")
    A = MarkedAnnulus(N=6)

    # Pair top edge at square 1 with bottom edge at square 4 (orientation preserving)
    A.add_marked_pair(A.edge(Side.TOP, 1), A.edge(Side.BOTTOM, 4), orientation_reversing=False)

    # Pair top edge at square 3 with bottom edge at square 0 (orientation reversing)
    A.add_marked_pair(A.edge(Side.TOP, 3), A.edge(Side.BOTTOM, 0), orientation_reversing=True)

    print(A)
    print()

    # Query examples
    e = A.edge(Side.TOP, 3)
    pe, rev = A.pair_info(e)
    print(f"Query: {e} is paired with {pe}, orientation_reversing={rev}")
    print(f"Marked edges: {A.marked_edges()}")
    print(f"Marked squares: {A.marked_squares()}")
    print(f"Unmarked squares: {A.unmarked_squares()}")
    print(f"Unmarked segments: {A.unmarked_segments()}")
    print(f"Export dict: {A.to_dict()}")
    print()


def _demo_validation_errors() -> None:
    print("=== Demo: validation errors ===")
    A = MarkedAnnulus(N=4)
    A.add_marked_pair(A.edge(Side.TOP, 0), A.edge(Side.BOTTOM, 2))
    try:
        # Attempt to reuse an already-marked endpoint
        A.add_marked_pair(A.edge(Side.TOP, 0), A.edge(Side.BOTTOM, 3))
    except ValueError as ex:
        print("Caught expected error:", ex)

    try:
        # Out-of-range edge
        A.add_marked_pair(BoundaryEdge(Side.TOP, 10), A.edge(Side.BOTTOM, 1))
    except ValueError as ex:
        print("Caught expected error:", ex)
    print()


def _demo_overwrite() -> None:
    print("=== Demo: overwrite pair ===")
    A = MarkedAnnulus(N=5)
    A.add_marked_pair(A.edge(Side.TOP, 0), A.edge(Side.BOTTOM, 0))
    print("Initial:")
    print(A)
    print()

    # Overwrite pairing involving TOP0
    A.add_marked_pair(
        A.edge(Side.TOP, 0),
        A.edge(Side.BOTTOM, 3),
        orientation_reversing=True,
        overwrite=True,
    )
    print("After overwrite:")
    print(A)
    print()


if __name__ == "__main__":
    _demo_basic()
    _demo_validation_errors()
    _demo_overwrite()

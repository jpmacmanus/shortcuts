"""
square.py

Square with four oriented edges: Top, Bottom, Left, Right.
Each edge is an Edge instance with its own linearly ordered ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, Tuple

from edge import Edge


class Side(str, Enum):
    """Canonical square sides with fixed orientation."""

    TOP = "top"
    RIGHT = "right"
    BOTTOM = "bottom"
    LEFT = "left"

    def short(self) -> str:
        return {"top": "T", "right": "R", "bottom": "B", "left": "L"}[self.value]


@dataclass
class Square:
    """
    A square consists of four oriented edges.

    The orientation is fixed by the Side enum. Ports live on edges and are
    ordered linearly within each edge.
    """

    top: Edge
    right: Edge
    bottom: Edge
    left: Edge

    @classmethod
    def empty(cls) -> "Square":
        """Construct a square with four empty edges."""
        return cls(
            top=Edge(base_side=Side.TOP.value),
            right=Edge(base_side=Side.RIGHT.value),
            bottom=Edge(base_side=Side.BOTTOM.value),
            left=Edge(base_side=Side.LEFT.value),
        )

    def edge(self, side: Side) -> Edge:
        """Return the edge for a given side."""
        if side == Side.TOP:
            return self.top
        if side == Side.RIGHT:
            return self.right
        if side == Side.BOTTOM:
            return self.bottom
        if side == Side.LEFT:
            return self.left
        raise ValueError(f"Unknown side: {side}")

    def edges(self) -> Tuple[Tuple[Side, Edge], ...]:
        """Return all edges in canonical order (TOP, RIGHT, BOTTOM, LEFT)."""
        return (
            (Side.TOP, self.top),
            (Side.RIGHT, self.right),
            (Side.BOTTOM, self.bottom),
            (Side.LEFT, self.left),
        )

    def edge_map(self) -> Dict[Side, Edge]:
        """Return a dictionary mapping each side to its edge."""
        return dict(self.edges())

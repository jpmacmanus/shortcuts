"""
strips.py

Models linear strips and cyclic annuli of squares. Adjacency is by identifying
the right edge of square i with the left edge of square i+1 (and wraparound for
annuli). Edge identification is represented by sharing the same Edge object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from square import Square


@dataclass
class SquareStrip:
    """
    Linear strip of squares.

    Squares are indexed 0..n-1. For i in [0, n-2], square i's RIGHT edge is
    identified with square (i+1)'s LEFT edge by sharing the Edge instance.
    """

    squares: List[Square]

    @classmethod
    def build(cls, n: int) -> "SquareStrip":
        if n <= 0:
            raise ValueError("n must be positive")
        squares = [Square.empty() for _ in range(n)]
        for i in range(n - 1):
            # Share the same Edge object for the interior identification.
            squares[i].right = squares[i + 1].left
        return cls(squares=squares)

    def __len__(self) -> int:
        return len(self.squares)

    def __iter__(self) -> Iterable[Square]:
        return iter(self.squares)

    def square(self, i: int) -> Square:
        return self.squares[i]


@dataclass
class Annulus:
    """
    Cyclic annulus of squares.

    Squares are indexed 0..n-1 with wraparound adjacency. As in the strip,
    square i's RIGHT edge is identified with square (i+1)'s LEFT edge; additionally,
    square (n-1)'s RIGHT edge is identified with square 0's LEFT edge.
    """

    squares: List[Square]

    @classmethod
    def build(cls, n: int) -> "Annulus":
        if n <= 0:
            raise ValueError("n must be positive")
        squares = [Square.empty() for _ in range(n)]
        for i in range(n - 1):
            # Linear identifications, then wrap the last to the first.
            squares[i].right = squares[i + 1].left
        squares[-1].right = squares[0].left
        return cls(squares=squares)

    def __len__(self) -> int:
        return len(self.squares)

    def __iter__(self) -> Iterable[Square]:
        return iter(self.squares)

    def square(self, i: int) -> Square:
        return self.squares[i % len(self.squares)]

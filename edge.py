"""
edge.py

Minimal model for an edge with linearly ordered ports.

An Edge is just an ordered list of Port objects. Ports have no geometry here;
their order in the list represents their linear order along the edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple


@dataclass(frozen=True, eq=False)
class Port:
    """
    A simple marker object for an edge port.

    The optional label is for debugging or external identification only; it is
    not used for ordering. Ordering is determined by the Edge's internal list.
    Equality is by identity, not by label.
    """

    label: Optional[str] = None

    def __str__(self) -> str:
        return f"Port({self.label})" if self.label is not None else "Port()"


class Edge:
    """
    Edge with a linear ordering of ports.

    Ports are stored in a list; the list order represents their position along
    the edge from left to right (or any consistent direction you choose).
    """

    def __init__(self, *, base_side: Optional[str] = None) -> None:
        self._ports: List[Port] = []
        # Optional hint for orientation: side string that this edge was created for.
        self.base_side: Optional[str] = base_side

    def ports(self) -> Tuple[Port, ...]:
        """Return the ports in order as an immutable tuple."""
        return tuple(self._ports)

    def add_port(self, label: Optional[str] = None, *, index: Optional[int] = None) -> Port:
        """
        Add a new port.

        If index is None, append at the end. Otherwise insert at the given index
        (0 <= index <= len(ports)).
        """
        port = Port(label=label)
        if index is None:
            self._ports.append(port)
        else:
            if not (0 <= index <= len(self._ports)):
                raise IndexError("index out of range for port insertion")
            self._ports.insert(index, port)
        return port

    def add_port_between(
        self,
        *,
        left: Optional[Port] = None,
        right: Optional[Port] = None,
        label: Optional[str] = None,
    ) -> Port:
        """
        Insert a new port between two existing ports.

        - If both left and right are given, they must be adjacent with left
          immediately before right.
        - If left is None, insert immediately before right.
        - If right is None, insert immediately after left.
        - If both are None, insert at the end (equivalent to add_port()).
        """
        if left is None and right is None:
            return self.add_port(label=label)

        if right is None:
            if left not in self._ports:
                raise ValueError("left port not on this edge")
            idx = self._ports.index(left) + 1
            return self.add_port(label=label, index=idx)

        if left is None:
            if right not in self._ports:
                raise ValueError("right port not on this edge")
            idx = self._ports.index(right)
            return self.add_port(label=label, index=idx)

        if left not in self._ports or right not in self._ports:
            raise ValueError("left/right ports must be on this edge")

        left_idx = self._ports.index(left)
        right_idx = self._ports.index(right)
        if right_idx != left_idx + 1:
            raise ValueError("left and right ports are not adjacent in order")

        return self.add_port(label=label, index=right_idx)

    def remove_port(self, port: Port) -> None:
        """Remove the given port by identity."""
        self._ports.remove(port)

    def remove_port_at(self, index: int) -> Port:
        """Remove and return the port at the given index."""
        return self._ports.pop(index)

    def index_of(self, port: Port) -> int:
        """Return the index of a port in the current order."""
        return self._ports.index(port)

    def __len__(self) -> int:
        return len(self._ports)

    def __iter__(self) -> Iterable[Port]:
        return iter(self._ports)

    def _set_ports(self, ports: List[Port]) -> None:
        """Replace the port list (internal use only)."""
        self._ports = ports

"""
shortcut_completeness.py

Feasibility checker for candidate simple shortcuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

from track_state import TrackState


@dataclass(frozen=True)
class CompletenessResult:
    complete: bool
    reason: str


def _collect_dx(track_states: Iterable[TrackState]) -> List[Sequence[int]]:
    dx_rows: List[Sequence[int]] = []
    for st in track_states:
        if not st.is_closed():
            raise ValueError("All track states must be closed for completeness checking")
        dx_rows.append(st.dx)
    return dx_rows


def _linprog_feasible(a_eq: List[List[float]], b_eq: List[float]) -> bool:
    """
    Return True if there exists x >= 0 with A_eq x = b_eq.
    """
    try:
        from scipy.optimize import linprog
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "scipy is required for completeness checking (scipy.optimize.linprog)."
        ) from exc

    if not a_eq:
        return True

    n = len(a_eq[0])
    res = linprog(
        c=[0.0] * n,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=[(0, None)] * n,
        method="highs",
    )
    return res.success


def is_complete(track_states: Iterable[TrackState]) -> bool:
    """
    Return True iff the candidate set is complete (LP is infeasible).
    """
    rows = _collect_dx(track_states)
    if not rows:
        return False
    n = len(rows[0]) - 1
    if n < 0:
        raise ValueError("dx must have length at least 1")
    for r in rows:
        if len(r) != n + 1:
            raise ValueError("All dx vectors must have the same length")

    # dx[0] + sum_i dx[i] * x_i = 0  =>  sum_i dx[i] * x_i = -dx[0]
    a_eq = [[float(v) for v in r[1:]] for r in rows]
    b_eq = [float(-r[0]) for r in rows]
    feasible = _linprog_feasible(a_eq, b_eq)
    return not feasible


def check_completeness(track_states: Iterable[TrackState]) -> CompletenessResult:
    """
    Return a result with a simple reason string for logging.
    """
    rows = _collect_dx(track_states)
    if not rows:
        return CompletenessResult(complete=False, reason="empty candidate set")
    complete = is_complete(track_states)
    return CompletenessResult(
        complete=complete,
        reason="infeasible" if complete else "feasible",
    )


def _greedy_complete_subset(track_states: List[TrackState], *, rng) -> List[TrackState]:
    shuffled = track_states[:]
    rng.shuffle(shuffled)
    chosen: List[TrackState] = []
    for st in shuffled:
        chosen.append(st)
        if is_complete(chosen):
            return chosen
    return []


def _minimal_complete_subset(track_states: List[TrackState]) -> List[TrackState]:
    from itertools import combinations

    n = len(track_states)
    for size in range(1, n + 1):
        for combo in combinations(track_states, size):
            subset = list(combo)
            if is_complete(subset):
                return subset
    return []


def reduce_complete_set(
    track_states: Iterable[TrackState],
    *,
    seed: int | None = None,
    max_minimize_size: int = 20,
) -> List[TrackState]:
    """
    Return a smaller complete subset, using randomized greedy selection.

    If the greedy subset size is <= max_minimize_size, run an exhaustive
    search to find a minimal complete subset of that greedy subset.
    """
    from random import Random

    states = list(track_states)
    if not states:
        return []
    if not is_complete(states):
        raise ValueError("Input set must be complete")

    rng = Random(seed)
    greedy = _greedy_complete_subset(states, rng=rng)
    if not greedy:
        return []
    if len(greedy) <= max_minimize_size:
        minimal = _minimal_complete_subset(greedy)
        return minimal if minimal else greedy
    return greedy

# search_shortcuts_weighted.py
"""
Weighted candidate shortcut search, patched to handle OR edges used multiple times by
RECOMPUTING degrees from the current TrackState (instead of maintaining incremental
degree dictionaries whose slot labels become stale under OR slot-shifts).

This patch fixes: TrackState has no .square(i); it has .square_chords(i).
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
from fractions import Fraction
from itertools import combinations
import random
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple, Union

from annulus import BoundaryEdge, MarkedAnnulus, Side
from chord_diagram import BoundaryPoint, PortKind
from track_state import TrackState, initial_state, step, vertical_edge_id, boundary_edge_for

from search_shortcuts import find_shortcut as find_unweighted_shortcut
from search_shortcuts import global_point_for as unweighted_global_point_for
from search_shortcuts import ShortcutResult as UnweightedShortcutResult


# ----------------------------
# Config
# ----------------------------

START_FROM_VERTICAL_EDGES: bool = False  # usually False is best; start from marked edges


# ----------------------------
# Simple shortcut fast path
# ----------------------------

def find_simple_shortcut(
    ann: MarkedAnnulus,
    *,
    start_square: Optional[int] = None,
    start_point: Optional[BoundaryPoint] = None,
    max_steps: Optional[int] = None,
    max_nodes: int = 250_000,
    verbose: bool = False,
    max_pair_slots: int = 8,
    max_total_pair_slots: int = 64,
) -> Optional[UnweightedShortcutResult]:
    return find_unweighted_shortcut(
        ann,
        start_square=start_square,
        start_point=start_point,
        max_steps=max_steps,
        max_nodes=max_nodes,
        verbose=verbose,
        simple_only=True,
        max_pair_slots=max_pair_slots,
        max_total_pair_slots=max_total_pair_slots,
    )


# ----------------------------
# Helpers
# ----------------------------

def _side_code(side: Side) -> int:
    if side == Side.TOP:
        return 0
    if side == Side.BOTTOM:
        return 1
    v = getattr(side, "value", None)
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return 0 if v.lower().startswith("t") else 1
    return 0


def _try_vertical_edge_id(ann: MarkedAnnulus, square_i: int, kind: PortKind) -> Optional[int]:
    try:
        return int(vertical_edge_id(ann, square_i, kind))
    except ValueError:
        return None


# ----------------------------
# dy bookkeeping
# ----------------------------

@dataclass(frozen=True)
class DyBookkeeping:
    Y: int = 0
    s_sign: int = 1
    turn_parity: int = 0
    piece_start_side: Optional[Side] = None


def update_dy(
    ann: MarkedAnnulus,
    st_before: TrackState,
    exit_kind: PortKind,
    dybk: DyBookkeeping,
) -> DyBookkeeping:
    i = st_before.pos.square

    Y = dybk.Y
    s = dybk.s_sign
    turn_parity = dybk.turn_parity
    piece_start_side = dybk.piece_start_side

    if exit_kind in (PortKind.TOP, PortKind.BOTTOM):
        end_side = Side.TOP if exit_kind == PortKind.TOP else Side.BOTTOM
        if piece_start_side is None:
            piece_start_side = end_side

        if piece_start_side == end_side:
            s = -s
            turn_parity ^= 1
        else:
            Y += s

        e = boundary_edge_for(i, exit_kind)
        e2, _is_rev = ann.pair_info(e)
        piece_start_side = e2.side

    return DyBookkeeping(Y=Y, s_sign=s, turn_parity=turn_parity, piece_start_side=piece_start_side)


def final_dy(dybk: DyBookkeeping) -> int:
    return 0 if (dybk.turn_parity & 1) else abs(dybk.Y)


# ----------------------------
# Weighted dx bookkeeping: affine linear form
# ----------------------------

@dataclass(frozen=True)
class DxLinearForm:
    # dx(w) = const + sum_e coeff[e] * w_e
    const: int = 0
    coeffs: Tuple[Tuple[int, int], ...] = tuple()  # sorted (edge_id, coeff)

    def as_dict(self) -> Dict[int, int]:
        return dict(self.coeffs)

    @staticmethod
    def from_dict(const: int, d: Dict[int, int]) -> "DxLinearForm":
        d2 = {k: v for k, v in d.items() if v != 0}
        return DxLinearForm(const=const, coeffs=tuple(sorted(d2.items())))

    def add_edge_crossing(self, edge_id: int, sign: int) -> "DxLinearForm":
        d = self.as_dict()
        d[edge_id] = d.get(edge_id, 0) + sign
        # contribution sign*(w+1): coeff += sign; const += sign
        return DxLinearForm.from_dict(self.const + sign, d)


@dataclass(frozen=True)
class WeightedBookkeeping:
    dx: DxLinearForm
    dominant_right: bool
    rev_parity: int
    dy: DyBookkeeping


def update_weighted_bookkeeping(
    ann: MarkedAnnulus,
    st_before: TrackState,
    exit_kind: PortKind,
    bk: WeightedBookkeeping,
    *,
    simple_only: bool = False,
) -> Optional[WeightedBookkeeping]:
    i = st_before.pos.square

    dx = bk.dx
    dominant_right = bk.dominant_right
    rev_parity = bk.rev_parity
    dy = bk.dy

    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        d = 1 if exit_kind == PortKind.RIGHT else -1

        if simple_only:
            dom = 1 if dominant_right else -1
            if d != dom:
                return None

        vid = _try_vertical_edge_id(ann, i, exit_kind)
        if vid is None:
            return None

        dom = 1 if dominant_right else -1
        sign = 1 if d == dom else -1
        dx = dx.add_edge_crossing(int(vid), sign)

        return WeightedBookkeeping(dx=dx, dominant_right=dominant_right, rev_parity=rev_parity, dy=dy)

    if exit_kind in (PortKind.TOP, PortKind.BOTTOM):
        dy2 = update_dy(ann, st_before, exit_kind, dy)

        e = boundary_edge_for(i, exit_kind)
        _e2, is_rev = ann.pair_info(e)
        if is_rev:
            dominant_right = not dominant_right
            rev_parity ^= 1

        return WeightedBookkeeping(dx=dx, dominant_right=dominant_right, rev_parity=rev_parity, dy=dy2)

    return None


# ----------------------------
# Global degree recomputation from state
# ----------------------------

GlobalPoint = Union[Tuple[str, int], Tuple[str, BoundaryEdge, int]]


def _gp_sort_key(gp: GlobalPoint) -> Tuple[Any, ...]:
    if gp[0] == "V":
        return (0, gp[1])
    return (1, _side_code(gp[1].side), gp[1].i, gp[2])  # type: ignore[index]


def _inc_deg(d: Dict[GlobalPoint, int], gp: GlobalPoint, delta: int = 1) -> None:
    d[gp] = d.get(gp, 0) + delta
    if d[gp] == 0:
        del d[gp]


def compute_global_degrees(ann: MarkedAnnulus, st: TrackState) -> Tuple[Tuple[GlobalPoint, int], ...]:
    d: Dict[GlobalPoint, int] = {}
    for i, sq in st.squares:
        for a, b in sq.chords:
            gp_a = unweighted_global_point_for(ann, i, a)
            gp_b = unweighted_global_point_for(ann, i, b)
            _inc_deg(d, gp_a, 1)
            _inc_deg(d, gp_b, 1)
    return tuple(sorted(d.items(), key=lambda kv: _gp_sort_key(kv[0])))


def max_degree(degrees: Tuple[Tuple[Any, int], ...]) -> int:
    return max((v for _, v in degrees), default=0)


def all_degrees_two_global(gdegrees: Tuple[Tuple[GlobalPoint, int], ...]) -> bool:
    return all(v == 2 for _, v in gdegrees)


def count_degree_one_global(gdegrees: Tuple[Tuple[GlobalPoint, int], ...]) -> int:
    return sum(1 for _, v in gdegrees if v == 1)


# ----------------------------
# Local occupancy check (no incremental local-degree dict)
# ----------------------------

def endpoint_used_in_square(sq_chords: Iterable[Tuple[BoundaryPoint, BoundaryPoint]], p: BoundaryPoint) -> bool:
    for a, b in sq_chords:
        if a == p or b == p:
            return True
    return False


# ----------------------------
# Slot budgets (pair-class)
# ----------------------------

def _pair_class_reps(ann: MarkedAnnulus) -> Dict[BoundaryEdge, Tuple[BoundaryEdge, BoundaryEdge]]:
    reps: Dict[BoundaryEdge, Tuple[BoundaryEdge, BoundaryEdge]] = {}
    for p in ann.all_pairs():
        rep = min(p.a, p.b)
        reps[rep] = (p.a, p.b)
    return reps


def slot_budget_ok(
    ann: MarkedAnnulus,
    st: TrackState,
    *,
    max_pair_slots: int,
    max_total_pair_slots: int,
) -> bool:
    reps = _pair_class_reps(ann)
    total = 0
    for _rep, (a, b) in reps.items():
        sa = st.slot_count(a)
        sb = st.slot_count(b)
        m = sa if sa >= sb else sb
        if m > max_pair_slots:
            return False
        total += m
        if total > max_total_pair_slots:
            return False
    return True


# ----------------------------
# Start positions
# ----------------------------

def _start_positions(ann: MarkedAnnulus) -> List[Tuple[int, BoundaryPoint]]:
    starts: List[Tuple[int, BoundaryPoint]] = []

    for i in range(ann.N):
        if ann.is_marked(ann.edge(Side.TOP, i)):
            starts.append((i, BoundaryPoint(PortKind.TOP, 1)))
        if ann.is_marked(ann.edge(Side.BOTTOM, i)):
            starts.append((i, BoundaryPoint(PortKind.BOTTOM, 1)))

    if START_FROM_VERTICAL_EDGES:
        for i in range(ann.N):
            if _try_vertical_edge_id(ann, i, PortKind.LEFT) is not None:
                starts.append((i, BoundaryPoint(PortKind.LEFT, 1)))
            if _try_vertical_edge_id(ann, i, PortKind.RIGHT) is not None:
                starts.append((i, BoundaryPoint(PortKind.RIGHT, 1)))

    seen = set()
    out: List[Tuple[int, BoundaryPoint]] = []
    for s in starts:
        key = (s[0], s[1].kind, s[1].slot)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


# ----------------------------
# Move ordering (strip-safe)
# ----------------------------

def possible_exit_kinds(ann: MarkedAnnulus, st: TrackState) -> List[PortKind]:
    i = st.pos.square
    kinds: List[PortKind] = []

    if _try_vertical_edge_id(ann, i, PortKind.LEFT) is not None:
        kinds.append(PortKind.LEFT)
    if _try_vertical_edge_id(ann, i, PortKind.RIGHT) is not None:
        kinds.append(PortKind.RIGHT)

    if ann.is_marked(ann.edge(Side.TOP, i)):
        kinds.append(PortKind.TOP)
    if ann.is_marked(ann.edge(Side.BOTTOM, i)):
        kinds.append(PortKind.BOTTOM)

    return kinds


def _exit_cost(ann: MarkedAnnulus, st: TrackState, exit_kind: PortKind) -> int:
    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        return 0
    i = st.pos.square
    e = boundary_edge_for(i, exit_kind)
    if not ann.is_marked(e):
        return 10_000
    e2, _rev = ann.pair_info(e)
    k = st.slot_count(e) + 1
    other = st.slot_count(e2)
    resulting = k if k >= other else other
    return 10 + resulting


def ordered_exit_kinds(ann: MarkedAnnulus, st: TrackState) -> List[PortKind]:
    return sorted(possible_exit_kinds(ann, st), key=lambda k: _exit_cost(ann, st, k))


def chord_fingerprint(st: TrackState) -> Tuple[Tuple[int, int, int, int], ...]:
    out = []
    for i, sq in st.squares:
        try:
            h = hash(sq.chords)
        except Exception:
            h = hash(tuple(sorted(sq.chords)))
        out.append((i, h, getattr(sq, "top_slots", 0), getattr(sq, "bottom_slots", 0)))
    return tuple(sorted(out))


# ----------------------------
# Search node / candidate
# ----------------------------

@dataclass(frozen=True)
class WeightedNode:
    st: TrackState
    bk: WeightedBookkeeping
    word: Tuple[PortKind, ...]


@dataclass(frozen=True)
class WeightedCandidate:
    st: TrackState
    dx: DxLinearForm
    dy: int
    lam: int
    rev_parity: int
    word: Tuple[PortKind, ...]


# ----------------------------
# Transition + candidate predicate
# ----------------------------

def apply_move(
    ann: MarkedAnnulus,
    node: WeightedNode,
    exit_kind: PortKind,
    *,
    simple_only: bool,
) -> Optional[WeightedNode]:
    st = node.st
    i = st.pos.square

    # Prune redundant "same-side" chords: do not allow a chord to begin and end
    # on the same side of a square (TOP→TOP, BOTTOM→BOTTOM, LEFT→LEFT, RIGHT→RIGHT).
    if exit_kind == st.pos.point.kind:
        return None


    if exit_kind in (PortKind.LEFT, PortKind.RIGHT):
        if _try_vertical_edge_id(ann, i, exit_kind) is None:
            return None
        b_pt = BoundaryPoint(exit_kind, 1)
    else:
        e = boundary_edge_for(i, exit_kind)
        if not ann.is_marked(e):
            return None
        b_pt = BoundaryPoint(exit_kind, st.slot_count(e) + 1)

    # Use TrackState.square_chords(i), not .square(i)
    sq = st.square_chords(i)

    # Local gate occupancy check in THIS square (current labels)
    if endpoint_used_in_square(sq.chords, b_pt):
        return None
    if endpoint_used_in_square(sq.chords, st.pos.point):
        return None

    bk2 = update_weighted_bookkeeping(ann, st, exit_kind, node.bk, simple_only=simple_only)
    if bk2 is None:
        return None

    try:
        st2 = step(ann, st, exit_kind)
    except Exception:
        return None

    return WeightedNode(st=st2, bk=bk2, word=node.word + (exit_kind,))


def is_candidate(ann: MarkedAnnulus, node: WeightedNode) -> Optional[WeightedCandidate]:
    if not node.word:
        return None
    if node.bk.rev_parity & 1:
        return None

    dy = final_dy(node.bk.dy)
    if dy == 0:
        return None

    gdeg = compute_global_degrees(ann, node.st)
    if not all_degrees_two_global(gdeg):
        return None

    return WeightedCandidate(
        st=node.st,
        dx=node.bk.dx,
        dy=dy,
        lam=node.st.lam,
        rev_parity=node.bk.rev_parity,
        word=node.word,
    )


# ----------------------------
# Candidate enumeration (BFS)
# ----------------------------

def find_candidates_weighted(
    ann: MarkedAnnulus,
    *,
    start_square: Optional[int] = None,
    start_point: Optional[BoundaryPoint] = None,
    max_steps: Optional[int] = None,
    max_nodes: int = 250_000,
    max_candidates: int = 5_000,
    verbose: bool = False,
    max_pair_slots: int = 8,
    max_total_pair_slots: int = 64,
) -> List[WeightedCandidate]:
    if max_steps is None:
        max_steps = 8 * ann.N + 8

    if start_square is not None or start_point is not None:
        if start_square is None or start_point is None:
            raise TypeError("start_square and start_point must be provided together")
        starts = [(start_square, start_point)]
    else:
        starts = _start_positions(ann)

    if not starts:
        return []

    out: List[WeightedCandidate] = []

    for (sq0, pt0) in starts:
        st0 = initial_state(sq0, pt0)

        start_side: Optional[Side]
        if pt0.kind == PortKind.TOP:
            start_side = Side.TOP
        elif pt0.kind == PortKind.BOTTOM:
            start_side = Side.BOTTOM
        else:
            start_side = None

        root = WeightedNode(
            st=st0,
            bk=WeightedBookkeeping(
                dx=DxLinearForm(),
                dominant_right=True,
                rev_parity=0,
                dy=DyBookkeeping(piece_start_side=start_side),
            ),
            word=tuple(),
        )

        q: Deque[WeightedNode] = deque([root])
        seen: Set[Tuple[Any, ...]] = set()
        expanded = 0

        while q:
            node = q.popleft()
            expanded += 1
            if expanded > max_nodes:
                if verbose:
                    print(f"[search_shortcuts_weighted] hit max_nodes={max_nodes} (start={sq0}:{pt0})")
                break

            depth = len(node.word)
            if depth > max_steps:
                continue

            if not slot_budget_ok(
                ann,
                node.st,
                max_pair_slots=max_pair_slots,
                max_total_pair_slots=max_total_pair_slots,
            ):
                continue

            gdeg = compute_global_degrees(ann, node.st)
            u1 = count_degree_one_global(gdeg)
            remaining = max_steps - depth
            if (u1 + 1) // 2 > remaining:
                continue

            cand = is_candidate(ann, node)
            if cand is not None:
                out.append(cand)
                if len(out) >= max_candidates:
                    return out

            key = (
                node.st.pos.square,
                node.st.pos.point.kind,
                node.st.pos.point.slot,
                node.st.lam,
                node.st.used_vertical,
                node.st.slot_counts,
                node.bk,
                gdeg,
                chord_fingerprint(node.st),
            )
            if key in seen:
                continue
            seen.add(key)

            for exit_kind in ordered_exit_kinds(ann, node.st):
                nxt = apply_move(ann, node, exit_kind, simple_only=False)
                if nxt is None:
                    continue
                q.append(nxt)

    return out


# ----------------------------
# Feasibility: A w = b, w >= 0 over reals (pure Python, rationals)
# ----------------------------

@dataclass(frozen=True)
class RealFeasibilityResult:
    feasible: bool
    witness_w: Optional[Tuple[float, ...]] = None
    status: str = ""


def _num_vertical_edges(ann: MarkedAnnulus) -> int:
    return int(getattr(ann, "N", 0))


def _to_fraction_matrix(A: List[List[int]], b: List[int]) -> Tuple[List[List[Fraction]], List[Fraction]]:
    Af = [[Fraction(x) for x in row] for row in A]
    bf = [Fraction(x) for x in b]
    return Af, bf


def _row_reduce_independent(A: List[List[Fraction]], b: List[Fraction]) -> Tuple[List[List[Fraction]], List[Fraction]]:
    m = len(A)
    if m == 0:
        return [], []
    n = len(A[0])

    rows = [A[i][:] + [b[i]] for i in range(m)]
    r = 0

    for c in range(n):
        pivot = None
        for i in range(r, m):
            if rows[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue

        rows[r], rows[pivot] = rows[pivot], rows[r]
        pv = rows[r][c]
        rows[r] = [x / pv for x in rows[r]]

        for i in range(r + 1, m):
            if rows[i][c] == 0:
                continue
            f = rows[i][c]
            rows[i] = [rows[i][j] - f * rows[r][j] for j in range(n + 1)]

        r += 1
        if r == m:
            break

    for i in range(r, m):
        if all(rows[i][j] == 0 for j in range(n)) and rows[i][n] != 0:
            return [[Fraction(0)] * n], [Fraction(1)]

    indep = rows[:r]
    A2 = [row[:n] for row in indep]
    b2 = [row[n] for row in indep]
    return A2, b2


def _solve_square_system(A: List[List[Fraction]], b: List[Fraction]) -> Optional[List[Fraction]]:
    r = len(A)
    if r == 0:
        return []
    M = [A[i][:] + [b[i]] for i in range(r)]

    for col in range(r):
        pivot = None
        for i in range(col, r):
            if M[i][col] != 0:
                pivot = i
                break
        if pivot is None:
            return None
        M[col], M[pivot] = M[pivot], M[col]
        pv = M[col][col]
        M[col] = [x / pv for x in M[col]]
        for i in range(col + 1, r):
            if M[i][col] == 0:
                continue
            f = M[i][col]
            M[i] = [M[i][j] - f * M[col][j] for j in range(r + 1)]

    x = [Fraction(0) for _ in range(r)]
    for i in range(r - 1, -1, -1):
        rhs = M[i][r] - sum(M[i][j] * x[j] for j in range(i + 1, r))
        x[i] = rhs
    return x


def _mat_vec(A: List[List[Fraction]], w: List[Fraction]) -> List[Fraction]:
    return [sum(Ai[j] * w[j] for j in range(len(w))) for Ai in A]


def feasibility_Aw_eq_b_w_ge_0(A_int: List[List[int]], b_int: List[int]) -> RealFeasibilityResult:
    if not A_int:
        return RealFeasibilityResult(feasible=True, witness_w=None, status="no equations")

    Af, bf = _to_fraction_matrix(A_int, b_int)
    Ared, bred = _row_reduce_independent(Af, bf)

    if len(Ared) == 1 and all(x == 0 for x in Ared[0]) and bred[0] == 1:
        return RealFeasibilityResult(feasible=False, witness_w=None, status="inconsistent equations")

    r = len(Ared)
    n = len(Ared[0]) if r > 0 else len(Af[0])

    if r == 0:
        return RealFeasibilityResult(
            feasible=True,
            witness_w=tuple(0.0 for _ in range(n)),
            status="all equations redundant",
        )

    cols = range(n)
    for basis in combinations(cols, r):
        Ab = [[Ared[i][j] for j in basis] for i in range(r)]
        xb = _solve_square_system(Ab, bred)
        if xb is None:
            continue

        w = [Fraction(0) for _ in range(n)]
        ok = True
        for idx, col in enumerate(basis):
            if xb[idx] < 0:
                ok = False
                break
            w[col] = xb[idx]
        if not ok:
            continue

        lhs = _mat_vec(Ared, w)
        if any(lhs[i] != bred[i] for i in range(r)):
            continue

        return RealFeasibilityResult(
            feasible=True,
            witness_w=tuple(float(x) for x in w),
            status=f"feasible (rank={r}, basis={basis})",
        )

    return RealFeasibilityResult(feasible=False, witness_w=None, status=f"infeasible (rank={r})")


def is_complete_candidate_set_over_reals(
    ann: MarkedAnnulus,
    candidates: Iterable[WeightedCandidate],
) -> Tuple[bool, RealFeasibilityResult]:
    cands = list(candidates)
    if not cands:
        return False, RealFeasibilityResult(feasible=True, witness_w=None, status="no candidates")

    n = _num_vertical_edges(ann)

    A: List[List[int]] = []
    b: List[int] = []
    for cand in cands:
        row = [0] * n
        for eid, coeff in cand.dx.coeffs:
            if not (0 <= eid < n):
                return False, RealFeasibilityResult(
                    feasible=False,
                    witness_w=None,
                    status=f"edge id {eid} out of range [0,{n})",
                )
            row[eid] = coeff
        A.append(row)
        b.append(-cand.dx.const)

    feas = feasibility_Aw_eq_b_w_ge_0(A, b)
    if feas.feasible:
        return False, feas
    return True, feas


# ----------------------------
# Candidate set minimisation (heuristic)
# ----------------------------
def _greedy_small_infeasible_subset(
    ann: MarkedAnnulus,
    candidates: List[WeightedCandidate],
    *,
    seed: Optional[int] = None,
    max_grow: int = 512,
    max_shrink_passes: int = 4,
    verbose: bool = False,
) -> Tuple[List[WeightedCandidate], RealFeasibilityResult]:
    """
    Heuristically shrink a *complete* candidate set, with multi-restart greedy and
    optional exact minimisation when the set is small.

    Key properties:
      - Deduplicates by dx signature: identical dx equations are redundant constraints.
      - Runs greedy several times (random restarts) and keeps the smallest subset found.
      - If the best subset is small enough, runs an exact minimum-cardinality search
        (iterative deepening) with strict caps on feasibility checks.

    This does NOT change candidate generation or completeness logic; it only chooses
    a smaller subset of already-found candidates.
    """

    # ----------------------------
    # Tunables (safe defaults)
    # ----------------------------
    GREEDY_RESTARTS = 24                    # how many greedy restarts to try
    EXACT_TRIGGER_MAX_SIZE = 22             # only attempt exact min if best <= this
    EXACT_MAX_M = 18                        # exact search only if m <= this
    EXACT_MAX_FEAS_CHECKS = 6000            # hard cap on feasibility checks in exact phase

    if not candidates:
        return [], RealFeasibilityResult(feasible=True, witness_w=None, status="no candidates")

    # Deduplicate by dx signature up-front (keep first occurrence).
    # Identical dx equations impose identical constraints dx(w)=0 and are redundant.
    by_dx: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], WeightedCandidate] = {}
    for cand in candidates:
        sig = (cand.dx.const, cand.dx.coeffs)
        if sig not in by_dx:
            by_dx[sig] = cand
    uniq = list(by_dx.values())

    # If dx-deduplication collapsed to empty, fall back safely.
    if not uniq:
        complete_all, feas_all = is_complete_candidate_set_over_reals(ann, candidates)
        return candidates, feas_all

    base_rng = random.Random(seed)

    def _one_greedy_run(run_seed: int) -> Optional[Tuple[List[WeightedCandidate], RealFeasibilityResult]]:
        rng = random.Random(run_seed)
        perm = list(uniq)
        rng.shuffle(perm)

        # Grow phase: add candidates until infeasible.
        subset: List[WeightedCandidate] = []
        feas_now = RealFeasibilityResult(feasible=True, witness_w=None, status="unset")

        for cand in perm:
            subset.append(cand)
            complete, feas = is_complete_candidate_set_over_reals(ann, subset)
            if complete:
                feas_now = feas
                break
            if len(subset) >= max_grow:
                return None

        if not subset:
            return None
        complete_now, feas_now2 = is_complete_candidate_set_over_reals(ann, subset)
        if not complete_now:
            return None

        feas_now = feas_now2

        # Shrink phase: deletion passes.
        for _p in range(max_shrink_passes):
            if len(subset) <= 1:
                break
            idxs = list(range(len(subset)))
            rng.shuffle(idxs)
            removed_any = False
            for idx in idxs:
                trial = subset[:idx] + subset[idx + 1 :]
                complete_trial, feas_trial = is_complete_candidate_set_over_reals(ann, trial)
                if complete_trial:
                    subset = trial
                    feas_now = feas_trial
                    removed_any = True
                    break
            if not removed_any:
                break

        # Final safety check
        complete_final, feas_final = is_complete_candidate_set_over_reals(ann, subset)
        if not complete_final:
            return None
        return subset, feas_final

    # ----------------------------
    # Multi-restart greedy: keep best
    # ----------------------------
    best_subset: Optional[List[WeightedCandidate]] = None
    best_feas: Optional[RealFeasibilityResult] = None

    # Always include one deterministic run derived from 'seed' for reproducibility.
    seeds: List[int] = []
    if seed is None:
        seeds.append(base_rng.randint(0, 2**31 - 1))
    else:
        seeds.append(seed)

    for _ in range(GREEDY_RESTARTS - 1):
        seeds.append(base_rng.randint(0, 2**31 - 1))

    for j, s in enumerate(seeds, start=1):
        res = _one_greedy_run(s)
        if res is None:
            continue
        subset, feas_res = res
        if best_subset is None or len(subset) < len(best_subset):
            best_subset, best_feas = subset, feas_res
            if verbose:
                print(f"[search_shortcuts_weighted] minimiser: greedy best size={len(best_subset)} after {j}/{len(seeds)} restarts")
            # If we hit something trivially small, stop early.
            if len(best_subset) <= 3:
                break

    if best_subset is None or best_feas is None:
        # Fall back to original behaviour: return full set
        complete_all, feas_all = is_complete_candidate_set_over_reals(ann, candidates)
        if verbose:
            print("[search_shortcuts_weighted] minimiser: all greedy restarts failed; returning full set")
        return candidates, feas_all

    # ----------------------------
    # Optional exact minimisation (only if already small)
    # ----------------------------
    if len(best_subset) > EXACT_TRIGGER_MAX_SIZE:
        return best_subset, best_feas

    # Order candidates for exact search (heuristic: "stronger" constraints first)
    def _strength_key(c: WeightedCandidate) -> Tuple[int, int, int]:
        # More terms and larger coefficients first tends to find small infeasible subsets sooner.
        nnz = len(c.dx.coeffs)
        abs_sum = sum(abs(v) for _, v in c.dx.coeffs)
        cabs = abs(c.dx.const)
        return (nnz, abs_sum, cabs)

    exact_pool = sorted(best_subset, key=_strength_key, reverse=True)

    m = len(exact_pool)
    if m > EXACT_MAX_M:
        return best_subset, best_feas

    # Bitmask cache: mask -> infeasible? (True/False). Only for full feasibility checks.
    cache: Dict[int, bool] = {}
    feas_checks = 0

    def _is_infeasible_mask(mask: int) -> bool:
        nonlocal feas_checks
        if mask in cache:
            return cache[mask]
        subset = [exact_pool[i] for i in range(m) if (mask >> i) & 1]
        complete, _feas = is_complete_candidate_set_over_reals(ann, subset)
        infeas = bool(complete)
        cache[mask] = infeas
        feas_checks += 1
        return infeas

    # If somehow the pool isn't infeasible, return greedy best.
    full_mask = (1 << m) - 1
    if not _is_infeasible_mask(full_mask):
        return best_subset, best_feas

    best_mask = full_mask

    # Iterative deepening: find smallest k with an infeasible subset of size k.
    # Hard-stop if feasibility checks exceed cap.
    found_exact = False

    # Precompute indices to speed recursion
    idxs = list(range(m))

    def _dfs_find_k(k: int, start: int, chosen: List[int]) -> Optional[int]:
        nonlocal found_exact

        if feas_checks >= EXACT_MAX_FEAS_CHECKS:
            return None

        if len(chosen) == k:
            mask = 0
            for i in chosen:
                mask |= 1 << i
            if _is_infeasible_mask(mask):
                return mask
            return None

        remaining_slots = k - len(chosen)
        if (m - start) < remaining_slots:
            return None

        for i in range(start, m):
            chosen.append(i)
            out = _dfs_find_k(k, i + 1, chosen)
            chosen.pop()
            if out is not None:
                return out
            if feas_checks >= EXACT_MAX_FEAS_CHECKS:
                return None

        return None

    for k in range(1, m):
        if feas_checks >= EXACT_MAX_FEAS_CHECKS:
            break
        mask_k = _dfs_find_k(k, 0, [])
        if mask_k is not None:
            best_mask = mask_k
            found_exact = True
            if verbose:
                print(f"[search_shortcuts_weighted] minimiser: exact minimum found size={k} (feas_checks={feas_checks})")
            break

    if not found_exact:
        if verbose:
            print(f"[search_shortcuts_weighted] minimiser: exact phase did not improve (feas_checks={feas_checks})")
        return best_subset, best_feas

    exact_subset = [exact_pool[i] for i in range(m) if (best_mask >> i) & 1]

    # Safety check + return updated feasibility result
    complete_final, feas_final = is_complete_candidate_set_over_reals(ann, exact_subset)
    if not complete_final:
        # Should not happen; fall back
        return best_subset, best_feas

    return exact_subset, feas_final


# ----------------------------
# Wrapper: simple-first then weighted candidates
# ----------------------------

def find_complete_candidates_weighted(
    ann: MarkedAnnulus,
    *,
    try_simple_first: bool = True,
    simple_max_steps: Optional[int] = None,
    simple_max_nodes: int = 250_000,
    max_steps: Optional[int] = None,
    max_nodes: int = 250_000,
    max_candidates: int = 5_000,
    verbose: bool = False,
    max_pair_slots: int = 8,
    max_total_pair_slots: int = 64,
    return_simple_witness: bool = False,
    minimize_complete_set: bool = True,
    minimize_seed: Optional[int] = None,
    minimize_max_grow: int = 512,
    minimize_max_shrink_passes: int = 4,
) -> Tuple[
    List[WeightedCandidate],
    bool,
    RealFeasibilityResult,
    Optional[UnweightedShortcutResult],
]:
    if try_simple_first:
        simp = find_simple_shortcut(
            ann,
            max_steps=simple_max_steps,
            max_nodes=simple_max_nodes,
            verbose=verbose,
            max_pair_slots=max_pair_slots,
            max_total_pair_slots=max_total_pair_slots,
        )
        if simp is not None:
            feas = RealFeasibilityResult(
                feasible=False,
                witness_w=None,
                status="simple shortcut found (unweighted dx=lambda), therefore valid for all weights",
            )
            return [], True, feas, (simp if return_simple_witness else None)

    cands = find_candidates_weighted(
        ann,
        max_steps=max_steps,
        max_nodes=max_nodes,
        max_candidates=max_candidates,
        verbose=verbose,
        max_pair_slots=max_pair_slots,
        max_total_pair_slots=max_total_pair_slots,
    )
    complete, feas = is_complete_candidate_set_over_reals(ann, cands)

    # Heuristic minimisation: if the set is complete, shrink to a much smaller
    # infeasible subset (randomised greedy build + deletion passes).
    if complete and minimize_complete_set and cands:
        cands_small, feas_small = _greedy_small_infeasible_subset(
            ann,
            cands,
            seed=minimize_seed,
            max_grow=minimize_max_grow,
            max_shrink_passes=minimize_max_shrink_passes,
            verbose=verbose,
        )
        return cands_small, True, feas_small, None

    return cands, complete, feas, None

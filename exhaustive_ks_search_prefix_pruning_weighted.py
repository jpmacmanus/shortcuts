#!/usr/bin/env python3
"""
exhaustive_ks_search_prefix_pruning_weighted.py

(See previous version for full description.)

Reporting tweak:
- If no simple_witness is found but the weighted solver returns a COMPLETE set of size 1,
  we optionally "retcon" the display classification as SIMPLE to avoid confusing output.
  This does NOT modify any search logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Any, Dict

import sys
import time
import shutil
import re
import os

from klein_signature_generator import parse_klein_signature, build_surface_from_signature

from search_shortcuts_weighted import find_complete_candidates_weighted
from ascii_pretty import render_trackstate, RenderConfig

from annulus import Side
from chord_diagram import BoundaryPoint, PortKind
from track_state import initial_state


# ----------------------------
# Configuration
# ----------------------------

SURFACE_TYPE = "annulus"
KLEIN_SIGNATURE = "I V R I"

MAX_STEPS: Optional[int] = None
MAX_NODES: int = 20000
MAX_CANDIDATES: int = 5000

SIMPLE_MAX_STEPS: Optional[int] = None
SIMPLE_MAX_NODES: int = 2000

SIMPLE_ONLY: bool = False

START_PREFIX_LENGTH: int = 2
STOP_EARLY_IF_BASKET_EMPTY: bool = True

DEDUPLICATE_BY_SURFACE_SYMMETRY: bool = True

PRETTY_PRINT_SUCCESS: bool = True
PRETTY_PRINT_FAILURE_BLANK: bool = True
PRETTY_CFG = RenderConfig(width=22, height=11, pad_between=3)

ENABLE_COLOUR: bool = True

LIVE_UI: bool = True
LIVE_UI_REFRESH_EVERY: int = 1
LIVE_UI_MIN_REFRESH_SECONDS: float = 0.03
LIVE_UI_SHOW_PRETTY: bool = True

PRINT_PROGRESS: bool = True
PRINT_EACH_BAD_CASE: bool = True
PRINT_EACH_SUCCESS: bool = True
PRINT_DX_EQUATIONS_ON_TERMINAL: bool = False

WRITE_MARKDOWN_LOG: bool = True
MARKDOWN_LOG_PATH: Optional[str] = None

# New: reporting policy for singleton-complete
# - If True: always label singleton-complete as SIMPLE in output.
# - If False: only label it SIMPLE when its dx has no weight terms (coeffs empty).
RETCON_SINGLETON_AS_SIMPLE_ALWAYS: bool = True

OUTPUT_DIR = "solutions by Klein signature"


# ----------------------------
# ANSI colour + screen helpers
# ----------------------------

def _c(text: str, code: str) -> str:
    if not ENABLE_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(text: str) -> str:
    return _c(text, "92")


def _yellow(text: str) -> str:
    return _c(text, "93")


def _red(text: str) -> str:
    return _c(text, "91")


def _clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")


def _term_size() -> Tuple[int, int]:
    ts = shutil.get_terminal_size(fallback=(120, 40))
    return ts.columns, ts.lines


def _clamp_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def _progress_bar(done: int, total: int, width: int) -> str:
    if total <= 0:
        return "[" + (" " * max(0, width - 2)) + "]"
    done = max(0, min(done, total))
    frac = done / total
    fill = int(round(frac * max(0, width - 2)))
    return "[" + ("#" * fill) + ("-" * (max(0, width - 2) - fill)) + "]"


# ----------------------------
# Markdown logging
# ----------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _md_colour_span(kind: str, text: str) -> str:
    colour = {
        "good": "#1b7f2a",
        "warn": "#b38600",
        "bad":  "#b3261e",
        "info": "#0b57d0",
    }.get(kind, "#000000")
    return f'<span style="color: {colour}; font-weight: 600">{text}</span>'


def _sanitize_for_filename(s: str) -> str:
    compact = "".join(ch for ch in s if ch.isalnum())
    if compact:
        return compact
    s2 = re.sub(r"\s+", "_", s.strip())
    s2 = re.sub(r"[^A-Za-z0-9_\-]+", "", s2)
    return s2 or "sig"


def _format_dx_form(dx) -> str:
    terms = [str(dx.const)]
    for eid, coeff in dx.coeffs:
        terms.append(f"{coeff}*w{eid}")
    return " + ".join(terms)


def _format_dx_equation(dx) -> str:
    if not dx.coeffs:
        lhs = "0"
    else:
        lhs = " + ".join([f"{coeff}*w{eid}" for eid, coeff in dx.coeffs])
    rhs = str(-dx.const)
    return f"{lhs} = {rhs}"


@dataclass
class StageSummary:
    m: int
    total: int
    good: int
    bad: int
    good_simple: int
    good_weighted_only: int
    max_candidates_complete_weighted_only: int


@dataclass
class MarkdownLogger:
    final_path: str
    body_path: str
    body_fh: Any = field(init=False, repr=False)
    started: bool = field(default=False, init=False)

    stage_summaries: List[StageSummary] = field(default_factory=list)
    max_candidates_complete_weighted_only_global: int = 0

    def __post_init__(self) -> None:
        self.body_fh = open(self.body_path, "w", encoding="utf-8")

    def close(self) -> None:
        try:
            self.body_fh.close()
        except Exception:
            pass

    def write_body(self, s: str) -> None:
        self.body_fh.write(s)
        self.body_fh.flush()

    def start(self, *, surface: str, signature: str, N: int) -> None:
        self.started = True

    def stage_header(self, *, stage_m: int, sig_prefix: Sequence[str], basket_size: int, mode_line: str) -> None:
        self.write_body(f"\n\n## Stage m={stage_m}\n\n")
        self.write_body(f"- Prefix signature: `{list(sig_prefix)}`\n")
        self.write_body(f"- Basket size entering stage: `{basket_size}`\n")
        self.write_body(f"- {mode_line}\n\n")
        self.write_body("---\n")

    def perm_block(
        self,
        *,
        perm: Tuple[int, ...],
        verdict_line_ansi: str,
        pretty_block: str,
        equations_block: str = "",
    ) -> None:
        verdict_plain = _strip_ansi(verdict_line_ansi).strip()

        if verdict_plain.startswith("BAD"):
            verdict_md = _md_colour_span("bad", verdict_plain)
        elif "SIMPLE" in verdict_plain:
            verdict_md = _md_colour_span("good", verdict_plain)
        else:
            verdict_md = _md_colour_span("warn", verdict_plain)

        self.write_body(f"\n\n### perm={perm}\n\n{verdict_md}\n\n")

        if pretty_block:
            self.write_body("```text\n")
            self.write_body(pretty_block.rstrip("\n") + "\n")
            self.write_body("```\n\n")

        if equations_block:
            self.write_body("dx equations (in candidate order):\n\n")
            self.write_body("```text\n")
            self.write_body(equations_block.rstrip("\n") + "\n")
            self.write_body("```\n")

        self.write_body("\n---\n")

    def record_stage_summary(self, summ: StageSummary) -> None:
        self.stage_summaries.append(summ)
        if summ.max_candidates_complete_weighted_only > self.max_candidates_complete_weighted_only_global:
            self.max_candidates_complete_weighted_only_global = summ.max_candidates_complete_weighted_only

    def finalize_header(
        self,
        *,
        surface: str,
        signature: str,
        N: int,
        start_prefix: int,
        stopped_early: bool,
        final_m_reached: int,
        run_seconds: float,
    ) -> None:
        self.close()

        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lines: List[str] = []
        lines.append("# Weighted KS prefix-pruning run\n")
        lines.append(f"- Timestamp: `{ts}`")
        lines.append(f"- Surface: `{surface}`")
        lines.append(f"- Signature: `{signature}`")
        lines.append(f"- N: `{N}`")
        lines.append(f"- Start prefix length: `{start_prefix}`")
        lines.append(f"- MAX_STEPS: `{MAX_STEPS}`   MAX_NODES: `{MAX_NODES}`   MAX_CANDIDATES: `{MAX_CANDIDATES}`")
        lines.append(f"- SIMPLE_MAX_STEPS: `{SIMPLE_MAX_STEPS}`   SIMPLE_MAX_NODES: `{SIMPLE_MAX_NODES}`")
        lines.append(f"- DEDUPLICATE_BY_SURFACE_SYMMETRY: `{DEDUPLICATE_BY_SURFACE_SYMMETRY}`")
        lines.append(f"- SIMPLE_ONLY: `{SIMPLE_ONLY}`")
        lines.append(f"- Singleton retcon policy: `{RETCON_SINGLETON_AS_SIMPLE_ALWAYS}` (True=always simple; False=only if dx has no w-terms)")
        lines.append(f"- Stopped early: `{stopped_early}`")
        lines.append(f"- Max prefix length reached: `{final_m_reached}`")
        lines.append(f"- Wall time (seconds): `{run_seconds:.3f}`")
        lines.append("")

        lines.append("## Summary by stage\n")
        lines.append("| m | tested | GOOD total | GOOD simple | GOOD weighted-only | BAD | max candidates (weighted-only complete) |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        for s in self.stage_summaries:
            lines.append(
                f"| {s.m} | {s.total} | {s.good} | {s.good_simple} | {s.good_weighted_only} | {s.bad} | {s.max_candidates_complete_weighted_only} |"
            )
        lines.append("")
        lines.append(f"- Maximum candidates in any final complete weighted-only set (global): `{self.max_candidates_complete_weighted_only_global}`")
        lines.append("")

        header_text = "\n".join(lines).rstrip() + "\n\n---\n"

        with open(self.final_path, "w", encoding="utf-8") as out:
            out.write(header_text)
            with open(self.body_path, "r", encoding="utf-8") as body:
                shutil.copyfileobj(body, out)

        try:
            os.remove(self.body_path)
        except Exception:
            pass


# ----------------------------
# Symmetry canonicalisation on permutations
# ----------------------------

def _rotations(seq: Tuple[int, ...]) -> Iterable[Tuple[int, ...]]:
    n = len(seq)
    for k in range(n):
        yield seq[k:] + seq[:k]


def canonical_perm_under_surface_symmetry(perm: Tuple[int, ...], surface: str) -> Tuple[int, ...]:
    if not DEDUPLICATE_BY_SURFACE_SYMMETRY:
        return perm

    if surface == "strip":
        rev = tuple(reversed(perm))
        return min(perm, rev)

    if surface == "annulus":
        best = None
        for r in _rotations(perm):
            cand1 = r
            cand2 = tuple(reversed(r))
            if best is None or cand1 < best:
                best = cand1
            if cand2 < best:
                best = cand2
        return best  # type: ignore

    raise ValueError("surface must be 'annulus' or 'strip'")


# ----------------------------
# Prefix-pruning mechanics
# ----------------------------

def insert_label_everywhere(base: Tuple[int, ...], new_label: int) -> Iterable[Tuple[int, ...]]:
    m = len(base)
    for pos in range(m + 1):
        yield base[:pos] + (new_label,) + base[pos:]


def enumerate_stage_m_perms(m: int, surface: str) -> List[Tuple[int, ...]]:
    reps: Set[Tuple[int, ...]] = set()
    for perm in permutations(range(m)):
        reps.add(canonical_perm_under_surface_symmetry(tuple(perm), surface))
    return sorted(reps)


def blank_trackstate_for_surface(surface_obj) -> object:
    for i in range(surface_obj.N):
        if surface_obj.is_marked(surface_obj.edge(Side.TOP, i)):
            return initial_state(i, BoundaryPoint(PortKind.TOP, 1))
    for i in range(surface_obj.N):
        if surface_obj.is_marked(surface_obj.edge(Side.BOTTOM, i)):
            return initial_state(i, BoundaryPoint(PortKind.BOTTOM, 1))
    return initial_state(0, BoundaryPoint(PortKind.LEFT, 1))


# ----------------------------
# Stats + live UI state
# ----------------------------

@dataclass
class StageStats:
    tested: int = 0
    good: int = 0
    bad: int = 0
    good_simple: int = 0
    good_weighted_only: int = 0
    max_candidates_complete_weighted_only: int = 0


@dataclass
class LiveState:
    stage_m: int = 0
    stage_total: int = 0
    stage_sig: str = ""
    verdict_line: str = ""
    pretty_block: str = ""
    stats: StageStats = field(default_factory=StageStats)
    basket_size: int = 0
    mode_line: str = ""
    last_redraw_ts: float = 0.0


def _render_dashboard(ls: LiveState) -> str:
    cols, rows = _term_size()

    header: List[str] = []
    header.append(
        f"Surface type: {SURFACE_TYPE}    Signature: {KLEIN_SIGNATURE}    SIMPLE_ONLY={SIMPLE_ONLY}    DEDUP={DEDUPLICATE_BY_SURFACE_SYMMETRY}"
    )
    header.append(
        f"Bounds: MAX_STEPS={MAX_STEPS} MAX_NODES={MAX_NODES} MAX_CANDIDATES={MAX_CANDIDATES}    Simple: SIMPLE_MAX_STEPS={SIMPLE_MAX_STEPS} SIMPLE_MAX_NODES={SIMPLE_MAX_NODES}"
    )
    header.append("")

    bar_width = max(10, min(cols - 20, 60))
    bar = _progress_bar(ls.stats.tested, ls.stage_total, bar_width)
    pct = (100.0 * ls.stats.tested / ls.stage_total) if ls.stage_total else 0.0

    header.append(f"Stage m={ls.stage_m}  prefix={ls.stage_sig}")
    header.append(
        f"Progress: {bar}  {ls.stats.tested}/{ls.stage_total}  ({pct:5.1f}%)   basket_size={ls.basket_size}"
    )
    header.append(
        f"Counts: good={ls.stats.good} bad={ls.stats.bad}  (simple={ls.stats.good_simple} weighted_only={ls.stats.good_weighted_only})"
    )
    header.append(f"Max candidates (weighted-only complete) this stage: {ls.stats.max_candidates_complete_weighted_only}")
    header.append(ls.mode_line)
    header.append("")

    verdict = ls.verdict_line or ""
    if cols > 5 and len(verdict) > cols - 1:
        verdict = verdict[: cols - 4] + "..."
    header.append("Last example:")
    header.append(verdict)
    header.append("")

    used = len(header)
    remaining = max(0, rows - used - 1)
    pretty = ls.pretty_block if LIVE_UI_SHOW_PRETTY else ""
    pretty = _clamp_lines(pretty, remaining) if remaining > 0 else ""

    body = "\n".join(header)
    if pretty:
        body = body + pretty + "\n"
    return body


def _maybe_redraw(ls: LiveState, force: bool = False) -> None:
    if not LIVE_UI:
        return
    now = time.time()
    if not force and (now - ls.last_redraw_ts) < LIVE_UI_MIN_REFRESH_SECONDS:
        return
    _clear_screen()
    sys.stdout.write(_render_dashboard(ls))
    sys.stdout.flush()
    ls.last_redraw_ts = now


# ----------------------------
# Stage runner
# ----------------------------

def run_stage(
    sig_prefix: Sequence[str],
    candidates: Sequence[Tuple[int, ...]],
    surface: str,
    *,
    live: bool,
    stage_m: int,
    basket_size: int,
    md: Optional[MarkdownLogger] = None,
) -> Tuple[Set[Tuple[int, ...]], StageStats]:
    bad: Set[Tuple[int, ...]] = set()
    stats = StageStats()

    ls = LiveState(
        stage_m=stage_m,
        stage_total=len(candidates),
        stage_sig=str(list(sig_prefix)),
        basket_size=basket_size,
        stats=stats,
        mode_line=("Mode: SIMPLE_ONLY (prefix pruning sound)" if SIMPLE_ONLY else "Mode: FULL WEIGHTED completeness (prefix pruning not asserted)"),
    )

    if live:
        _maybe_redraw(ls, force=True)

    if md is not None:
        md.stage_header(stage_m=stage_m, sig_prefix=sig_prefix, basket_size=basket_size, mode_line=ls.mode_line)

    for perm in candidates:
        stats.tested += 1

        surface_obj = build_surface_from_signature(sig_prefix, perm, surface=surface)

        cands, complete, feas, simple_witness = find_complete_candidates_weighted(
            surface_obj,
            try_simple_first=True,
            simple_max_steps=SIMPLE_MAX_STEPS,
            simple_max_nodes=SIMPLE_MAX_NODES,
            max_steps=MAX_STEPS,
            max_nodes=MAX_NODES,
            max_candidates=MAX_CANDIDATES,
            verbose=False,
            max_pair_slots=8,
            max_total_pair_slots=64,
            return_simple_witness=True,
        )

        has_simple = (simple_witness is not None)
        has_weighted_complete = bool(complete)

        # Reporting retcon for singleton-complete
        singleton_complete = (has_weighted_complete and (not has_simple) and len(cands) == 1)
        singleton_dx_has_no_w_terms = singleton_complete and (len(getattr(cands[0].dx, "coeffs", ())) == 0)
        treat_singleton_as_simple = singleton_complete and (
            RETCON_SINGLETON_AS_SIMPLE_ALWAYS or singleton_dx_has_no_w_terms
        )

        # Classification used for pruning logic remains unchanged.
        if SIMPLE_ONLY:
            is_good = has_simple
        else:
            is_good = has_weighted_complete

        pretty_block = ""
        equations_block = ""

        if LIVE_UI_SHOW_PRETTY or (not live):
            try:
                if is_good and PRETTY_PRINT_SUCCESS:
                    if has_simple:
                        pretty_block = render_trackstate(simple_witness.st, surface_obj, cfg=PRETTY_CFG)  # type: ignore
                    else:
                        if cands:
                            rendered: List[str] = []
                            eqs: List[str] = []
                            for idx, cand in enumerate(cands, start=1):
                                rendered.append(f"Candidate {idx}/{len(cands)}")
                                rendered.append(render_trackstate(cand.st, surface_obj, cfg=PRETTY_CFG))
                                rendered.append("")
                                eqs.append(f"[{idx}] dx = {_format_dx_form(cand.dx)}")
                                eqs.append(f"    dx(w)=0  <=>  {_format_dx_equation(cand.dx)}")
                                eqs.append("")
                            pretty_block = "\n".join(rendered).rstrip("\n")
                            equations_block = "\n".join(eqs).rstrip("\n")
                        else:
                            pretty_block = render_trackstate(blank_trackstate_for_surface(surface_obj), surface_obj, cfg=PRETTY_CFG)

                elif (not is_good) and PRETTY_PRINT_FAILURE_BLANK:
                    pretty_block = render_trackstate(blank_trackstate_for_surface(surface_obj), surface_obj, cfg=PRETTY_CFG)

            except Exception as e:
                pretty_block = f"(pretty print failed: {type(e).__name__}: {e})"
                equations_block = ""

        if not is_good:
            stats.bad += 1
            bad.add(perm)

            ls.verdict_line = _red(f"BAD perm={perm}   ({feas.status})")
            ls.pretty_block = pretty_block

            if live:
                if (stats.tested % LIVE_UI_REFRESH_EVERY) == 0:
                    _maybe_redraw(ls)
            else:
                if PRINT_EACH_BAD_CASE:
                    print(_red(f"  BAD perm={perm}"))
                    if pretty_block:
                        print(pretty_block)
                        print()

            if md is not None:
                md.perm_block(perm=perm, verdict_line_ansi=ls.verdict_line, pretty_block=pretty_block, equations_block="")
            continue

        stats.good += 1

        # Count “simple” category for summary purposes:
        if has_simple or treat_singleton_as_simple:
            stats.good_simple += 1
        else:
            stats.good_weighted_only += 1
            if len(cands) > stats.max_candidates_complete_weighted_only:
                stats.max_candidates_complete_weighted_only = len(cands)

        if has_simple:
            lam = getattr(simple_witness, "lam", None)
            dx = getattr(simple_witness, "dx", None)
            dy = getattr(simple_witness, "dy", None)
            ls.verdict_line = _green(
                f"GOOD perm={perm}   SIMPLE  lambda={lam} dx={dx} dy={dy} steps={len(simple_witness.word)}"  # type: ignore
            )
        elif treat_singleton_as_simple:
            # Retconned reporting only
            cand0 = cands[0]
            ls.verdict_line = _green(
                f"GOOD perm={perm}   SIMPLE (singleton-complete)  candidates=1  dx={_format_dx_form(cand0.dx)}  ({feas.status})"
            )
        else:
            ls.verdict_line = _yellow(
                f"GOOD perm={perm}   WEIGHTED completeness (no simple)  candidates={len(cands)}  ({feas.status})"
            )

        ls.pretty_block = pretty_block

        if live:
            if (stats.tested % LIVE_UI_REFRESH_EVERY) == 0:
                _maybe_redraw(ls)
        else:
            if PRINT_EACH_SUCCESS:
                print(ls.verdict_line)
            if pretty_block:
                print(pretty_block)
                print()
            if PRINT_DX_EQUATIONS_ON_TERMINAL and equations_block:
                print("dx equations (in candidate order):")
                print(equations_block)
                print()

        if md is not None:
            md.perm_block(
                perm=perm,
                verdict_line_ansi=ls.verdict_line,
                pretty_block=pretty_block,
                equations_block=equations_block,
            )

    if live:
        _maybe_redraw(ls, force=True)

    return bad, stats


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    sig_full = parse_klein_signature(KLEIN_SIGNATURE)
    N = len(sig_full)

    md_logger: Optional[MarkdownLogger] = None

    if WRITE_MARKDOWN_LOG:
        path = MARKDOWN_LOG_PATH
        if path is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            sig_token = _sanitize_for_filename(KLEIN_SIGNATURE)

            # Ensure output directory exists
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            filename = f"results_{SURFACE_TYPE}_{sig_token}_{ts}.md"
            path = os.path.join(OUTPUT_DIR, filename)

        body_path = path + ".body.tmp"
        md_logger = MarkdownLogger(final_path=path, body_path=body_path)
        md_logger.start(surface=SURFACE_TYPE, signature=KLEIN_SIGNATURE, N=N)


    if SURFACE_TYPE not in ("annulus", "strip"):
        raise ValueError("SURFACE_TYPE must be 'annulus' or 'strip'")
    if START_PREFIX_LENGTH < 2 or START_PREFIX_LENGTH > N:
        raise ValueError("START_PREFIX_LENGTH must satisfy 2 <= START_PREFIX_LENGTH <= len(signature)")

    live = bool(LIVE_UI and sys.stdout.isatty())
    if not live and LIVE_UI:
        print("LIVE_UI requested, but stdout is not a TTY; falling back to scrolling output.\n")

    if live:
        _clear_screen()
        sys.stdout.flush()

    if (not live) and PRINT_PROGRESS:
        print(f"Surface type: {SURFACE_TYPE}")
        print(f"Signature length N: {N}")
        print(f"Signature: {KLEIN_SIGNATURE}")
        print(f"Prefix start length: {START_PREFIX_LENGTH}")
        print(f"DEDUPLICATE_BY_SURFACE_SYMMETRY: {DEDUPLICATE_BY_SURFACE_SYMMETRY}")
        print(f"SIMPLE_ONLY: {SIMPLE_ONLY}")
        print(f"Bounds: MAX_STEPS={MAX_STEPS}, MAX_NODES={MAX_NODES}, MAX_CANDIDATES={MAX_CANDIDATES}")
        print(f"Simple bounds: SIMPLE_MAX_STEPS={SIMPLE_MAX_STEPS}, SIMPLE_MAX_NODES={SIMPLE_MAX_NODES}")
        print()

    basket: Set[Tuple[int, ...]] = set()
    stopped_early = False
    final_m_reached = START_PREFIX_LENGTH
    t0 = time.time()

    try:
        m0 = START_PREFIX_LENGTH
        sig0 = sig_full[:m0]
        initial_candidates = enumerate_stage_m_perms(m0, SURFACE_TYPE)

        basket, stats = run_stage(sig0, initial_candidates, SURFACE_TYPE, live=live, stage_m=m0, basket_size=0, md=md_logger)
        final_m_reached = m0

        if md_logger is not None:
            md_logger.record_stage_summary(
                StageSummary(
                    m=m0,
                    total=stats.tested,
                    good=stats.good,
                    bad=stats.bad,
                    good_simple=stats.good_simple,
                    good_weighted_only=stats.good_weighted_only,
                    max_candidates_complete_weighted_only=stats.max_candidates_complete_weighted_only,
                )
            )

        if STOP_EARLY_IF_BASKET_EMPTY and not basket:
            stopped_early = True
            return

        for m in range(m0, N):
            new_label = m
            sig_prefix = sig_full[: m + 1]

            candidates_set: Set[Tuple[int, ...]] = set()
            for base in basket:
                for ext in insert_label_everywhere(base, new_label):
                    candidates_set.add(canonical_perm_under_surface_symmetry(ext, SURFACE_TYPE))
            candidates = sorted(candidates_set)

            basket, stats = run_stage(sig_prefix, candidates, SURFACE_TYPE, live=live, stage_m=m + 1, basket_size=len(basket), md=md_logger)
            final_m_reached = m + 1

            if md_logger is not None:
                md_logger.record_stage_summary(
                    StageSummary(
                        m=m + 1,
                        total=stats.tested,
                        good=stats.good,
                        bad=stats.bad,
                        good_simple=stats.good_simple,
                        good_weighted_only=stats.good_weighted_only,
                        max_candidates_complete_weighted_only=stats.max_candidates_complete_weighted_only,
                    )
                )

            if STOP_EARLY_IF_BASKET_EMPTY and not basket:
                stopped_early = True
                return

    finally:
        t1 = time.time()
        if md_logger is not None:
            md_logger.finalize_header(
                surface=SURFACE_TYPE,
                signature=KLEIN_SIGNATURE,
                N=N,
                start_prefix=START_PREFIX_LENGTH,
                stopped_early=stopped_early,
                final_m_reached=final_m_reached,
                run_seconds=(t1 - t0),
            )

    if live:
        sys.stdout.write(f"\nCompleted. Final basket size: {len(basket)}\n")
        sys.stdout.flush()
    else:
        if PRINT_PROGRESS:
            print("Completed.")
            print(f"Final basket size (bad cases at full length): {len(basket)}")


if __name__ == "__main__":
    main()

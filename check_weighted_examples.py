#!/usr/bin/env python3
"""
check_weighted_examples.py

Purpose
- Quickly test specific Klein-signature + permutation examples for WEIGHTED shortcut completeness.
- Provides a stable, non-scrolling terminal UI (ANSI dashboard) when run in a real terminal.
- Designed to be easy to modify: edit the CONFIG section and/or EXAMPLES list.

Input format (as you described)
- KS:   [I, R, V, H, I, R, V]
- Perm: (0, 2, 4, 6, 1, 3, 5)

What it does for each example
1) Build the surface from (KS, Perm) using build_surface_from_signature
2) Run weighted solver: find_complete_candidates_weighted(...)
   - tries simple shortcut first
   - otherwise enumerates candidates (closed tracks with dy!=0, lambda<N, final OR parity even)
   - certifies completeness by checking infeasibility of dx_i(w)=0 over w>=0 (reals)
3) Shows:
   - SIMPLE witness status (and pretty-print if enabled)
   - If no simple witness:
       * candidate count
       * completeness result and status
       * if incomplete: a feasible weight assignment witness (w) that kills all candidates' dx
   - optionally pretty-print K candidate tracks (first K)

Notes
- This script assumes you have:
    klein_signature_generator.py (parse/build functions)
    search_shortcuts_weighted.py (patched version)
    ascii_pretty.py
- No third-party dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import re
import sys
import time
import shutil

from klein_signature_generator import parse_klein_signature, build_surface_from_signature
from search_shortcuts_weighted import find_complete_candidates_weighted
from ascii_pretty import render_trackstate, RenderConfig


# ============================
# CONFIG (edit me)
# ============================

SURFACE_TYPE: str = "annulus"  # "annulus" or "strip"

# Weighted candidate search bounds (used only if simple shortcut not found)
MAX_STEPS: Optional[int] = None
MAX_NODES: int = 20000 # 2000 is not enough to see all solutions on length 6, so be prepared to increase this number.
MAX_CANDIDATES: int = 10_000

# Simple (unweighted dx=lambda) fast-path bounds
SIMPLE_MAX_STEPS: Optional[int] = None
SIMPLE_MAX_NODES: int = 2000

# Slot budgets (kept consistent with your unweighted search defaults)
MAX_PAIR_SLOTS: int = 8
MAX_TOTAL_PAIR_SLOTS: int = 64

# UI
LIVE_UI: bool = True  # non-scrolling dashboard in real terminal
ENABLE_COLOUR: bool = True
PAUSE_BETWEEN_EXAMPLES_SEC: float = 0.0  # set e.g. 0.5 to slow down
SHOW_PRETTY: bool = True
SHOW_PRETTY_FOR_SIMPLE: bool = True
SHOW_PRETTY_FOR_WEIGHTED_ONLY: bool = True
SHOW_PRETTY_FOR_INCOMPLETE: bool = True

# How many candidate tracks to print (when no simple witness exists)
SHOW_CANDIDATE_COUNT: int = 3  # set 0 to disable
# If True, also print candidate dx forms (const + coeffs) for displayed candidates
SHOW_CANDIDATE_DX_FORMS: bool = True

# Pretty settings
PRETTY_CFG = RenderConfig(width=26, height=12, pad_between=3)

# ============================
# EXAMPLES (edit me)
# ============================

EXAMPLES: List[Tuple[str, str]] = [
    ("[I, R, V, H, I, R, V]", "(0, 2, 4, 6, 1, 3, 5)"),
    # Add more:
    # ("[I, R, V, H, I]", "(0, 1, 3, 4, 2)"),
    # ("[I, R, V, H, I, R]", "(0, 2, 4, 1, 3, 5)"),
    # ("[I, R, V, H, I]", "(0, 2, 4, 1, 3)"),
    # ("[I, R, V, H]", "(0, 2, 1, 3)"),
    # ("[I, R, V]", "(0, 2, 1)"),
    # ("[I, R]", "(0, 1)"),
]


# ============================
# ANSI helpers
# ============================

def _isatty() -> bool:
    return bool(sys.stdout.isatty())


def _c(text: str, code: str) -> str:
    if not ENABLE_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(s: str) -> str:
    return _c(s, "92")


def yellow(s: str) -> str:
    return _c(s, "93")


def red(s: str) -> str:
    return _c(s, "91")


def dim(s: str) -> str:
    return _c(s, "90")


def bold(s: str) -> str:
    return _c(s, "1")


def clear_screen() -> None:
    sys.stdout.write("\033[2J\033[H")


def term_size() -> Tuple[int, int]:
    ts = shutil.get_terminal_size(fallback=(120, 40))
    return ts.columns, ts.lines


def clamp_lines(text: str, max_lines: int) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[:max_lines])


def hr(char: str = "─") -> str:
    cols, _ = term_size()
    return char * max(10, cols)


# ============================
# Parsing utilities
# ============================

def parse_ks_bracketed(s: str) -> str:
    """
    Accepts:
      "[I, R, V, H]"  -> "I R V H"
      "I R V H"       -> "I R V H"
      "IRVH"          -> "I R V H" (heuristic)
    Returns a string consumable by parse_klein_signature.
    """
    ss = s.strip()

    # bracketed list
    m = re.match(r"^\[\s*(.*?)\s*\]$", ss)
    if m:
        inner = m.group(1)
        toks = [t.strip() for t in inner.split(",") if t.strip()]
        return " ".join(toks)

    # spaced tokens already
    if " " in ss:
        return ss

    # compact (IRVH...)
    toks = list(ss)
    return " ".join(toks)


def parse_perm_tuple(s: str) -> Tuple[int, ...]:
    """
    Accepts:
      "(0, 2, 4)" or "0,2,4" or "[0, 2, 4]"
    """
    ss = s.strip()
    ss = ss.strip("()[]")
    if not ss:
        return tuple()
    parts = [p.strip() for p in ss.split(",") if p.strip()]
    return tuple(int(p) for p in parts)


# ============================
# Reporting
# ============================

@dataclass
class ExampleResult:
    ks_raw: str
    perm_raw: str
    ks_tokens: Sequence[str]
    perm: Tuple[int, ...]
    simple_found: bool
    complete: bool
    status: str
    candidate_count: int
    witness_w: Optional[Tuple[float, ...]]


def solve_one(ks_raw: str, perm_raw: str) -> Tuple[ExampleResult, Optional[str]]:
    ks_str = parse_ks_bracketed(ks_raw)
    ks_tokens = parse_klein_signature(ks_str)
    perm = parse_perm_tuple(perm_raw)

    surface_obj = build_surface_from_signature(ks_tokens, perm, surface=SURFACE_TYPE)

    candidates, complete, feas, simple_witness = find_complete_candidates_weighted(
        surface_obj,
        try_simple_first=True,
        simple_max_steps=SIMPLE_MAX_STEPS,
        simple_max_nodes=SIMPLE_MAX_NODES,
        max_steps=MAX_STEPS,
        max_nodes=MAX_NODES,
        max_candidates=MAX_CANDIDATES,
        verbose=False,
        max_pair_slots=MAX_PAIR_SLOTS,
        max_total_pair_slots=MAX_TOTAL_PAIR_SLOTS,
        return_simple_witness=True,
    )

    # Pretty rendering block
    pretty = ""
    if SHOW_PRETTY:
        blocks: List[str] = []
        if simple_witness is not None and SHOW_PRETTY_FOR_SIMPLE:
            blocks.append(bold("SIMPLE witness track"))
            blocks.append(render_trackstate(simple_witness.st, surface_obj, cfg=PRETTY_CFG))

        if simple_witness is None:
            if complete and SHOW_PRETTY_FOR_WEIGHTED_ONLY:
                blocks.append(bold("WEIGHTED-only certificate (sample candidate track shown)"))
                if candidates:
                    blocks.append(render_trackstate(candidates[0].st, surface_obj, cfg=PRETTY_CFG))
                else:
                    blocks.append(dim("(no candidates to render)"))

            if (not complete) and SHOW_PRETTY_FOR_INCOMPLETE:
                blocks.append(bold("INCOMPLETE (blank/sample)"))
                if candidates:
                    blocks.append(render_trackstate(candidates[0].st, surface_obj, cfg=PRETTY_CFG))
                else:
                    # Fall back to “blank surface” rendering by rendering the surface anchored on a marked edge.
                    # (ascii_pretty expects a TrackState; easiest is to just render the first candidate if exists.)
                    blocks.append(dim("(no candidates to render)"))

        if SHOW_CANDIDATE_COUNT > 0 and simple_witness is None and candidates:
            k = min(SHOW_CANDIDATE_COUNT, len(candidates))
            blocks.append(bold(f"First {k} candidate tracks"))
            for i in range(k):
                c = candidates[i]
                if SHOW_CANDIDATE_DX_FORMS:
                    blocks.append(dim(f"Candidate {i}: lam={c.lam} dy={c.dy} dx_const={c.dx.const} dx_coeffs={list(c.dx.coeffs)}"))
                blocks.append(render_trackstate(c.st, surface_obj, cfg=PRETTY_CFG))

        pretty = "\n\n".join(blocks)

    res = ExampleResult(
        ks_raw=ks_raw,
        perm_raw=perm_raw,
        ks_tokens=ks_tokens,
        perm=perm,
        simple_found=(simple_witness is not None),
        complete=bool(complete),
        status=feas.status,
        candidate_count=len(candidates),
        witness_w=feas.witness_w,
    )

    return res, pretty


def format_summary(res: ExampleResult) -> str:
    lines: List[str] = []
    lines.append(bold("Example"))
    lines.append(f"  KS:   {res.ks_raw}")
    lines.append(f"  Perm: {res.perm_raw}")
    lines.append("")

    if res.simple_found:
        lines.append(green("SIMPLE shortcut found (weight-robust by definition)."))
        lines.append(dim(f"Status: {res.status}"))
        return "\n".join(lines)

    lines.append(yellow("No SIMPLE shortcut found."))
    lines.append(f"Candidates found: {res.candidate_count}")

    if res.complete:
        lines.append(green("WEIGHTED completeness certified: for every w>=0, some candidate has dx(w) != 0."))
        lines.append(dim(f"Status: {res.status}"))
    else:
        lines.append(red("NOT complete: there exists w>=0 such that all candidates have dx(w)=0."))
        lines.append(dim(f"Status: {res.status}"))
        if res.witness_w is not None:
            # Print a truncated witness if very long
            w = res.witness_w
            if len(w) > 16:
                lines.append(f"Witness weights (prefix): {tuple(w[:16])} ... (len={len(w)})")
            else:
                lines.append(f"Witness weights: {w}")

    return "\n".join(lines)


def render_dashboard(index: int, total: int, summary: str, pretty: Optional[str]) -> str:
    cols, rows = term_size()

    hdr: List[str] = []
    hdr.append(bold(f"Weighted example checker  ({index}/{total})"))
    hdr.append(f"Surface={SURFACE_TYPE}   SIMPLE_MAX_NODES={SIMPLE_MAX_NODES}   MAX_NODES={MAX_NODES}   MAX_CANDIDATES={MAX_CANDIDATES}")
    hdr.append(hr())
    hdr.append(summary)
    hdr.append(hr())
    body = "\n".join(hdr)

    # Pretty area
    if pretty and SHOW_PRETTY:
        remaining = max(0, rows - body.count("\n") - 2)
        pretty2 = clamp_lines(pretty, remaining)
        body = body + "\n" + pretty2

    return body + "\n"


# ============================
# Main
# ============================

def main() -> None:
    live = bool(LIVE_UI and _isatty())
    if LIVE_UI and not live:
        print("LIVE_UI requested but stdout is not a TTY; using scrolling output.\n")

    total = len(EXAMPLES)
    for idx, (ks_raw, perm_raw) in enumerate(EXAMPLES, start=1):
        if live:
            clear_screen()
            sys.stdout.write(render_dashboard(idx, total, "Working... (running search)", None))
            sys.stdout.flush()

        res, pretty = solve_one(ks_raw, perm_raw)
        summary = format_summary(res)

        if live:
            clear_screen()
            sys.stdout.write(render_dashboard(idx, total, summary, pretty))
            sys.stdout.flush()
        else:
            print(render_dashboard(idx, total, summary, pretty))

        if PAUSE_BETWEEN_EXAMPLES_SEC > 0:
            time.sleep(PAUSE_BETWEEN_EXAMPLES_SEC)

    if live:
        sys.stdout.write("\nDone.\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()

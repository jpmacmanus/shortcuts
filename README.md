# Shortcuts (Dunwoody track search)

This project models marked strips/annuli built from squares and searches for
Dunwoody tracks (via chord diagrams) using a configurable BFS. It also generates
marked surfaces from Klein signatures and can certify completeness of candidate
simple shortcuts via linear infeasibility.

## Quick start

1) Configure the search in `exhaustive_search.py` (see the Config section).
2) Run:

```bash
python3 exhaustive_search.py
```

Output is ASCII diagrams and diagnostics for each surface. Example diagram:

```
TOP:                C↻                      D↻                                              C↻                      D↻           
        +---1------->-------0---+---5-@----->-------4---+-----------------------+---0-------<-------1---+---4-------<-------5---+
        |   *               *   |   *               *   |                       |   **              *   |   *               *   |
        |  *                 *  |    **              *  |                       |     **             *  |  *                 *  |
        | *                   * |      **            *  |                       |       **           *  | *                  *  |
        V *                   * |        **           * |                       |         **          * | *                   * V
        V*                     *|*         **          *|***********************|*          **         *|*                     *V
        V                       | *          **         |                       | *           **        |                       V
        |                       |  *           **       |                       | *             **      |                       |
        |                       |  *             **     |                       |  *              **    |                       |
        |                       |   *              **   |                       |   *               *   |                       |
        +-----------------------+---7------->-------8---+-----------A-----------+---7------->-------8---+-----------A-----------+
BOTTOM:                                     A                       B                       A                       B            
```

## Key modules

- `edge.py`: edges with ordered ports.
- `square.py`: oriented squares with top/bottom/left/right edges.
- `chords.py`: chords and crossing logic on a square.
- `chord_diagram.py`: square + chords with a pretty ASCII render.
- `strips.py`: linear strips and annuli (edge identification by shared edges).
- `marked_strips.py`: marked strips/annuli with edge identifications and ASCII render.
- `pattern.py`: chord diagrams across a surface with compatibility constraints.
- `track_state.py`: BFS state and move generation.
- `track_bfs.py`: search utilities and wrappers.
- `klein_signature_generator.py`: generate marked surfaces from Klein signatures.
- `shortcut_completeness.py`: LP-based completeness checks and minimization.
- `exhaustive_search.py`: main script and configuration.

## Configuration (exhaustive_search.py)

The main script is driven by constants near the top of the file:

- `SIGNATURE`, `SURFACE`, `UNIQUE`, `EXCLUDE_ADJACENT_I`, `PREFIX_PRUNING`:
  generation controls.
- `REQUIRE_DY_NONZERO`, `LIMIT_INTERIOR_CROSSINGS`, `REQUIRE_DX_INFEASIBLE`:
  search constraints.
- `REQUIRE_EVEN_TURNING`, `REQUIRE_EVEN_OR_PAIRS`:
  parity constraints.
- `MAX_NODES`, `MAX_CANDIDATES`, `MAX_PORTS_PER_EDGE`:
  search bounds.
- `SHOW_PROGRESS`, `PROGRESS_INTERVAL`:
  progress output.
- `DEBUG_UNSOLVED`, `DEBUG_TRACE_BFS`, `DEBUG_BFS_STEPS`:
  debug output (can be noisy).

## Dependencies

- Python 3.11+ (tested with 3.11/3.12).
- `scipy` is required for linear infeasibility checks in
  `shortcut_completeness.py`.

Install:

```bash
python3 -m pip install scipy
```

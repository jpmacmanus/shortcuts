# Shortcuts (Dunwoody Track Search)

This project is a computer search for Dunwoody tracks (via chord diagrams) using a configurable BFS, on specific types of 'marked surfaces' constructed in a niche way.

## Quick start

1) Open `exhaustive_search.py` and edit the config block near the top.
2) Run:

```bash
python3 exhaustive_search.py
```

You’ll see an ASCII diagram for each surface, search progress, and a result
summary.

## Configuration

All configuration is done via constants at the top of `exhaustive_search.py`.

### Search input
- `SIGNATURE`: Klein signature string (e.g. `"I R V H I R"`).
- `SURFACE`: `"annulus"` or `"strip"`.
- `DIRECTED_MODE`: `True` to use directed surfaces, `False` for undirected.

### Generation / symmetry controls
- `UNIQUE`: if `True`, deduplicate by symmetry:
  - undirected annulus: dihedral symmetry
  - undirected strip: reflection symmetry
  - directed annulus: cyclic symmetry
  - directed strip: reflection symmetry
- `EXCLUDE_ADJACENT_I`: filter permutations with adjacent `I` entries (ignored in directed mode).
- `PREFIX_PRUNING`: enable the prefix-pruning search strategy.
- `START_PREFIX_LENGTH`: initial prefix length for prefix pruning.

### Acceptance constraints
These toggle which closed tracks are accepted as solutions.
- `REQUIRE_DY_NONZERO`
- `REQUIRE_DX_INFEASIBLE`: when `True`, may return 'complete candidate sets' rather than a single solution.
- `REQUIRE_EVEN_TURNING`
- `REQUIRE_EVEN_OR_PAIRS`
- `DOMINANT_DIR_ONLY`: only move along dominant x-direction (free first horizontal move).
- `LIMIT_INTERIOR_CROSSINGS`: cap interior crossings to one.
- `REJECT_ALL_INTERIOR_USED`: reject solutions that use every interior edge.

### BFS bounds / minimization
- `MAX_NODES`
- `MAX_CANDIDATES`
- `MAX_PORTS_PER_EDGE`
- `MINIMIZE_SEED`
- `MAX_MINIMIZE_SIZE`

### Debug / output
- `SHOW_PROGRESS`, `PROGRESS_INTERVAL`
- `DEBUG_UNSOLVED`, `DEBUG_UNSOLVED_MAX`
- `DEBUG_TRACE_BFS`, `DEBUG_TRACE_BFS_MAX`
- `DEBUG_BFS_STEPS`, `DEBUG_BFS_STEPS_MAX`

## Output

Each case prints in three phases:

1) **Header + diagram** (blue)
2) **Progress bar** while the BFS runs
3) **Result** (solution, complete candidate set, or unsolved)

At the end, a diagnostics table summarizes results per stage and totals.

### Example output

```
Annulus [25 / 36]
MarkedAnnulus(N=6)
                    C↻                      D↻                      E↻                      C↻                      D↻                      E↻
        +----------->-----------+----------->-----------+----------->-----------+-----------<-----------+-----------<-----------+-----------<-----------+
        |                       |                       |                       |                       |                       |                       |
        |                       |                       |                       |                       |                       |                       |
        |                       |                       |                       |                       |                       |                       |
        V                       |                       |                       |                       |                       |                       V
        V                       |                       |                       |                       |                       |                       V
        V                       |                       |                       |                       |                       |                       V
        |                       |                       |                       |                       |                       |                       |
        |                       |                       |                       |                       |                       |                       |
        |                       |                       |                       |                       |                       |                       |
        +-----------------------+----------->-----------+----------->-----------+----------->-----------+----------->-----------+-----------------------+
                                            A                       B                       A                       B
start 1/6 [################--------] 3400/5000 queue 28 seen 33999
start 2/6 [########################] 5000/5000 queue 639 seen 4999
start 3/6 [#######################-] 4800/5000 queue 26 seen 47999
start 4/6 [#######################-] 4800/5000 queue 27 seen 47999
start 5/6 [########################] 5000/5000 queue 639 seen 4999
start 6/6 [################--------] 3400/5000 queue 28 seen 33999
Complete candidate set found (size 2).
  [1] dy: -4  dx: -x_3 + x_4 + x_5 - x_6
                    C↻                      D↻                      E↻                      C↻                      D↻                      E↻
        +-----------0-----------+---5-------4-------3---+---B------->-------A---+-----------0-----------+---3-------4-------5---+---A-------<-------B---+
        |          **           |   *       *       *   |   *               *   |           *           |   *       *       *   |   *               *   |
        |       ***             |   *       *       *   |   *              *    |           *           |    *       *       *  |  *                 *  |
        |     **                |   *       *       *   |   *             *     |           *           |     *       *       * |  *                  * |
        V  ***                  |   *       *       *   |   *            *      |           *           |      *       *      * | *                   * V
        V**                     |   *       *       *   |   *           *      *|*          *          *|*      *       *      *|*                     *V
        V                       |   *       *       *   |   *          *      * | *         *         * | *      *       *      |                       V
        |                       |   *       *       *   |   *         *       * |  *        *         * |  *      *       *     |                       |
        |                       |   *       *       *   |   *        *       *  |  *        *        *  |  *       *       *    |                       |
        |                       |   *       *       *   |   *       *       *   |   *       *       *   |   *       *       *   |                       |
        +-----------------------+---7-------8-------9---+---D-------E-------F---+---7-------8-------9---+---D-------E-------F---+-----------------------+
                                            A                       B                       A                       B

  [2] dy: 4  dx: -2 - x_2 - x_3 + x_4 - x_6
                    C↻                      D↻                      E↻                      C↻                      D↻                      E↻
        +-----------0-----------+---4------->-------3---+-----------7---@-------+-----------0-----------+---3-------<-------4---+-----------7-----------+
        |          **           |   *               *   |           **          |           **          |   *               *   |           **          |
        |       ***             |    *               *  |             ***       |             ***       |  *               *    |             ***       |
        |     **                |     *              *  |                **     |                **     | *               *     |                **     |
        V  ***                  |      *              * |                  ***  |                  ***  | *              *      |                  ***  V
        V**                     |       *              *|**                   **|**                   **|*              *       |                     **V
        V                       |        *              |  ***                  |  ***                  |              *        |                       V
        |                       |         *             |     **                |     **                |             *         |                       |
        |                       |          *            |       ***             |       ***             |            *          |                       |
        |                       |           *           |          **           |          **           |           *           |                       |
        +-----------------------+-----------6-----------+-----------9-----------+-----------6-----------+-----------9-----------+-----------------------+
                                            A                       B                       A                       B
```

### Reading the diagrams
- Each row of squares is rendered as a contiguous ASCII strip.
- Marked edge labels appear above and below the strip.
- Direction arrows appear near the right end of edges in directed mode.
- OP/OR markers (`>`/`<`) appear along marked edges.

### Result messages
If a simple shortcut is found, the track diagram is printed with its `dy` and
linear `dx` expression. If no shortcut is found and `REQUIRE_DX_INFEASIBLE` is
enabled, a minimized complete candidate set is printed instead.

## Key modules

- `edge.py`: edges with ordered ports
- `square.py`: oriented squares with top/bottom/left/right edges
- `chords.py`: chords and crossing logic
- `chord_diagram.py`: square + chords with ASCII render
- `strips.py`: strips/annuli and adjacency
- `marked_strips.py`: marked strips/annuli with identifications and render
- `directed_marked_strips.py`: directed variants with per-edge in/out/undirected
- `pattern.py`: chord diagrams across a surface
- `track_state.py`: BFS state and move generation
- `track_bfs.py`: BFS search utilities and wrappers
- `klein_signature_generator.py`: undirected surface generation
- `directed_klein_signature_generator.py`: directed surface generation
- `shortcut_completeness.py`: LP-based completeness checks
- `exhaustive_search.py`: main executable script

## Dependencies

- Python 3.11+
- `scipy` (for linear infeasibility checks in `shortcut_completeness.py`)

Install:

```bash
python3 -m pip install scipy
```

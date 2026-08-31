# Analysis notebooks

Flat, numbered from `00` in run order. **The analysis lives here.** A notebook calls functions from
`src/` to *build* its results, then visualises them — it does not read its own results back from a
previous run.

The one thing a notebook reads from disk is an **earlier stage's** output. Notebook 05 reads the
coherent-GGI series that 04 built and the validation tables that 02 built; it builds its own
comparison tables itself. Reading your own output back is how a figure ends up disagreeing with the
number beside it.

Each notebook writes to a folder named after itself:
`outputs/tables/<notebook>/` and `outputs/fig/<notebook>/`, via `params.table_dir(NOTEBOOK)` /
`params.fig_dir(NOTEBOOK)`, where `NOTEBOOK` is set in the setup cell.

| Notebook | Builds | Reads from | State |
| --- | --- | --- | --- |
| `00_data_creation.ipynb` | UN population panel | the R/Quarto survey chain | **runs** |
| `01_fit_final_models.ipynb` | the six production models, LOCO, error betas | the fitting panel | **runs** |
| `02_model_performance.ipynb` | LOCO for all ten variants, pickle verification, survey validation | 04's series | **runs** |
| `03_unseen_survey_validation.ipynb` | figures on the 17 unseen surveys | 02 | **runs** |
| `04_coherent_ggi_figures.ipynb` | the coherent GGI series and its figures | the golden series | **runs** |
| `05_coherent_ggi_comparison.ipynb` | coherent-vs-direct tables and figures; **sets the 0.07 discrepancy threshold** (D43) | 04, 02 | **runs** |
| `06_coherent_ggi_decomposition.ipynb` | the exact decomposition, on the filtered series (D43) | 04 | **runs** |
| `07_aggregation_methods.ipynb` | the aggregation audit | 04 | **runs** |
| `08_prediction_and_convergence.ipynb` | prediction maps, beta/sigma convergence — calls `convergence` | 04's annual panel | **runs** (D47, D48, D49) |
| `09_direct_ggi_trend_figures.ipynb` | the trend figures on the **directly predicted** GGI — `trend.qmd` ported, with its uncertainty ribbons and event annotations; national-vs-subnational; ground-truth coverage | the golden series, the subnational aggregate, the survey ground truth | **runs** (D51) |
| `10`, `11` adolescent | a separate analysis line | — | not part of the national pipeline |
| `12_specification_stability.ipynb` | `robustipy` sweep, each spec variable as the focal X; LOCO for all 127 distinct specs | the fitting panel | **runs** (D39, E1, E2) |

Four exploration notebooks were archived to `src/archive/` on 2026-08-31, prefixes dropped (D51):
`12_s1_investigations`, `13_extended_panel_exploration`, `15_experimental_model_design` and
`16_experimental_model_design_new_parameters`. No part of the paper rests on them. The specification
sweep that survived was renumbered `14` → **`12`**, and its `outputs/fig/` and `outputs/tables/`
folders were renamed with it, so a figure's path still names the step that made it. `doc/` entries
written under the old numbering keep it; `src/archive/README.md` and the note at the top of
`doc/decisions_exploration.md` resolve both.

The R and Quarto data-creation stages are **not** notebooks — they read DHS/MICS microdata with
`haven`. They are in `src/data_creation/`, numbered `00`–`04`; see `doc/workflow.md`.

## Setup cell

No `_bootstrap` file. Each notebook opens with:

```python
import sys
from pathlib import Path

SRC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'params.py').exists())
sys.path.insert(0, str(SRC))
```

which works whether Jupyter was launched in the notebook folder or the project root.

## Style

**One scheme, one file.** The palette, the ink, the type ladder and the figure sizes live in
[`src/shared_style.py`](../shared_style.py) (D52, D53). `params.py` re-exports them so
`params.colors` and `params.STYLE` keep working, and `src/plotting.py` is the code that applies
them — `plot.use_style()` runs on import, so the font family and the default sizes are already set
before a notebook draws anything. Edit `shared_style.py` and nowhere else.

**Do not write a hex, a figure size or a font size in a notebook.** None of them does. The checks,
over the code cells, should all return 0:

    grep -c '#[0-9a-fA-F]\{6\}'        # hexes
    grep -c 'figsize=('                 # figure sizes — use plot.figsize(...)
    grep -c 'fontsize=[0-9]'            # font sizes  — use STYLE['..._fs']

### The palette

Six steps off ColorBrewer's RdYlBu — a warm end, a cool end, and the pale midpoint between them
(D53):

| slot | name | hex | | slot | name | hex |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | `red` | `#D73027` | | 3 | `cream` | `#FEE090` |
| 1 | `navy` | `#313695` | | 4 | `steel_blue` | `#4575B4` |
| 2 | `light_blue` | `#ABD9E9` | | 5 | `blue` | `#74ADD1` |

`plot_color_order` is the assignment order above: the ramp's two ends first, then ordered so each
additional series costs as little separation as possible. **Four or fewer series is where this
palette is comfortable** — the worst pair among the first four slots is 29.3 dE2000, the fifth
brings it to 17.6 and the sixth to 13.1, because the cool end is four steps of one hue.
**Six is the cap** — `plot.categorical()` raises past it rather than cycling. Fold the tail into
"Other", use small multiples, or pass `extend=True` and accept weaker separation.

When the variable is *paired* rather than a flat list — women/men, observed/predicted,
internet/mobile — use `plot.FAMILIES` instead (`'warm'` is cream→red, `'blue'` is
light_blue→blue→steel_blue→navy): one family per side, one step per level. That is the reading the
scheme is built for.

`cream` is the ramp's midpoint and is pale: put it on a **fill**, not on a hairline. A 0.7 pt cream
line disappears against the page.

### Figure size and type

Both are named, and a notebook asks for them by name:

| want | use |
| --- | --- |
| a figure size | `plot.figsize('wide')` — `single`, `square`, `wide`, `full`, `tall`, `banner`, `poster` |
| a grid whose panel count is data-dependent | `plot.figsize('full', nrow=n)`, or `plot.figsize(nrow=n, ncol=k)` |
| a stack of full-width panels | `plot.figsize('full', nrow=n, row_h=STYLE['panel_h'])` |
| a font size | `STYLE['title_fs']` 16 · `['label_fs']` 13 · `['tick_fs']` 11 · `['annot_fs']` 10 · `['small_fs']` 9 |

The ladder has five steps and nothing between them — `annot_fs` is the value label on a mark,
`small_fs` is the floor. Never write `tick_fs - 1`.

### What to reach for

| want | use | not |
| --- | --- | --- |
| a figure size | `plot.figsize(name)` | `figsize=(14, 5.4)` |
| a font size | `STYLE['annot_fs']` and the four other steps | `fontsize=8`, `tick_fs - 1` |
| a series, a category | `PALETTE[i]`, or `plot.categorical(values)` for a name → hue map | a hand-picked hex |
| the two sides of a pair | `plot.FAMILIES['blue']` / `['warm']` | two arbitrary slots |
| ordered bands (HDI, year, quartile) | `plot.ordinal(n)` | `n` categorical hues, which assert the bands are unordered |
| more categories than slots | `plot.spread(n)`, or fold the tail into "Other" | cycling the palette |
| a choropleth with a natural zero | `plot.sequential()` | `viridis`, `mako`, a hand-rolled ramp |
| a choropleth read against parity | `plot.diverging()` + `TwoSlopeNorm(vcenter=1.0)` | a ramp whose midpoint drifts with the sample |
| the outline on any filled mark | `plot.EDGE` (black) | `'white'`, `'black'` |
| frame, labels, gridlines | `plot.INK`, `plot.MUTED`, `plot.GRID` (`plot.tidy()` applies all three) | `'black'`, `'grey'`, `'lightgrey'` |
| "this value is degenerate" | `plot.FLAG` | `'red'` |
| "not one of the coloured classes" | `plot.NEUTRAL` | a desaturated hue that reads as an accident |
| map ground where nothing was predicted | `plot.NODATA` | `'grey'` |

`plot.categorical()` sorts before it assigns. Zipping `PALETTE` against `Series.unique()` does not —
`unique()` returns *data* order, so the same region takes a different hue in each figure, and a
reader who carries a colour between two panels is misled. That bug was live in `10` until
2026-08-31.

### Saving

**Every figure is written as a PDF and a PNG**, in that order — `shared_style.FORMATS`. PDF is the
one that matters: it is vector, so a figure scaled into the manuscript stays sharp. Saving is **on
by default** (`STYLE['save'] = True`); set it to `False` in `shared_style.py` for a run that should
not touch disk.

```python
plot.save(fig, NOTEBOOK, 'name')        # -> outputs/fig/<NOTEBOOK>/name.{pdf,png}
plot.save_fig(fig, some_path)           # any path; the suffix is replaced per format
```

Do not call `fig.savefig` directly — it writes one format, and which one depends on the suffix
whoever wrote the line happened to type.

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
| `09_post_analysis_trend.ipynb` | trend comparison | — | partial — missing model folder |
| `10`, `11` adolescent | a separate analysis line | — | not part of the national pipeline |
| `12_s1_investigations.ipynb` | supplementary checks | — | partial |
| `13_extended_panel_exploration.ipynb` | the panel plus the new surveys, refit and re-validated; **§4 the new surveys under both models**, **§5 the `gdp_pcap` vintage test** | the fitting panel, the survey ground truth, FB, offline predictors, WDI API | **runs** — exploration only (D39, D44, D45) |
| `14_specification_stability.ipynb` | `robustipy` sweep, each spec variable as the focal X; LOCO for all 127 distinct specs | the fitting panel | **runs** (D39, E1, E2) |
| `15_experimental_model_design.ipynb` | a five-term spec (`fb_18_999_r` + the offline four) scored against production, then swept | the fitting panel | **runs** — exploration only (E3-E5) |
| `16_experimental_model_design_new_parameters.ipynb` | drops `gdp_pcap` only, then sweeps all 46 development indicators as substitutes and as additions | the fitting panel | **runs** — exploration only (E6) |

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

Charts read `params.STYLE` through `src/plotting.py` — `plot.tidy()`, `plot.legend()`,
`plot.save()`, `plot.PALETTE`, `plot.grouped_bars()`, `plot.beeswarm()`. Do not redefine axis
treatment or colours in a notebook.

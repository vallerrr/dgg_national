# Notebooks

One folder per pipeline stage, numbered in run order. **The analysis code is here; the functions it
calls are in `src/`** (CONVENTIONS.md §7) — a notebook chooses inputs, calls a function, and shows
the result. It should not define the analysis.

The same numbering names the outputs: a notebook in `03_model_performance/` writes to
`outputs/tables/03_model_performance/` and `outputs/fig/03_model_performance/`, via
`params.table_dir()` / `params.fig_dir()`.

| Stage | Notebook | Calls | State |
| --- | --- | --- | --- |
| 01 data creation | `01_01_data_creation.ipynb` | `population` | **runs** |
| 02 model fitting | `02_01_fit_final_models.ipynb` | `modelling` | **runs** |
| 03 model performance | `03_01_model_performance.ipynb` | `model_performance` | **runs** — reproduces the published performance figure |
| | `03_02_unseen_survey_validation.ipynb` | `model_performance` tables | **runs** |
| 04 coherent GGI | `04_01_coherent_ggi_figures.ipynb` | `coherent_ggi` | **runs** |
| | `04_02_coherent_ggi_comparison.ipynb` | `ggi_comparison`, `model_performance` | **runs** |
| | `04_03_coherent_ggi_decomposition.ipynb` | `ggi_decomposition` | blocked — the Natural Earth shapefiles are 0-byte Dropbox placeholders |
| | `04_04_aggregation_methods.ipynb` | `aggregation` | **runs** |
| 05 trend analysis | `05_01_prediction_and_convergence.ipynb` | — | imported (D20) |
| | `05_02_post_analysis_trend.ipynb` | — | partial — missing model folder |
| `adolescent/` | two notebooks | — | a separate analysis line, not part of the national pipeline |
| `supplementary/` | `s1_investigations.ipynb` | — | partial |
| `technical_report/` | `*.qmd` | — | Quarto |

The R and Quarto stages of data creation are **not** notebooks — they read DHS/MICS microdata with
`haven`. They live in `src/data_creation/` and keep their own numbering; see `doc/workflow.md`.

## Import convention

Jupyter puts only the notebook's own folder on `sys.path`, so every folder carries a copy of
`_bootstrap.py`, which walks up to the directory holding `params.py`:

```python
import _bootstrap  # noqa: F401  — must precede the params/utils imports
import params
import plotting as plot
```

No absolute paths: everything resolves through `params`.

## Style

Charts read their styling from `params.STYLE` through `src/plotting.py` — `plot.tidy()`,
`plot.legend()`, `plot.save()`, `plot.PALETTE`. Do not redefine axis treatment or colours in a
notebook; that is how figures drift apart.

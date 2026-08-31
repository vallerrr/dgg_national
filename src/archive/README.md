# Archive

Retired from the active pipeline. Kept for provenance, not maintained.

**Most of it does not run**: the random-forest line and the superseded scripts each depend on an
input that no longer exists on Dropbox, on fitted pickles that were never re-created, or on the
91-variable Facebook set the upstream pipeline stopped producing. The four notebooks under
*Exploration outside the paper* are the exception — those still run, and are archived on purpose
rather than by decay.

Paths and imports were repaired during the 2026-08 reorganisation, so these are readable and
structurally sound — they are archived because their *purpose* is served, not because they are
beyond repair.

**Numeric prefixes are dropped here**, notebooks included, so nothing in this folder competes with
the active `src/` and `src/analysis_notebooks/` numbering (D51). Names are otherwise unchanged; the
`Was` column resolves any cross-reference written under the old numbering.

## The random-forest line (D13, D51)

The 2024 modelling exercise fitted random forests alongside OLS and built a set of notebooks on top
of them. D13 retired the method on 2026-08-03; D51 finished the job on 2026-08-31 by archiving the
notebooks that still depended on it, so **no random forest is fitted, loaded or predicted from
anywhere in the active tree**.

What that leaves behind, deliberately:

- `outputs/models/18_plus/` — the fitted RF pickles, kept for provenance. Nothing reads them.
  `params.model_folder`, the global that used to select that directory, is gone; the production
  folder is `params.FINAL_MODEL['model_folder']` (`OLS`).
- Every RF site here hardcodes `RandomForestRegressor(random_state=42)` — six instantiations across
  three files, checked. So the RF *was* reproducible; what it did not do is read `params.SEED`
  (§9 wants one seed constant, not a literal repeated at each call site). `doc/methodology.md` said
  the opposite until 2026-08-31; the correction is in D51.

| File | Was | Retired because |
| --- | --- | --- |
| `random_forest.py` | `05_random_forest.py` | RF models with leave-one-out validation — the head of the line. Retired 2026-08-02: the published model is OLS (D12, D13). |
| `validation.py` | `06_validation.py` | Compared linear vs RF validation results. Inputs `baseline_var_select_aligned_comparison_result.csv` and `random_forest_results_summary.csv` are gone (D5); with RF retired there is nothing left to compare. |
| `model_feature_importance_rf.ipynb` | `04_model_feature_importance_rf.ipynb` | RF feature importance and SHAP over the 91-variable FB set. Fits `RandomForestRegressor` against `models/rf/`, which is gone. Feature importance for the production model is `analysis_notebooks/02` and the D39/E1–E2 sweep in `12` (which was `14` until D52). |
| `result_present.ipynb` | `01_result_present.ipynb` | Presented RF results from `models/rf/` and `files/logs/random_forest_*_results.csv`, looping over six model folders of which only `18_plus` survives — `models/rf/`, `aggregate`, `paired_levels`, `18_plus_paired_levels` and `aggregate_paired_levels` are all gone. |
| `result_compare_yearly_paa.ipynb` | `02_result_compare_yearly_paa.ipynb` | The PAA yearly comparison, on the same RF results. Its successor on the published series is `analysis_notebooks/09_direct_ggi_trend_figures`. **This file does not validate as nbformat** — one of its cells is `cell_type: markdown` while carrying `execution_count` and `outputs`, the same corruption seen in `post_analysis_trend_superseded`. Jupyter opens it; `nbformat.validate` refuses it. Left as found. |
| `monthly_model_fitting.ipynb` | `03_monthly_model_fitting.ipynb` | Monthly RF refits across the six model folders. The production monthly prediction is `src/scripts/prediction.py`, on the OLS. |
| `model_fitting_original.ipynb` | (unnumbered) | The original combined fitting notebook, RF and OLS side by side. Superseded by `src/modelling.py` + `analysis_notebooks/01`. |
| `post_analysis_trend_superseded.ipynb` | the first thirty cells of `09_post_analysis_trend.ipynb` | RF re-prediction with leave-one-out validation, then ITU-kept vs ITU-deleted comparisons. Its pickles (`models/18_plus_no_itu_with_year/`) and its `results/pred_by_year_data/` inputs are both gone. Several of its cells had **never run at all** — a stray `}` opening one, `predicted_results.loc[predicted_results]`, a `'a dll_true'` key, and two code cells saved as `cell_type: markdown`; its saved error outputs are kept as the record of that. Its two working pieces were rebuilt on the published predictions as §8 and §9 of `analysis_notebooks/09_direct_ggi_trend_figures`. |

## Exploration outside the paper (2026-08-31)

Four analysis notebooks archived because **no result in the main paper rests on them**. Unlike
everything else in this folder these still *run* — their inputs are all present. They are here to
keep `src/analysis_notebooks/` to the notebooks the paper actually uses, not because they broke.

Their findings are not lost: `13`'s are D39, D44 and D45 in `doc/decisions_shared.md`, and `15`/`16`
are E3–E6 in `doc/decisions_exploration.md`. Those entries keep the old numbering in their `Scope:`
line; the `Was` column below resolves it.

Their `outputs/fig/` and `outputs/tables/` folders keep the numbered names, because the `NOTEBOOK`
constant in each setup cell is unchanged — an archived notebook that is re-run still writes where
its existing output already sits.

| File | Was | Archived because |
| --- | --- | --- |
| `s1_investigations.ipynb` | `12_s1_investigations.ipynb` | Supplementary S1 checks, never finished — the README carried it as `partial`. Nothing downstream reads it. |
| `extended_panel_exploration.ipynb` | `13_extended_panel_exploration.ipynb` | The panel plus the new surveys, refit and re-validated; §4 the new surveys under both models, §5 the `gdp_pcap` vintage test. Exploration only — it never fed a production artefact (D14, D39, D44, D45). |
| `experimental_model_design.ipynb` | `15_experimental_model_design.ipynb` | The five-term experimental spec scored against production and swept. The result was that it is **worse** (E3–E5), so production is unchanged and the paper reports the production spec. |
| `experimental_model_design_new_parameters.ipynb` | `16_experimental_model_design_new_parameters.ipynb` | Drops `gdp_pcap`, then sweeps all 46 development indicators as substitutes and additions. Concluded `gdp_pcap` is irreplaceable (E6); again, production unchanged. |

## Superseded by a later analysis

| File | Was | Retired because |
| --- | --- | --- |
| `benchmark_model.py` | `03_benchmark_model.py` | OLS benchmarking across specifications. Input `{indicator}_aligned_final_*.csv` is gone (D5); the benchmark question was settled by the final specification (D12). |
| `model_var_selection.py` | `04_model_var_selection.py` | Forward selection / LASSO over the candidate predictors. Input `combined_multiple_years_*_{year}-06.csv` is gone (D5); its output is the seven-term specification now in `params.FINAL_MODEL`. |
| `feature_selection.qmd` | `05_feature_selection.qmd` | The R-side twin of the variable selection above, same fate. |
| `coherent_ggi_figures_superseded.ipynb` | `09_coherent_ggi_figures.ipynb` | Superseded by `analysis_notebooks/04_coherent_ggi_figures.ipynb`. |

Leave-one-country-out validation for the surviving model now lives in `src/modelling.py`, driven by
`analysis_notebooks/01_fit_final_models.ipynb`.

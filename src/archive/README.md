# Archived scripts

Retired from the active pipeline. Kept for provenance, not maintained. None of them run: each
depends on an input that no longer exists on Dropbox, or on the 91-variable Facebook set the
upstream pipeline stopped producing.

Their paths and imports were repaired during the 2026-08 reorganisation, so they are readable and
structurally sound — they are archived because their *purpose* is served, not because they are
beyond repair. Numeric prefixes were dropped so they do not compete with the active `src/`
numbering.

| Script | Was | Retired because |
| --- | --- | --- |
| `benchmark_model.py` | `03_benchmark_model.py` | OLS benchmarking across specifications. Input `{indicator}_aligned_final_*.csv` is gone (D5); the benchmark question was settled by the final specification (D12). |
| `model_var_selection.py` | `04_model_var_selection.py` | Forward selection / LASSO over the candidate predictors. Input `combined_multiple_years_*_{year}-06.csv` is gone (D5); its output is the seven-term specification now recorded in `params.FINAL_MODEL`. |
| `random_forest.py` | `05_random_forest.py` | Random-forest models with leave-one-out validation. Retired 2026-08-02: the published model is OLS, and RF is no longer part of the analysis (D13). Its fitted pickles remain in `outputs/models/18_plus/`. |
| `validation.py` | `06_validation.py` | Compared linear vs random-forest validation results. Inputs `baseline_var_select_aligned_comparison_result.csv` and `random_forest_results_summary.csv` are gone (D5); with RF retired there is nothing left to compare. |

Leave-one-country-out validation for the surviving model now lives in `src/11_fit_final_models.py`.

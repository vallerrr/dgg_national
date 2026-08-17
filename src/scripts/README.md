# Not yet converted

These are still **top-level scripts**: the body runs on import, so they cannot be called from a
notebook and they are not functions. They are kept out of `src/` root so that folder's rule —
functions only (D38) — stays true rather than aspirational.

| script | stage | state |
| --- | --- | --- |
| `outcomes.py` | 01 data creation | runs — merges the indicators into the modelling outcomes |
| `predictors.py` | 01 | runs — year-aligns background predictors, imputes, builds the model matrix |
| `predictors_by_year.py` | 01 | runs — the same, forced to one alignment year |
| `missingness.py` | 01 | runs — per-country missingness, drives the imputation exclusion lists |
| `prediction.py` | 02 model fitting | runs — monthly predictions from the final model |
| `facebook_yearly.py` | 01 | **blocked** — superseded, source gone (D6) |
| `facebook_monthly.py` | 01 | **blocked** — inputs are Dropbox placeholders (D11) |

Run them from the project root:

```bash
python src/scripts/outcomes.py
```

**Converting them** means lifting each body into functions in a `src/` module and leaving a thin
notebook cell that calls it, the way `modelling.py` and `model_performance.py` are already
arranged. Worth doing; not done, and pretending otherwise would be worse than saying so.

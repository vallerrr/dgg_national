# Analysis notebooks

Result presentation, comparison and figure production. These **consume** the pipeline's outputs;
they are not part of it. Retired analyses live in `../archive/`.

## Import convention

Pipeline scripts sit in `src/`, which Python puts on `sys.path[0]` automatically. Modules here get
`src/analysis` instead, so anything importing the project must first do:

```python
import _bootstrap  # noqa: F401  — puts src/ on sys.path
import params
import utils
```

No absolute paths: everything resolves through `params` (`params.RAW`, `params.PROCESSED`,
`params.RESULTS`, `params.FIG`, …). The legacy `params.dropbox_*` names used by the imported
notebooks point through the `data/external` symlink and are equally portable.

## Contents

| Notebook | Does | State |
| --- | --- | --- |
| `01_result_present.ipynb` | Result presentation, maps | not migrated — old imports |
| `02_result_compare_yearly_paa.ipynb` | Yearly comparison for the PAA poster; early direct-vs-composited GGI comparison, superseded by `src/13_coherent_ggi.py` | not migrated — old imports |
| `03_monthly_model_fitting.ipynb` | Monthly model fitting exploration | not migrated — old imports |
| `04_predict_result_analysis.ipynb` | Prediction maps, regional trends, **beta and sigma convergence** | **runs — 14/14 cells** |
| `05_adolescent_analysis.ipynb` | **Adolescent vs adult** predictions, faceted by region / continent / HDI; combined model-fit graph | **runs — 36/36 cells** |
| `06_adolescent_national_prediction.ipynb` | Adolescent national-level prediction, error estimates, FB count comparisons | partial — 28/50 (missing inputs, below) |
| `07_post_analysis_trend.ipynb` | Post-analysis trends, mobile DGG, comparison against the previous prediction round | partial — 17/32 (missing model folder) |
| `08_s1_investigations.ipynb` | Supplementary investigations: **small and negative predictions**, YEM model parameters, threshold checks, alternative specifications | partial — 48/53 |
| `09_coherent_ggi_figures.ipynb` | Trend and regional figures on the coherent GGI → `outputs/fig/coherent_ggi/` | **runs — all cells** |
| `technical_report/*.qmd` | Technical report and the original trend figures (R/Quarto) | Quarto |

`04`–`08` were imported from `dgg_research`'s `origin/pipeline` branch, which this repository never
inherited (D20). Their imports, paths and figure destinations were rewritten to the conventions
above; the analysis logic is unchanged apart from the fixes recorded in D21.

## Known gaps in the partially-running notebooks

Missing **inputs**, not code faults — the same class as D5/D11:

| Notebook | Missing |
| --- | --- |
| `06` | `external/national/adolescent_modelling/data/ins/ins_data_2025.csv`; and one input read from `~/Downloads` that has no shared location — marked `FIXME (D20)` in the cell |
| `07` | `models/18_plus_no_itu_with_year/` — a model folder that does not exist on Dropbox |
| `08` | one cell depends on an earlier cell that needs the above |

## Caveat carried by `04`

Beta convergence is contaminated by the upstream ceiling: where every country in a group reaches
GGI 1.0, `delta = 1 - initial` identically and the regression returns `beta = -1.000`, `R² = 1.000`
by arithmetic rather than by evidence. South America does exactly this — all 12 countries finish at
1.0 on both indicators. The notebook carries this warning inline. See D21.

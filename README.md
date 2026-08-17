# National-Level DGG Model — Data Refresh and Technical Report

Estimates the national **digital gender gap** (DGG) in internet use and mobile ownership, by
combining survey ground truth (DHS, MICS, GSMA, ITU) with Facebook audience-size data and offline
development indicators, then modelling the female/male ratio for countries without recent surveys.

Maintainer: jiaxuan.li@sociology.ox.ac.uk

## Layout

```
├── CONVENTIONS.md        project standards — read this first
├── environment.yml
├── data/                 symlinks into shared Dropbox (see data/README.md)
├── src/                  pipeline scripts (00-12), params.py, utils.py
│   ├── analysis/         result presentation, trend figures, technical report
│   └── archive/          retired scripts
├── doc/
│   ├── workflow.md       file map, dependency spine, refresh procedure
│   ├── modelling.md      which model, on what data, and how it performs
│   ├── methodology.md    methods narrative, assumptions, limitations
│   └── decisions.md      data & method decision records
├── outputs/
│   ├── fig/              technical-report figures (committed)
│   └── {results,models,graphs}/   symlinks into Dropbox
└── logs/
```

## Setup

```bash
mamba env create -f environment.yml     # or: mamba activate dgg_research
```

Then recreate the Dropbox symlinks — see [data/README.md](data/README.md). Nothing runs without
them; `params.py` addresses all data through `params.ROOT`, so no absolute path is baked in.

Verify:

```bash
python -c "import sys; sys.path.insert(0,'src'); import params; print(params.RAW.exists(), params.PROCESSED.exists())"
```

## Running the pipeline

Scripts are numbered in dependency order. Run from the project root:

```bash
python src/05_outcome_data.py
```

The R and Quarto stages (00–03) read their paths from `src/R_params.R`, the R-side companion to
`params.py`. Quarto is not installed on every machine here; `Rscript src/run_qmd.R <file.qmd>`
runs a `.qmd`'s chunks without it.

They are **exploratory analysis scripts, not idempotent jobs** — top-level code with plotting and
inspection interleaved, meant to be stepped through in an IDE as much as run end to end. Several
write into shared Dropbox; read the script before running it.

| # | Script | Does | State |
| --- | --- | --- | --- |
| 00 | `00_data_cleaning_primary.R` | Unzips and files downloaded survey archives | **runs** — `DRY_RUN = TRUE` by default; it moves and deletes files in shared Dropbox (D30) |
| 01 | `01_population_data.R` | UN WPP population counts by age and sex, joined onto the ground truth | **runs** |
| 01_01 | `01_01_un_population_wpp.py` | Rebuilds the full 1950–2023 `un_1950_2023_processed` panel from the WPP raws | **runs** (D29) |
| 02 | `02_ground_truth_data_calculation.qmd` | Internet/mobile penetration from DHS, MICS, GSMA, ITU + UN population | **runs** |
| 03 | `03_internet_indicator_cleaning.qmd` | Harmonises the internet indicator, picks one survey per country | **runs** — map chunks need `rnaturalearth` |
| 04 | `04_data_availability_check.R` | Survey coverage check | R |
| 05 | `05_outcome_data.py` | Merges both indicators into the modelling outcomes | **runs** |
| 06 | `06_facebook_data.py` | Yearly FB ratios standardised on UN population | blocked — superseded, source gone (D6) |
| 07 | `07_facebook_data_monthly.py` | Monthly FB ratios, rolling window | blocked — inputs are Dropbox placeholders (D11) |
| 08 | `08_background_data_for_model.py` | Year-aligns background predictors, imputes, builds the model matrix | **runs** |
| 09 | `09_background_data_by_year.py` | Same, forced to one alignment year per run | **runs** |
| 10 | `10_missing_check.py` | Per-country missingness, drives the imputation exclusion lists | **runs** |
| 11 | `11_fit_final_models.py` | **Fits the final production models from scratch** + LOCO validation + error betas | **runs** |
| 11_01 | `11_01_model_performance.py` | **LOCO performance for all ten model variants** + verification against the shipped pickles | **runs** (D37) |
| 12 | `12_monthly_pred.py` | Monthly predictions from the final model | **runs** |
| 13 | `13_coherent_ggi.py` | **Coherent GGI** from the predicted female/male levels; country-month, country-year and regional tables | **runs** |
| 14 | `14_coherent_ggi_comparison.py` | Figure-ready tables for the coherent-vs-direct appendix | **runs** |
| 15 | `15_coherent_ggi_decomposition.py` | **Exact decomposition** of GGI change into female and male components; typology, weights, regional aggregates | **runs** |
| 16 | `16_aggregation_methods.py` | **Aggregation audit**: seven regional/global summary methods, coverage audit, difference attribution | **runs** |

**[doc/workflow.md](doc/workflow.md) is the map** — which file produces which artefact, what
reads it, the dependency spine, and what a refresh actually requires. Start there.

Analysis and presentation code lives in [src/analysis/](src/analysis/); retired scripts in
[src/archive/](src/archive/).

| Script | Does |
| --- | --- |
| `analysis/01_result_present.ipynb` | Result presentation, maps |
| `analysis/02_result_compare_yearly_paa.ipynb` | Yearly comparison for the PAA poster |
| `analysis/03_monthly_model_fitting.ipynb` | Monthly model fitting exploration |
| `analysis/04_predict_result_analysis.ipynb` | **Beta and sigma convergence**, prediction maps, regional trends |
| `analysis/05_adolescent_analysis.ipynb` | **Adolescent vs adult** predictions by region / continent / HDI |
| `analysis/06_adolescent_national_prediction.ipynb` | Adolescent national prediction, error estimates |
| `analysis/07_post_analysis_trend.ipynb` | Post-analysis trends, comparison against the previous round |
| `analysis/08_s1_investigations.ipynb` | Supplementary: small and negative predictions, threshold checks |
| `analysis/08_01_unseen_survey_validation.ipynb` | **Predicted outcomes against the 17 surveys the models never saw** |
| `analysis/09_coherent_ggi_figures.ipynb` | Trend figures on the coherent GGI -> `outputs/fig/coherent_ggi/` |
| `analysis/09_01_coherent_ggi_comparison.ipynb` | **Appendix**: coherent vs direct GGI (scatter, differences, country trends, regional robustness, appendix text) |
| `analysis/09_02_coherent_ggi_decomposition.ipynb` | **Decomposition**: female vs male drivers of GGI change, typology, maps, regional aggregates |
| `analysis/10_aggregation_methods.ipynb` | **Aggregation audit**: method comparison, rank sensitivity, difference channels, standardisation note |
| `analysis/technical_report/*.qmd` | Technical report and original trend figures; output in `outputs/fig/` |

See [src/analysis/README.md](src/analysis/README.md) for per-notebook status and known input gaps.
`04`-`08` were imported from `dgg_research`'s `origin/pipeline` branch, which this repo never
inherited (D20).

Notebooks 01–03 are **not migrated** — they still carry the old `national.data_refresh.src`
imports. They are not duplicates of the `src/` pipeline: they consume its outputs to make
figures and exploratory tables. Two do overlap with code that has since moved into the pipeline —
`04_model_feature_importance.ipynb` computes the error-estimation betas now produced by
`11_fit_final_models.py`, and `02_result_compare_yearly_paa.ipynb` contains an early
direct-vs-composited GGI comparison now superseded by `13_coherent_ggi.py`. Python modules in `src/analysis/` need `import _bootstrap` before `import params`, since
only `src/` lands on `sys.path` automatically.

"Blocked" means the imports and paths are fixed and the script is structurally sound, but an input
it needs is gone or a methodology question is open. See [doc/decisions.md](doc/decisions.md).

## The final model

The published estimates come from an **OLS**, not a random forest — confirmed against
`dgg_pipeline/src/modelling/national_model.py`, which is the production loader (decision D12):

```
{indicator}_{outcome} ~ fb_18_999_men + fb_18_999_wom + fb_18_999_r + hdi + gdi + gdp_pcap + year
```

Six models (`internet`/`mobile` × `ggi`/`wom`/`men`), stored as
`outputs/models/OLS/{indicator}_combined_with_CIS_{indicator}_{outcome}_full_model.pkl`. The
specification lives in `params.FINAL_MODEL`.

```bash
python src/11_fit_final_models.py --verify   # refit and compare against the shipped models
python src/11_fit_final_models.py            # refit and write to outputs/models/OLS_refit_<date>/
```

The refit reproduces the shipped coefficients to ~1e-15. It writes to a **date-stamped directory
and never overwrites `models/OLS/`** — promotion into production is a deliberate manual step (D14).

`dgg_pipeline` remains canonical for the published series; `src/12_monthly_pred.py` is the local
equivalent and reproduces its predictions value-for-value.

## Documentation

- [CONVENTIONS.md](CONVENTIONS.md) — the standards every script follows
- [data/README.md](data/README.md) — provenance, dictionary, symlink recreation
- [doc/methodology.md](doc/methodology.md) — methods narrative, assumptions, limitations
- [doc/decisions.md](doc/decisions.md) — data and method decision records
- [doc/data_updates.md](doc/data_updates.md) — data currency audit and refresh candidates
- [doc/appendix_coherent_ggi_comparison.md](doc/appendix_coherent_ggi_comparison.md) — generated appendix text
- [doc/decomposition_coherent_ggi.md](doc/decomposition_coherent_ggi.md) — generated decomposition summary
- [doc/aggregation_methods.md](doc/aggregation_methods.md) — generated aggregation standardisation note

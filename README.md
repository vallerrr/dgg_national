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

They are **exploratory analysis scripts, not idempotent jobs** — top-level code with plotting and
inspection interleaved, meant to be stepped through in an IDE as much as run end to end. Several
write into shared Dropbox; read the script before running it.

| # | Script | Does | State |
| --- | --- | --- | --- |
| 00 | `00_data_cleaning_primary.R` | Preprocesses downloaded survey ZIPs | reference only — predates the current Dropbox layout |
| 01 | `01_population_data.R` | UN WPP population counts by age and sex | R |
| 02 | `02_ground_truth_data_calculation.qmd` | Internet/mobile penetration from DHS, MICS, GSMA, ITU + UN population | R |
| 03 | `03_internet_indicator_cleaning.qmd` | Harmonises the internet indicator, picks one survey per country | R |
| 04 | `04_data_availability_check.R` | Survey coverage check | R |
| 05 | `05_outcome_data.py` | Merges both indicators into the modelling outcomes | **runs** |
| 06 | `06_facebook_data.py` | Yearly FB ratios standardised on UN population | blocked — superseded, source gone (D6) |
| 07 | `07_facebook_data_monthly.py` | Monthly FB ratios, rolling window | blocked — inputs are Dropbox placeholders (D11) |
| 08 | `08_background_data_for_model.py` | Year-aligns background predictors, imputes, builds the model matrix | **runs** |
| 09 | `09_background_data_by_year.py` | Same, forced to one alignment year per run | **runs** |
| 10 | `10_missing_check.py` | Per-country missingness, drives the imputation exclusion lists | **runs** |
| 11 | `11_fit_final_models.py` | **Fits the final production models from scratch** + LOCO validation + error betas | **runs** |
| 12 | `12_monthly_pred.py` | Monthly predictions from the final model | **runs** |
| 13 | `13_coherent_ggi.py` | **Coherent GGI** from the predicted female/male levels; country-month, country-year and regional tables | **runs** |

Analysis and presentation code lives in [src/analysis/](src/analysis/); retired scripts in
[src/archive/](src/archive/).

| Script | Does |
| --- | --- |
| `analysis/01_result_present.ipynb` | Result presentation, maps |
| `analysis/02_result_compare_yearly_paa.ipynb` | Yearly comparison for the PAA poster |
| `analysis/03_monthly_model_fitting.ipynb` | Monthly model fitting exploration |
| `analysis/04_model_feature_importance.ipynb` | Feature importance, error-estimation betas |
| `analysis/05_feature_selection.qmd` | R-side feature selection |
| `analysis/technical_report/*.qmd` | Technical report and trend figures; output in `outputs/fig/` |

Notebooks 01–04 are **not migrated** — they still carry the old `national.data_refresh.src`
imports. Python modules in `src/analysis/` need `import _bootstrap` before `import params`, since
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

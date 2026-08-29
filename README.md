# National-Level DGG Model — Data Refresh and Technical Report

Estimates the national **digital gender gap** (DGG) in internet use and mobile ownership, by
combining survey ground truth (DHS, MICS, GSMA, ITU) with Facebook audience-size data and offline
development indicators, then modelling the female/male ratio for countries without recent surveys.

Maintainer: jiaxuan.li@sociology.ox.ac.uk

**New here, or coming back after a while? Read [`doc/STATE.md`](doc/STATE.md) first.** One page:
what is decided, what the project treats as established, what is still open. Then
[`doc/workflow.md`](doc/workflow.md) for how the pieces connect.

**Before touching any data, read its sensitivity tier** — [`doc/data/`](doc/data/), one file per
source. DHS/MICS microdata is `blind` and Facebook audience data is `schema-only`; both tiers are
currently *unconfirmed*. See [`doc/conventions.md`](doc/conventions.md) §7.

---

## Layout

```
├── environment.yml            env name: dgg_research
├── data/                      gitignored symlinks into shared Dropbox — see data/README.md
├── src/
│   ├── params.py              all paths, config, the model registry, SEED, STYLE
│   ├── utils.py               shared helpers
│   ├── plotting.py            the project's chart style
│   ├── <topic>.py             the pipeline's functions — population, modelling,
│   │                          model_performance, coherent_ggi, ggi_comparison,
│   │                          ggi_decomposition, aggregation
│   ├── R_params.R, R_utils.R, run_qmd.R    the R twins of params/utils
│   ├── data_creation/         the R / Quarto survey chain (00–04) and the predictor
│   │                          updaters (05, 06)
│   ├── analysis_notebooks/    the analysis, flat and numbered from 00
│   ├── scripts/               stages not yet converted to functions (D38)
│   ├── technical_report/      the report .qmd sources
│   └── archive/               retired
├── doc/                       the document set — see below
├── outputs/
│   ├── fig/<notebook>/        figures, foldered by the notebook that made them
│   ├── tables/<notebook>/     date-stamped CSVs, same
│   └── {results,models,graphs}/    symlinks into Dropbox
└── logs/
```

`outputs/fig` and `outputs/tables` are **local and not committed** — 82 MB of regenerable artefacts.
Re-run the notebook whose name is the folder.

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

Run from the project root. The R and Quarto stages read their paths from `src/R_params.R`, the
R-side companion to `params.py`; Quarto itself is not required:

```bash
Rscript src/run_qmd.R src/data_creation/02_ground_truth_data_calculation.qmd
python src/data_creation/05_offline_predictors_update.py --dry-run
python src/scripts/outcomes.py
```

`src/scripts/` holds seven stages that are still **top-level scripts, not idempotent jobs** — the
body runs on import, with plotting and inspection interleaved. Read one before running it; several
write into shared Dropbox. Two are blocked for good: `facebook_yearly` (D6) and `facebook_monthly`
(D11). They are kept out of `src/` root so that folder's rule — functions only (D38) — stays true
rather than aspirational.

## The analysis

**A notebook builds its own results and visualises them.** It calls functions from `src/`; it reads
from disk only what an *earlier* notebook produced. Each writes to `outputs/tables/<notebook>/` and
`outputs/fig/<notebook>/`, so an artefact's folder is the notebook that made it.

| # | Notebook | Calls |
| --- | --- | --- |
| 00 | `00_data_creation.ipynb` | `population` |
| 01 | `01_fit_final_models.ipynb` | `modelling` |
| 02 | `02_model_performance.ipynb` | `model_performance` |
| 03 | `03_unseen_survey_validation.ipynb` | reads 02 |
| 04 | `04_coherent_ggi_figures.ipynb` | `coherent_ggi` |
| 05 | `05_coherent_ggi_comparison.ipynb` | `ggi_comparison` — **sets the 0.07 threshold** (D43) |
| 06 | `06_coherent_ggi_decomposition.ipynb` | `ggi_decomposition` |
| 07 | `07_aggregation_methods.ipynb` | `aggregation` |
| 08–12 | trend, adolescent, supplementary | imported analyses (D20) |
| 13–16 | explorations | **nothing here changes a production artefact** (D39, E1–E6) |

See [src/analysis_notebooks/README.md](src/analysis_notebooks/README.md) for per-notebook status and
known input gaps. Notebooks 05, 06 and 07 each **write a report into `doc/`** — edit the notebook and
re-run it; a hand edit to the generated file is lost on the next run.

## The final model

The published estimates come from an **OLS**, not a random forest — confirmed against
`dgg_pipeline/src/modelling/national_model.py`, which is the production loader (D12):

```
{indicator}_{outcome} ~ fb_18_999_men + fb_18_999_wom + fb_18_999_r + hdi + gdi + gdp_pcap + year
```

Six models (`internet`/`mobile` × `ggi`/`wom`/`men`), stored as
`outputs/models/OLS/{indicator}_combined_with_CIS_{indicator}_{outcome}_full_model.pkl`. The
specification lives in `params.FINAL_MODEL`. Fitted on **108** internet and **99** mobile rows,
survey years **2015–2022**.

The refit reproduces the shipped coefficients to ~1e-15, writes to a **date-stamped directory and
never overwrites `models/OLS/`** — promotion into production is a deliberate manual step (D14).
`dgg_pipeline` remains canonical for the published series; `src/scripts/prediction.py` is the local
equivalent and reproduces its predictions value-for-value.

**Predictions run to 2026-05 but the ground truth stops at survey year 2022**, so about a third of
the published series is extrapolation through the linear `year` term. See
[doc/data_updates.md](doc/data_updates.md).

## Documentation

Organised to [`doc/00_project_conventions.md`](doc/00_project_conventions.md), the universal
standard. What this project does differently is in [`doc/conventions.md`](doc/conventions.md), and
only that.

| | |
| --- | --- |
| [doc/STATE.md](doc/STATE.md) | **start here** — active decisions, the ledger, open questions |
| [doc/workflow.md](doc/workflow.md) | the file map: what produces what, the dependency spine, refresh procedures, known traps |
| [doc/data/](doc/data/) | one file per source: provenance, dictionary, **sensitivity tier**, traps |
| [doc/data_creation_methodology.md](doc/data_creation_methodology.md) | how each input is acquired, built, and where it lands |
| [doc/modelling_methodology.md](doc/modelling_methodology.md) | which model, on what data, and how it performs |
| [doc/methodology.md](doc/methodology.md) | methods narrative, assumptions, limitations |
| [doc/data_updates.md](doc/data_updates.md) | data currency audit and refresh candidates |
| [doc/decisions_shared.md](doc/decisions_shared.md) | the decision log, D1–D46 |
| [doc/decisions_exploration.md](doc/decisions_exploration.md) | the same for the exploratory notebooks, E1–E6 |
| [doc/05_coherent_ggi_comparison_report.md](doc/05_coherent_ggi_comparison_report.md) | *generated* — coherent vs directly predicted GGI |
| [doc/06_coherent_ggi_decomposition_report.md](doc/06_coherent_ggi_decomposition_report.md) | *generated* — female vs male drivers of GGI change |
| [doc/07_aggregation_methods_report.md](doc/07_aggregation_methods_report.md) | *generated* — the aggregation standardisation note |
| [data/README.md](data/README.md) | symlink recreation and the tree layout |
| [doc/conventions.md](doc/conventions.md) | this project's deltas from the standard |
| [doc/00_project_conventions.md](doc/00_project_conventions.md) | the universal standard |

**Do not open the decision log to start a session.** It is long and append-only, so it gets skimmed
rather than read. `STATE.md` exists precisely so the current position fits on one page.

# Data provenance and dictionary

Nothing in `data/` is stored in this repository. `raw/`, `processed/` and `external/` are
**gitignored symlinks into the shared Dropbox tree**, so the bytes stay in one place for the
whole team and `params.py` can still address them by project-relative path (CONVENTIONS.md §1).

Recreate them on a new machine with:

```bash
DB=~/Dropbox/dgg_research/national/data_refresh/new_national_pipeline_files
ln -sfn "$DB/files"          data/raw
ln -sfn "$DB/data"           data/processed
ln -sfn ~/Dropbox/dgg_research data/external
ln -sfn "$DB/results"        outputs/results
ln -sfn "$DB/models"         outputs/models
ln -sfn "$DB/graphs"         outputs/graphs
```

| Symlink | Points at | Holds |
| --- | --- | --- |
| `data/raw` | `…/new_national_pipeline_files/files` | Source inputs — never written by this project |
| `data/processed` | `…/new_national_pipeline_files/data` | Derived datasets, reproducible from `raw/` |
| `data/external` | `~/Dropbox/dgg_research` | Upstream artefacts owned by the Facebook pipeline and the shapefile library |
| `outputs/results` | `…/new_national_pipeline_files/results` | Model predictions and run logs |
| `outputs/models` | `…/new_national_pipeline_files/models` | Fitted model pickles |
| `outputs/graphs` | `…/new_national_pipeline_files/graphs` | Figures written by the pipeline scripts |

`outputs/fig/` is the one genuinely local output directory: it holds the technical-report
figures that are committed alongside the `.qmd` sources that produce them.

## Sources

| Source | Acquired via | Lands in |
| --- | --- | --- |
| DHS / MICS surveys | `src/00_data_cleaning_primary.R` from downloaded ZIPs | `raw/full_groundtruth.csv`, `raw/groundtruth_offline_predictors.csv` |
| GSMA Mobile Gender Gap | manual, annual report | folded into the groundtruth files |
| ITU | manual download | folded into the groundtruth files |
| UN World Population Prospects (WPP2022) | `src/01_population_data.R` | `raw/un_1950_2023_processed.csv`, `raw/un_pop_2001_2021.csv`, `raw/population_count.csv` |
| Offline predictors (HDI, WDI, GGGI, V-Dem, WBGI, UNESCO…) | assembled upstream | `raw/offline_predictors.csv` |
| Facebook MAU counts | upstream `dgg_pipeline` | `external/national/pipeline/files/preprocessed_counts/` |
| Facebook national standardised series | upstream `dgg_pipeline` | `external/pipeline/preprocessed/national/fb_national_sd_rolling_std_202606.csv` |
| Natural Earth 110m admin-0 | shapefile library | `external/shape_files/ne_110m_admin_0_countries/` |
| World Bank income classification | manual | `raw/income_classification.csv`, `raw/World_Bank_Group_country_classifications_by_income_level.xlsx` |

## Key derived files

| File | Written by | Contents |
| --- | --- | --- |
| `raw/internet_mobile_indicator_clean.csv` | `src/03_internet_indicator_cleaning.qmd` | Harmonised internet + mobile gender-gap indicators, one row per country-survey |
| `processed/outcome_vars_multiple_years.csv` | `src/05_outcome_data.py` | Both indicators merged; `{indicator}_{ggi,wom,men,year,survey_type}` |
| `processed/fb_data/fb_all_sd_monthly_rolling.csv` | `src/07_facebook_data_monthly.py` | Facebook ratios standardised against UN population, monthly, rolling window |
| `processed/combined_data/updated_ground_truth_and_fb/{indicator}/…_fb_aligned.csv` | `src/08_background_data_for_model.py` | Model matrix: outcomes + background predictors + FB, year-aligned, continent-mean imputed |
| `processed/combined_data/updated_ground_truth_and_fb/{indicator}/year_align/…_{year}.csv` | `src/09_background_data_by_year.py` | Same, but every predictor forced to a single alignment year |
| `processed/outcome_vars/itu_deletion_*.csv` | upstream / manual | Outcome variants with ITU observations removed |
| `outputs/results/pred_by_year_and_month/combined_with_CIS/{date}.csv` | `src/12_monthly_pred.py` | Monthly predictions with confidence intervals |

## Variable naming

- `{indicator}` is `internet` or `mobile`.
- `_ggi` — female/male ratio (the gender gap outcome); `_wom` / `_men` — the sex-specific rates.
- `_r` — a female/male ratio of a background variable; `_year` — which year that value was taken from.
- `fb_{lo}_{hi}_{r,wom,men}` — Facebook audience ratios by age band; `999` means "and over".
- Column dictionaries for the background predictors live in `params.bg_col_dict`.

## Caveats

- **The Facebook count files are Dropbox online-only placeholders.** Every file in
  `external/national/pipeline/files/preprocessed_counts/` is 0 bytes locally and carries the
  `com.dropbox.placeholder` xattr. `07_facebook_data_monthly.py` cannot run until they are
  made available offline (right-click → Make Available Offline in Dropbox).
- **Some 2024-era inputs no longer exist on Dropbox.** See `doc/decisions.md` (D5, D6) for
  which scripts this blocks.
- Raw data keeps its acquisition date and is never re-dated or overwritten (§3).

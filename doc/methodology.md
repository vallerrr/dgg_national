# Methodology

Readable synthesis of the `method`-kind entries in [decisions.md](decisions.md), plus the
assumptions and limitations that matter for write-up (CONVENTIONS.md §8).

## What is being estimated

The **digital gender gap** at country level, for two indicators:

- `internet` — used the internet in the past 12 months
- `mobile` — owns a mobile phone

The modelled outcome is the **gender gap index** `{indicator}_ggi`, the female/male ratio of the
indicator. `_wom` and `_men` (the sex-specific rates) are modelled separately with the same
specification, so a predicted ratio can be decomposed.

## Pipeline

1. **Ground truth** (`00_`-`04_`, R/Quarto). Internet and mobile penetration by sex from DHS, MICS,
   GSMA and ITU, harmonised to a common definition and joined to UN WPP population counts.
   `03_internet_indicator_cleaning.qmd` resolves the priority order: most recent DHS or MICS,
   falling back to ITU, and only surveys starting 2015 or later.
2. **Outcome assembly** (`05_outcome_data.py`). GSMA rows are dropped; where a country has two
   surveys in the same year the rows are averaged and the survey types concatenated; observations
   before `cut_off_year = 2015` are discarded. The two indicators are outer-merged into
   `outcome_vars_multiple_years.csv`.
3. **Facebook predictors** (`06_`, `07_`). Facebook audience sizes by age band and sex are divided by the
   corresponding UN population counts to give standardised ratios, so audience size is not confounded
   by country size or age structure.
4. **Model matrix** (`08_`, `09_`). Each background predictor is **year-aligned** to the outcome's survey
   year, then imputed. Two variants: aligned to each country's own survey year (`08_`), or all
   countries forced to one year (`09_`, used for the yearly prediction series).
5. **Modelling** (`11_`). Fits the final OLS; benchmarking, feature selection and random forest are archived in `src/archive/`.
6. **Prediction** (`12_`). Monthly predictions from the fitted models.

## Assumptions

**Year alignment.** A background predictor is taken from the outcome's survey year where available;
otherwise the most recent earlier year back to 2015; otherwise the nearest later year up to 2022.
Countries with no outcome at all default to 2022. This assumes slow-moving predictors — defensible
for HDI or literacy, weaker for inflation (`hcpi_a`) and unemployment.

**Missing-value imputation.** Missing predictors are filled with the **continent mean**
(`utils.filling_missing_by_continent_mean`), assuming within-continent homogeneity. Countries above
~50% missingness are excluded rather than imputed — the lists are in
`params.countries_to_exclude_for_imputation`, derived by `10_missing_check.py`.

**Validation.** Leave-one-country-out (LOCO) is the headline metric, not in-sample R², because the
model's purpose is prediction for countries with no survey. Leave-one-year-out (LOYO) and
leave-one-continent-out are reported alongside.

**Facebook coverage as a proxy.** The online models assume the female/male ratio of Facebook
audience size tracks the female/male ratio of internet use. Platform penetration, self-reported
profile sex, and multiple-account behaviour all vary by country.

## Limitations

- **Sanctioned countries** (`params.sactioned_countries`: RUS, SDN, CHI) have no Facebook data.
- **Ground-truth sparsity.** Coverage checks in `08_background_data_for_model.py` give 155 country-year observations usable
  for fitting across 127 unique countries, against 243–255 country-years available for prediction.
  Roughly half the prediction set is out-of-sample in the strong sense.
- **ITU comparability.** ITU is not age-restricted, whereas DHS/MICS are 15–49. Variants with ITU
  removed (`*_itu_deleted`, `*_no_ITU`) exist for sensitivity analysis.
- **Continent-mean imputation shrinks variance** and will understate uncertainty for countries with
  many imputed predictors.
- **The FB series changed vintage** between the 2024 run and now (`202411` → `202606`), so
  regenerated results will not reproduce the 2024 numbers exactly. See decision D3.
- **The upstream FB series is now 18+ only.** The 91-variable age-banded set the 2024 models used is
  no longer produced; only `fb_18_999_{r,wom,men}` survives.

## Reproducibility

`params.SEED = 42` is the project seed (§9). The final model is OLS and leave-one-country-out
validation is exhaustive, so **nothing in the production path is stochastic** — `11_fit_final_models.py`
is deterministic and the seed is unused there. The only randomised code was the random forest, now
archived (it never set `random_state`, so its results were not reproducible; see
`src/archive/README.md`).

## The final model

Established against `dgg_pipeline/src/modelling/national_model.py`, which is what actually produces
the published estimates (decision D12). It is an **OLS**, not a random forest:

```
{indicator}_{outcome} = const
                      + fb_18_999_men + fb_18_999_wom + fb_18_999_r     (online)
                      + hdi + gdi + gdp_pcap                            (offline)
                      + year                                            (linear time trend)
```

- Fitted separately for each of `internet`/`mobile` × `ggi`/`wom`/`men` — six models.
- `year` is `{indicator}_year − 2015`, so the intercept is interpretable at the start of the period.
- Training panel: `combined_multiple_years_no_missing_keep_countries_fb_aligned_itu_deleted.csv`,
  giving n = 108 (internet) and n = 99 (mobile) country-survey observations.
- Stored as `outputs/models/OLS/{indicator}_combined_with_CIS_{indicator}_{outcome}_full_model.pkl`.

The specification is deliberately small — three FB ratios and three development indicators. The
2024 exercise explored 91 age-banded FB variables and ~46 offline predictors via forward selection
and random forests (`03`, `04`, `05`); the final model keeps only what survived, which also makes it
robust to the upstream FB series having since narrowed to 18+ only.

`src/11_fit_final_models.py` refits all six from scratch and reproduces the shipped coefficients to
~1e-15.

### Fit and validation

| Model | n | R² | LOCO R² | Mean abs. LOCO error |
| --- | --- | --- | --- | --- |
| internet_ggi | 108 | 0.828 | 0.791 | 0.084 |
| internet_wom | 108 | 0.905 | 0.881 | 0.085 |
| internet_men | 108 | 0.873 | 0.843 | 0.082 |
| mobile_ggi | 99 | 0.739 | 0.683 | 0.065 |
| mobile_wom | 99 | 0.796 | 0.749 | 0.083 |
| mobile_men | 99 | 0.634 | 0.563 | 0.076 |

LOCO R² sits only 3–7 points below in-sample R², so the models generalise to unseen countries
reasonably well. `mobile_men` is the weakest by a clear margin and should be quoted with caution.

### Variant naming

`models/OLS/` holds ten variants; the suffixes encode the **training sample**, not the specification:

| Variant | Training file | n (internet / mobile) |
| --- | --- | --- |
| `combined` | `…_no_ITU.csv` | 71 / 75 |
| `combined_with_CIS` | `…_itu_deleted.csv` | 108 / 99 |
| `combined_with_CIS_align` | as above, reduced FB set (one FB term, matched to the outcome) | 108 / 99 |

`combined` and `combined_with_CIS` share an identical specification and differ only in how ITU
observations were dropped. `_align` is a reduced variant that fits worse (internet_ggi R² 0.765 vs
0.828) and is not the production model.

### Uncertainty band

`predicted_error` comes from a **non-negative least squares** fit of the absolute LOCO error on the
same regressors (`combined_with_CIS_country_model_error_estimation_betas.csv`). Non-negativity keeps
the band positive for every country.

**This band is close to meaningless as a per-country quantity.** The NNLS puts all weight on `gdi`
and zero on everything else, and `gdi` is ≈0.95 for nearly every country — so the "error model"
effectively returns a constant equal to the mean absolute error. Its R² against the actual absolute
errors is slightly **negative** (−0.03 to +0.001), i.e. no better than that constant. It should be
read as "typical error for this indicator", not as country-specific uncertainty.

## The coherent GGI

The paper models female levels, male levels and the GGI separately, so the directly predicted GGI
is not guaranteed to equal predicted-female / predicted-male. `src/13_coherent_ggi.py` derives that
ratio explicitly and it is now the **primary** measure:

```
raw    = predicted_female_level / predicted_male_level     (may exceed 1)
capped = min(raw, 1)                                       (primary)
```

The directly predicted GGI is retained everywhere as `{indicator}_ggi_direct` for robustness; no
original variable is overwritten.

| Variable | Meaning |
| --- | --- |
| `{ind}_ggi_coherent` | **primary** — parity-capped `min(raw, 1)` |
| `{ind}_ggi_coherent_raw` | uncapped ratio, retained for robustness |
| `{ind}_ggi_direct` | the separately modelled GGI (was `{ind}_fm_ratio`) |
| `{ind}_ggi_coherent_capped_flag` | whether the parity cap bound |
| `{ind}_coherent_flag` | `both_saturated` / `male_saturated` / `female_zero` / `both_zero` / `none` |

### Why the boundary flag matters

Levels are proportions clipped to [0, 1] upstream, which is correct — a predicted male adoption of
1.15 means "saturated", not "115% of men". But when **both** sexes sit at the ceiling the coherent
GGI is exactly 1 *by construction*, and after parity-capping that is indistinguishable from a
genuine cap. This is not rare: it affects **6.8%** of internet and **16.8%** of mobile
country-months. `{ind}_coherent_flag` keeps it auditable — any headline claim about parity should
exclude or at least report the `both_saturated` share.

### Coherent vs directly predicted

| Indicator | corr | mean abs diff | max abs diff | raw > 1 |
| --- | --- | --- | --- | --- |
| internet | 0.938 | 0.039 | 0.419 | 16.1% |
| mobile | 0.993 | 0.007 | 0.124 | 38.5% |

They agree closely on average but diverge materially for individual country-months (up to 0.42 for
internet), and the divergence is **directional at regional level** — for 2024 Sub-Saharan Africa the
coherent GGI is *higher* than the direct one (0.758 vs 0.716, i.e. a smaller gap) while Middle East
& North Africa moves the other way (0.864 vs 0.878). Switching measures is therefore not cosmetic.

### Annual aggregation

Both orders are computed, and they turn out to be equivalent: mean-of-monthly-ratios and
ratio-of-annual-mean-levels differ by a mean of 0.0000 and a maximum of 0.001 — below the 3-decimal
rounding of the source predictions. **The choice is immaterial**; `mean_of_ratios` matches the
existing trend figures and is the sensible default.

### Regional aggregation

World Bank regions (`data/raw/iso3_regions.csv`) — the scheme the trend figures already use, and
the only one covering all 214 countries with no gaps. Both unweighted and adult-population-weighted
regional means are produced. Population is available for 212 of 214 countries for 2015–2024; **2025
and 2026 have no weighted figure** rather than a carried-forward guess.

### Edge cases

- `NER 2015-01` is the only 0/0 case (both levels 0). Per the epsilon rule the denominator is
  floored at 1e-9, giving a coherent GGI of 0 — read as "nobody has internet", not "maximum gender
  gap". It carries `both_zero`. The directly predicted GGI reports 0.136 for the same row, which is
  exactly the sort of incoherence this measure exists to expose.
- No country-date is missing a female or male prediction: the panel is complete (19,893 × 6).
- Sanctioned countries (RUS, SDN, CHI) are already excluded upstream and stay excluded.

### Uncertainty

`predicted_error` is **not** propagated to the coherent GGI. A ratio's variance is not the ratio of
variances, and the underlying error model is near-degenerate (see above), so a derived band would
imply more precision than exists.

## Open questions

None blocking. The model-selection scripts whose inputs no longer exist (D5) and whose purpose was
served once the specification above was settled have been moved to `src/archive/` (D15).

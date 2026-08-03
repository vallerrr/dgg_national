"""
# Created by valler at 03/08/2026
Feature: build the internally coherent gender gap index from the predicted female and male levels.

The paper models female levels, male levels and the GGI separately. The directly predicted GGI is
not guaranteed to equal predicted_female / predicted_male, so this module derives that ratio
explicitly and makes it the primary measure:

    raw    = predicted_female_level / predicted_male_level      (may exceed 1)
    capped = min(raw, 1)                                        (primary — >1 is out of scope)

The directly predicted GGI is preserved throughout as `{indicator}_ggi_direct` for robustness
analyses; nothing is overwritten.

Reads dgg_pipeline's published series read-only — that is the golden standard the new numbers are
compared against — and writes only to outputs/tables/.

    python src/13_coherent_ggi.py
"""
from datetime import datetime

import numpy as np
import pandas as pd

import params
import utils

CFG = params.COHERENT_GGI
INDICATORS = CFG['indicators']
OUTDIR = params.OUTPUTS / 'tables'


def load_predictions():
    """
    the published country-date predictions, reshaped one row per country-date

    The source is long (`gid_0, outcome, predicted, predicted_error, date`) with six outcomes per
    country-date. Levels keep their names; the directly predicted GGI is renamed to `_ggi_direct`
    so it cannot be confused with the coherent measure built below.
    """
    long = pd.read_csv(CFG['source'])
    wide = long.pivot_table(index=['gid_0', 'date'], columns='outcome', values='predicted').reset_index()
    wide.columns.name = None

    wide = wide.rename(columns={f'{ind}_fm_ratio': f'{ind}_ggi_direct' for ind in INDICATORS})

    wide['year'] = wide['date'].str.slice(0, 4).astype(int)
    wide['month'] = wide['date'].str.slice(5, 7).astype(int)
    return wide


def add_geography(df):
    """attach World Bank region, continent and country name"""
    regions = utils.load_regions()
    df = df.merge(regions, left_on='gid_0', right_on='iso3', how='left').drop(columns='iso3')
    df['continent'] = [utils.get_continent_from_iso3(x) for x in df['gid_0']]
    df['country'] = [utils.get_country_name_from_iso3(x) for x in df['gid_0']]
    return df


def coherent_ggi(female, male):
    """
    female / male, with a zero male level floored at epsilon so the ratio stays defined

    Returns the raw ratio; capping is applied separately so both variants survive.
    """
    return female / male.where(male > 0, CFG['epsilon'])


def flag_boundary_cases(female, male, raw):
    """
    label the country-dates where the ratio is an artefact of the level bounds rather than a gap

    Levels are proportions clipped to [0, 1] upstream. When both sexes sit at the ceiling the ratio
    is exactly 1 by construction — indistinguishable from the parity cap unless it is flagged. When
    both sit at zero nobody has the technology at all, so a "gender gap" of 0 is not meaningful.
    """
    return np.select(
        [(male == 0) & (female == 0),
         (male == 1) & (female == 1),
         male == 1,
         female == 0],
        ['both_zero', 'both_saturated', 'male_saturated', 'female_zero'],
        default='none')


def build_country_month(df):
    """the country-month panel with both coherent variants and their audit flags"""
    for ind in INDICATORS:
        female, male = df[f'{ind}_women'], df[f'{ind}_men']

        raw = coherent_ggi(female, male)
        df[f'{ind}_ggi_coherent_raw'] = raw
        df[f'{ind}_ggi_coherent'] = raw.clip(upper=CFG['parity_cap'])
        df[f'{ind}_ggi_coherent_capped_flag'] = raw > CFG['parity_cap']
        df[f'{ind}_coherent_flag'] = flag_boundary_cases(female, male, raw)

    return df


def build_country_year(df):
    """
    annual series, aggregated both ways so the two definitions can be compared

    `mean_of_ratios` averages the twelve monthly ratios — what the existing trend figures do.
    `ratio_of_means` averages the monthly levels first, then divides. They differ by Jensen's
    inequality; the difference is reported in the diagnostics table.
    """
    keys = ['gid_0', 'country', 'region', 'continent', 'year']
    level_cols = [f'{ind}_{s}' for ind in INDICATORS for s in ('women', 'men')]
    direct_cols = [f'{ind}_ggi_direct' for ind in INDICATORS]
    ratio_cols = [f'{ind}_ggi_coherent_raw' for ind in INDICATORS]

    annual = df.groupby(keys, as_index=False)[level_cols + direct_cols + ratio_cols].mean()
    annual = annual.rename(columns={c: c.replace('_raw', '_mean_of_ratios_raw') for c in ratio_cols})

    for ind in INDICATORS:
        # order 1: mean of the monthly ratios
        mor_raw = annual[f'{ind}_ggi_coherent_mean_of_ratios_raw']
        annual[f'{ind}_ggi_coherent_mean_of_ratios'] = mor_raw.clip(upper=CFG['parity_cap'])

        # order 2: ratio of the annual mean levels
        rom_raw = coherent_ggi(annual[f'{ind}_women'], annual[f'{ind}_men'])
        annual[f'{ind}_ggi_coherent_ratio_of_means_raw'] = rom_raw
        annual[f'{ind}_ggi_coherent_ratio_of_means'] = rom_raw.clip(upper=CFG['parity_cap'])

    return annual


def load_adult_population():
    """
    18+ population by country-year, for population-weighted regional aggregates

    Only years the UN file actually covers; later years get no weight rather than a carried-forward
    guess, so a weighted regional figure is either real or absent.
    """
    pop = pd.read_csv(params.RAW / 'un_1950_2023_processed.csv', usecols=['iso3', 'Year', '18_inf_m', '18_inf_f'])
    pop['adult_pop'] = pop['18_inf_m'] + pop['18_inf_f']
    return pop.rename(columns={'iso3': 'gid_0', 'Year': 'year'})[['gid_0', 'year', 'adult_pop']]


def build_region_year(annual):
    """regional means, unweighted and weighted by adult population"""
    annual = annual.merge(load_adult_population(), on=['gid_0', 'year'], how='left')

    measures = [f'{ind}_ggi_{v}' for ind in INDICATORS
                for v in ('direct', 'coherent_mean_of_ratios', 'coherent_ratio_of_means')]

    rows = []
    for (region, year), grp in annual.groupby(['region', 'year']):
        row = {'region': region, 'year': year,
               'n_countries': len(grp),
               'n_countries_with_pop': int(grp['adult_pop'].notna().sum())}

        weights = grp['adult_pop']
        for m in measures:
            row[f'{m}_unweighted'] = grp[m].mean()
            ok = weights.notna() & grp[m].notna()
            row[f'{m}_pop_weighted'] = (
                np.average(grp.loc[ok, m], weights=weights[ok]) if ok.any() else np.nan)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(['region', 'year'], ignore_index=True)


def build_diagnostics(monthly, annual):
    """
    how much the choices actually matter

    Three comparisons per indicator: coherent vs directly predicted, the two annual aggregation
    orders against each other, and how often the parity cap and the boundary flags bind.
    """
    rows = []
    for ind in INDICATORS:
        raw = monthly[f'{ind}_ggi_coherent_raw']
        capped = monthly[f'{ind}_ggi_coherent']
        direct = monthly[f'{ind}_ggi_direct']
        mor = annual[f'{ind}_ggi_coherent_mean_of_ratios']
        rom = annual[f'{ind}_ggi_coherent_ratio_of_means']

        rows.append({
            'indicator': ind,
            'n_country_months': len(monthly),
            'corr_coherent_vs_direct': capped.corr(direct),
            'mean_abs_diff_coherent_vs_direct': (capped - direct).abs().mean(),
            'max_abs_diff_coherent_vs_direct': (capped - direct).abs().max(),
            'pct_raw_above_parity': (raw > CFG['parity_cap']).mean() * 100,
            'pct_flag_both_saturated': (monthly[f'{ind}_coherent_flag'] == 'both_saturated').mean() * 100,
            'pct_flag_male_saturated': (monthly[f'{ind}_coherent_flag'] == 'male_saturated').mean() * 100,
            'pct_flag_female_zero': (monthly[f'{ind}_coherent_flag'] == 'female_zero').mean() * 100,
            'n_flag_both_zero': int((monthly[f'{ind}_coherent_flag'] == 'both_zero').sum()),
            'annual_mean_abs_diff_mor_vs_rom': (mor - rom).abs().mean(),
            'annual_max_abs_diff_mor_vs_rom': (mor - rom).abs().max(),
        })
    return pd.DataFrame(rows)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stamp = f'{datetime.now():%Y%m%d}'

    monthly = build_country_month(add_geography(load_predictions()))
    annual = build_country_year(monthly)
    regional = build_region_year(annual)
    diagnostics = build_diagnostics(monthly, annual)

    for name, frame in [('country_month', monthly), ('country_year', annual),
                        ('region_year', regional), ('diagnostics', diagnostics)]:
        path = OUTDIR / f'coherent_ggi_{name}_{stamp}.csv'
        frame.to_csv(path, index=False)
        print(f'{path.relative_to(params.ROOT)}  ({len(frame)} rows)')

    print()
    print(diagnostics.round(4).to_string(index=False))


if __name__ == '__main__':
    main()

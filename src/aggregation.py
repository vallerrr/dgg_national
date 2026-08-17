"""
# Created by valler at 06/08/2026
Feature: audit and standardise every regional / global aggregation of the digital gender gap.

The paper has at various points referred to means, medians, population-weighted means and
Facebook-count ratios as though they were the same summary. They are not, and they can differ enough
to change a regional ordering. This module computes all of them side by side, on one sample and one
set of definitions, so any statistic in the manuscript can be traced to a named method.

Methods computed per indicator x region x year:

    mean_unweighted            mean of country GGI
    median_unweighted          median of country GGI
    mean_pop_weighted_female   mean of country GGI, weighted by adult FEMALE population
    mean_pop_weighted_total    mean of country GGI, weighted by adult TOTAL population
    aggregate_ggi              [sum(popF*f)/sum(popF)] / [sum(popM*m)/sum(popM)]
    aggregate_ggi_single_denom same, but total adult population as the denominator for both sexes
    aggregate_ggi_parity_capped  min(aggregate_ggi, 1)

`aggregate_ggi` is the only one that aggregates the sexes on their own denominators; the
`single_denom` variant exists solely to isolate how much the sex population structure contributes.

Denominators are the **18+ population by sex**, which is the age group the national models are
fitted on (`params.FINAL_MODEL`), so the denominator matches the target group rather than standing
in for it. Total population is reported alongside, never substituted silently.

    python src/16_aggregation_methods.py
"""
from datetime import datetime

import numpy as np
import pandas as pd

import params
import utils

CFG = params.COHERENT_GGI
INDICATORS = CFG['indicators']
STAGE = 'coherent_ggi'
OUTDIR = params.table_dir(STAGE)  # outputs/tables/<stage>/ — the path names the step
PREFIX = 'aggregation'

TREND_START, TREND_END = 2015, 2025      # 2026 is partial (D18)
COMPARE_YEARS = [2015, 2025]
BASE_YEAR = 2015                         # for the fixed-weight basis
GLOBAL = 'GLOBAL'

# The denominator actually used, and the age group it refers to. Recorded on every row so a figure
# or manuscript line can state it without the reader having to trace the code.
AGE_GROUP = params.COHERENT_GGI['age_group']     # '18_plus'
POP_FEMALE, POP_MALE, POP_TOTAL = '18_inf_f', '18_inf_m', '18_inf_t'

METHOD_LABEL = {
    'mean_unweighted': 'Unweighted mean of country GGI',
    'median_unweighted': 'Unweighted median of country GGI',
    'mean_pop_weighted_female': 'Country GGI, weighted by adult female population',
    'mean_pop_weighted_total': 'Country GGI, weighted by adult total population',
    'aggregate_ggi': 'Aggregate GGI (sex-specific 18+ denominators)',
    'aggregate_ggi_single_denom': 'Aggregate GGI (total 18+ denominator for both sexes)',
    'aggregate_ggi_parity_capped': 'Aggregate GGI, parity-capped (sex-specific denominators)',
}


# ====================================================================================================
# inputs
# ====================================================================================================
def latest(stem):
    matches = sorted(OUTDIR.glob(f'{stem}_*.csv'))
    if not matches:
        raise FileNotFoundError(f'no {stem}_*.csv — run src/13_coherent_ggi.py first')
    return pd.read_csv(matches[-1])


def load_levels():
    """country-year female and male adoption levels, plus geography"""
    annual = latest('coherent_ggi_country_year')
    return annual[annual['year'].between(TREND_START, TREND_END)].copy()


def load_population():
    """
    18+ population by sex and in total, per country-year

    18+ is the age group the national models are fitted on, so this is an age-aligned denominator
    rather than a stand-in. `population_coverage()` reports where it is unavailable in its own
    year and what the carry-forward rule (D32) supplies instead.
    """
    pop = pd.read_csv(params.UN_POP_PROCESSED,
                      usecols=['iso3', 'Year', POP_FEMALE, POP_MALE, POP_TOTAL])
    return pop.rename(columns={'iso3': 'gid_0', 'Year': 'year',
                               POP_FEMALE: 'pop_female', POP_MALE: 'pop_male',
                               POP_TOTAL: 'pop_total'})


def population_coverage(levels, pop):
    """
    which years have an age-aligned denominator, before and after the carry-forward rule

    Two different questions, and reporting only the first made the audit contradict the code:
    `*_own_year` is whether the UN file has that year itself; `*_effective` is what weighting
    actually uses after D32 carries the latest earlier year forward. A year can be 0% on the
    first and 100% on the second — that is precisely the case the `pop_year_used` column names.
    """
    rows = []
    for year in sorted(levels['year'].unique()):
        need = set(levels.loc[levels['year'] == year, 'gid_0'])
        have = set(pop.loc[(pop['year'] == year) & pop['pop_female'].notna(), 'gid_0'])
        # what the shared rule delivers: any country with population in that year or earlier
        carried = set(pop.loc[(pop['year'] <= year) & pop['pop_female'].notna(), 'gid_0'])
        source_years = pop.loc[pop['gid_0'].isin(need & carried) & (pop['year'] <= year)
                               & pop['pop_female'].notna(), 'year']
        rows.append({
            'year': year, 'age_group': AGE_GROUP,
            'n_countries_modelled': len(need),
            'n_with_sex_specific_pop': len(need & have),
            'coverage_pct': len(need & have) / len(need) * 100 if need else np.nan,
            'contemporaneous_weighting_possible': len(need & have) == len(need),
            'n_with_pop_effective': len(need & carried),
            'coverage_pct_effective': len(need & carried) / len(need) * 100 if need else np.nan,
            'pop_year_used': int(source_years.max()) if len(source_years) else pd.NA,
        })
    return pd.DataFrame(rows)


def attach_weights(levels, pop, basis):
    """
    join population on one of two bases

    'contemporaneous' uses each year's own population, falling back to the latest earlier year
    under the shared rule (D32) — so years past the end of the UN file are now weighted on carried
    population rather than left absent, with `pop_year` recording which year each row used.
    'fixed_2015' applies the base-year structure to every year, which keeps the whole 2015-2025
    window computable on one consistent weighting and is what the decomposition uses (D24).
    """
    if basis == 'contemporaneous':
        return utils.join_latest_available_year(levels, pop, key='gid_0',
                                                used_year_col='pop_year')
    base = pop[pop['year'] == BASE_YEAR].drop(columns='year')
    return levels.merge(base, on='gid_0', how='left')


# ====================================================================================================
# the aggregation methods
# ====================================================================================================
def _weighted(values, weights):
    ok = values.notna() & weights.notna() & (weights > 0)
    return np.average(values[ok], weights=weights[ok]) if ok.any() else np.nan


def aggregate_block(block, indicator):
    """every method, for one indicator over one set of countries"""
    female, male = block[f'{indicator}_women'], block[f'{indicator}_men']
    ggi = female / male.where(male > 0, np.nan)

    has_pop = block['pop_female'].notna() & block['pop_male'].notna()
    out = {
        'n_countries': int(ggi.notna().sum()),
        'n_countries_with_pop': int((ggi.notna() & has_pop).sum()),
        'mean_unweighted': ggi.mean(),
        'median_unweighted': ggi.median(),
        'mean_pop_weighted_female': _weighted(ggi, block['pop_female']),
        'mean_pop_weighted_total': _weighted(ggi, block['pop_total']),
    }

    # aggregate rates: each sex on its own denominator
    f_rate = _weighted(female, block['pop_female'])
    m_rate = _weighted(male, block['pop_male'])
    out['regional_female_rate'] = f_rate
    out['regional_male_rate'] = m_rate
    out['aggregate_ggi'] = f_rate / m_rate if (m_rate and np.isfinite(m_rate)) else np.nan
    out['aggregate_ggi_parity_capped'] = (min(out['aggregate_ggi'], CFG['parity_cap'])
                                          if np.isfinite(out['aggregate_ggi']) else np.nan)

    # same, but one shared denominator — the difference is the sex population structure
    f_single = _weighted(female, block['pop_total'])
    m_single = _weighted(male, block['pop_total'])
    out['aggregate_ggi_single_denom'] = (f_single / m_single
                                         if (m_single and np.isfinite(m_single)) else np.nan)

    # descriptive context for the decomposition
    out['sex_ratio_pop_m_over_f'] = (block['pop_male'].sum() / block['pop_female'].sum()
                                     if block['pop_female'].sum() else np.nan)
    out['ggi_min'] = ggi.min()
    out['ggi_max'] = ggi.max()
    out['ggi_iqr'] = ggi.quantile(0.75) - ggi.quantile(0.25)
    return out


def build_methods(levels, pop, basis):
    """all methods, for every indicator x region x year, plus a GLOBAL row"""
    data = attach_weights(levels, pop, basis)
    rows = []
    for indicator in INDICATORS:
        for year, year_block in data.groupby('year'):
            groups = [(GLOBAL, year_block)] + list(year_block.groupby('region'))
            for region, block in groups:
                rows.append({'indicator': indicator, 'region': region, 'year': int(year),
                             'weight_basis': basis, 'age_group': AGE_GROUP,
                             **aggregate_block(block, indicator)})
    return pd.DataFrame(rows)


# ====================================================================================================
# why the summaries differ — a stepwise attribution
# ====================================================================================================
STEPS = [
    ('country_composition',
     'mean_unweighted', 'mean_unweighted_pop_sample',
     'Restricting to countries that have an age-aligned population denominator'),
    ('extreme_country_values',
     'mean_unweighted_pop_sample', 'median_unweighted',
     'Mean to median on the same countries: the pull of outlying country GGIs'),
    ('population_weights',
     'mean_unweighted_pop_sample', 'mean_pop_weighted_total',
     'Giving each country weight in proportion to its adult population'),
    ('ratio_of_means',
     'mean_pop_weighted_total', 'aggregate_ggi_single_denom',
     'Aggregating the levels first and dividing, instead of averaging country ratios'),
    ('sex_population_structure',
     'aggregate_ggi_single_denom', 'aggregate_ggi',
     'Using each sex\'s own 18+ denominator instead of one shared denominator'),
    ('parity_ceiling',
     'aggregate_ggi', 'aggregate_ggi_parity_capped',
     'Capping the aggregate at parity (D25)'),
]


def build_decomposition(levels, pop, basis):
    """
    walk from the unweighted mean to the parity-capped aggregate, one channel at a time

    Each step changes exactly one thing, so its delta is attributable to that channel alone. The
    steps chain, so the deltas sum to the total difference between the two headline summaries.
    """
    data = attach_weights(levels, pop, basis)
    rows = []
    for indicator in INDICATORS:
        for year, year_block in data.groupby('year'):
            groups = [(GLOBAL, year_block)] + list(year_block.groupby('region'))
            for region, block in groups:
                vals = aggregate_block(block, indicator)
                with_pop = block[block['pop_female'].notna() & block['pop_male'].notna()]
                sub = aggregate_block(with_pop, indicator)
                vals['mean_unweighted_pop_sample'] = sub['mean_unweighted']
                vals['median_unweighted'] = sub['median_unweighted']

                for name, frm, to, why in STEPS:
                    rows.append({
                        'indicator': indicator, 'region': region, 'year': int(year),
                        'weight_basis': basis, 'channel': name,
                        'from_method': frm, 'to_method': to,
                        'from_value': vals.get(frm), 'to_value': vals.get(to),
                        'delta': (vals.get(to) - vals.get(frm)
                                  if pd.notna(vals.get(to)) and pd.notna(vals.get(frm)) else np.nan),
                        'explanation': why,
                    })
    return pd.DataFrame(rows)


def build_comparison_table(methods):
    """the 2015 vs 2025 table, one row per region x method"""
    keep = methods[methods['year'].isin(COMPARE_YEARS)]
    long = keep.melt(id_vars=['indicator', 'region', 'year', 'weight_basis',
                              'n_countries', 'n_countries_with_pop', 'age_group'],
                     value_vars=list(METHOD_LABEL),
                     var_name='method', value_name='value')
    wide = long.pivot_table(index=['indicator', 'weight_basis', 'region', 'method',
                                   'n_countries', 'age_group'],
                            columns='year', values='value').reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={y: f'ggi_{y}' for y in COMPARE_YEARS})
    if all(f'ggi_{y}' in wide for y in COMPARE_YEARS):
        wide['change'] = wide[f'ggi_{COMPARE_YEARS[1]}'] - wide[f'ggi_{COMPARE_YEARS[0]}']
        wide['rank_2025'] = (wide[wide['region'] != GLOBAL]
                             .groupby(['indicator', 'weight_basis', 'method'])[f'ggi_{COMPARE_YEARS[1]}']
                             .rank(ascending=False, method='min'))
    wide['method_label'] = wide['method'].map(METHOD_LABEL)
    return wide.sort_values(['indicator', 'weight_basis', 'method', 'region'], ignore_index=True)


def build_rank_disagreement(comparison):
    """where the choice of method reorders the regions — the load-bearing check"""
    rows = []
    block = comparison[comparison['region'] != GLOBAL]
    for (indicator, basis), grp in block.groupby(['indicator', 'weight_basis']):
        pivot = grp.pivot_table(index='region', columns='method', values='rank_2025')
        for region in pivot.index:
            ranks = pivot.loc[region].dropna()
            rows.append({
                'indicator': indicator, 'weight_basis': basis, 'region': region,
                **{f'rank_{m}': int(v) for m, v in ranks.items()},
                'n_distinct_ranks': int(ranks.nunique()),
                'max_rank_shift': int(ranks.max() - ranks.min()),
            })
    return pd.DataFrame(rows).sort_values(['indicator', 'weight_basis', 'region'], ignore_index=True)


# ====================================================================================================
def main():
    stamp = f'{datetime.now():%Y%m%d}'

    levels = load_levels()
    pop = load_population()

    coverage = population_coverage(levels, pop)
    methods = pd.concat([build_methods(levels, pop, b)
                         for b in ('fixed_2015', 'contemporaneous')], ignore_index=True)
    decomposition = pd.concat([build_decomposition(levels, pop, b)
                               for b in ('fixed_2015', 'contemporaneous')], ignore_index=True)
    comparison = build_comparison_table(methods)
    disagreement = build_rank_disagreement(comparison)

    outputs = {
        'population_coverage': coverage,
        'methods': methods,
        'comparison_2015_2025': comparison,
        'difference_decomposition': decomposition,
        'rank_disagreement': disagreement,
        'method_labels': pd.DataFrame({'method': list(METHOD_LABEL),
                                       'label': list(METHOD_LABEL.values())}),
    }
    for name, frame in outputs.items():
        path = OUTDIR / f'{PREFIX}_{name}_{stamp}.csv'
        frame.to_csv(path, index=False)
        print(f'  {path.relative_to(params.ROOT)}  ({len(frame)} rows)')

    print('\npopulation denominator coverage (18+ by sex):')
    print(coverage.to_string(index=False))

    print('\nGLOBAL, internet, fixed_2015 weights:')
    g = methods[(methods['region'] == GLOBAL) & (methods['indicator'] == 'internet')
                & (methods['weight_basis'] == 'fixed_2015')
                & (methods['year'].isin(COMPARE_YEARS))]
    print(g[['year'] + list(METHOD_LABEL)].round(4).to_string(index=False))


if __name__ == '__main__':
    main()

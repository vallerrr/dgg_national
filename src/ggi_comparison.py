"""
# Created by valler at 05/08/2026
Feature: figure-ready datasets for the coherent-vs-direct GGI appendix.

Builds every table the appendix notebook plots, so the analysis is reproducible without running a
notebook and the plotting code contains no data preparation. `src/analysis/09_01_coherent_ggi_comparison.ipynb`
reads these and draws.

Country selection for the trend panels follows a rule fixed here, applied before any figure is
inspected — see `select_trend_countries`.

    python src/14_coherent_ggi_comparison.py
"""
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

import params

CFG = params.COHERENT_GGI
INDICATORS = CFG['indicators']
STAGE = 'coherent_ggi'
OUTDIR = params.table_dir(STAGE)  # outputs/tables/<stage>/ — the path names the step
PREFIX = 'ggi_comparison'

SNAPSHOT_YEARS = CFG['focus_years']          # 2015, 2020, 2025
TREND_START, TREND_END = 2015, 2025          # last complete year; 2026 is partial (D18)
N_PER_GROUP = 3                              # countries per discrepancy group in panel C

DEFINITIONS = {
    'coherent_parity': '{ind}_ggi_coherent_parity',
    'coherent_raw': '{ind}_ggi_coherent_raw',
    'direct': '{ind}_ggi_direct',
}


# ====================================================================================================
# inputs
# ====================================================================================================
def latest(stem):
    matches = sorted(OUTDIR.glob(f'{stem}_*.csv'))
    if not matches:
        raise FileNotFoundError(f'no {stem}_*.csv — run src/13_coherent_ggi.py first')
    return pd.read_csv(matches[-1])


def load_country_year():
    """
    annual country series, restricted to complete years

    `13_coherent_ggi.py` already drops `params.excluded_countries` and the sanctioned countries are
    absent upstream, so no further filtering is applied here.
    """
    annual = latest('coherent_ggi_country_year')
    annual = annual[annual['year'].between(TREND_START, TREND_END)].copy()
    annual = annual.rename(columns={
        f'{ind}_ggi_coherent_mean_of_ratios': f'{ind}_ggi_coherent_parity' for ind in INDICATORS})
    annual = annual.rename(columns={
        f'{ind}_ggi_coherent_mean_of_ratios_raw': f'{ind}_ggi_coherent_raw' for ind in INDICATORS})
    return annual


def common_sample(annual):
    """
    countries usable under every definition, for all three indicators' comparisons

    Panel D requires identical country samples across definitions; applying that same restriction
    everywhere keeps panels A-E mutually consistent rather than each on its own footing.
    """
    cols = [d.format(ind=ind) for ind in INDICATORS for d in DEFINITIONS.values()]
    complete = annual.groupby('gid_0')[cols].apply(lambda g: g.notna().all().all())
    keep = complete[complete].index
    dropped = sorted(set(annual['gid_0']) - set(keep))
    return annual[annual['gid_0'].isin(keep)].copy(), dropped


# ====================================================================================================
# A. scatter: direct vs coherent raw
# ====================================================================================================
def build_scatter(annual):
    """one row per country-year-indicator for the snapshot years, plus per-panel statistics"""
    rows, stats_rows = [], []
    for ind in INDICATORS:
        for year in SNAPSHOT_YEARS:
            block = annual[annual['year'] == year]
            x = block[f'{ind}_ggi_direct']
            y = block[f'{ind}_ggi_coherent_raw']
            ok = x.notna() & y.notna()

            rows.append(pd.DataFrame({
                'indicator': ind, 'year': year,
                'gid_0': block['gid_0'], 'country': block['country'],
                'continent': block['continent'], 'region': block['region'],
                'direct': x, 'coherent_raw': y,
                'diff': y - x,
            }))
            stats_rows.append({
                'indicator': ind, 'year': year, 'n': int(ok.sum()),
                'pearson_r': stats.pearsonr(x[ok], y[ok])[0],
                'spearman_r': stats.spearmanr(x[ok], y[ok])[0],
                'n_above_1_coherent_raw': int((y > 1).sum()),
                'max_coherent_raw': float(y.max()),
                'mean_abs_diff': float((y - x).abs().mean()),
            })
    return pd.concat(rows, ignore_index=True), pd.DataFrame(stats_rows)


# ====================================================================================================
# B. difference distributions
# ====================================================================================================
def build_differences(annual):
    """coherent minus direct, long form, with the parity-capped and raw variants side by side"""
    rows = []
    for ind in INDICATORS:
        rows.append(pd.DataFrame({
            'indicator': ind,
            'gid_0': annual['gid_0'], 'country': annual['country'],
            'continent': annual['continent'], 'region': annual['region'],
            'year': annual['year'],
            'diff_parity': annual[f'{ind}_ggi_coherent_parity'] - annual[f'{ind}_ggi_direct'],
            'diff_raw': annual[f'{ind}_ggi_coherent_raw'] - annual[f'{ind}_ggi_direct'],
        }))
    long = pd.concat(rows, ignore_index=True)

    summary = (long.groupby(['indicator', 'continent'])
               .agg(n=('diff_parity', 'size'),
                    mean=('diff_parity', 'mean'), median=('diff_parity', 'median'),
                    sd=('diff_parity', 'std'),
                    q05=('diff_parity', lambda s: s.quantile(0.05)),
                    q95=('diff_parity', lambda s: s.quantile(0.95)),
                    pct_positive=('diff_parity', lambda s: (s > 0).mean() * 100))
               .reset_index())
    return long, summary


# ====================================================================================================
# C. country trend comparison — selection rule fixed in advance
# ====================================================================================================
def select_trend_countries(annual):
    """
    pick countries by mean signed discrepancy, without looking at any figure

    Rule, applied per indicator:
      1. discrepancy_i = mean over 2015-2025 of (coherent_parity - direct) for country i;
      2. rank all countries in the common sample by that value;
      3. take the top N (largest positive), bottom N (largest negative), and the N closest to zero
         by absolute value.
    Ties break on iso3 alphabetically, so the selection is deterministic and reproducible. No
    country is added or removed after inspecting the plots.
    """
    picks = []
    for ind in INDICATORS:
        d = (annual.assign(diff=annual[f'{ind}_ggi_coherent_parity'] - annual[f'{ind}_ggi_direct'])
             .groupby(['gid_0', 'country', 'continent'], as_index=False)['diff'].mean()
             .sort_values(['diff', 'gid_0'], ascending=[False, True], ignore_index=True))

        groups = {
            'largest_positive': d.head(N_PER_GROUP),
            'largest_negative': d.tail(N_PER_GROUP).sort_values(['diff', 'gid_0']),
            'smallest': (d.assign(abs_diff=d['diff'].abs())
                         .sort_values(['abs_diff', 'gid_0'], ignore_index=True)
                         .head(N_PER_GROUP).drop(columns='abs_diff')),
        }
        for label, frame in groups.items():
            picks.append(frame.assign(indicator=ind, group=label,
                                      mean_discrepancy=frame['diff']).drop(columns='diff'))
    return pd.concat(picks, ignore_index=True)


def build_trends(annual, selection):
    """annual trajectories under each definition for the selected countries only"""
    rows = []
    for ind in INDICATORS:
        chosen = selection[selection['indicator'] == ind]
        block = annual[annual['gid_0'].isin(chosen['gid_0'])]
        for name, template in DEFINITIONS.items():
            rows.append(pd.DataFrame({
                'indicator': ind, 'definition': name,
                'gid_0': block['gid_0'], 'country': block['country'],
                'continent': block['continent'], 'year': block['year'],
                'value': block[template.format(ind=ind)],
            }))
    trends = pd.concat(rows, ignore_index=True)
    return trends.merge(selection[['indicator', 'gid_0', 'group', 'mean_discrepancy']],
                        on=['indicator', 'gid_0'], how='left')


# ====================================================================================================
# D. regional robustness
# ====================================================================================================
def build_regional(annual):
    """
    regional means under each definition, on an identical country sample and aggregation rule

    Unweighted country means within World Bank region — the same aggregation the main regional
    figure uses, so the only thing varying across the three panels is the definition.
    """
    rows = []
    for ind in INDICATORS:
        for name, template in DEFINITIONS.items():
            g = (annual.groupby(['region', 'year'], as_index=False)
                 .agg(value=(template.format(ind=ind), 'mean'),
                      n_countries=('gid_0', 'nunique')))
            rows.append(g.assign(indicator=ind, definition=name))
    regional = pd.concat(rows, ignore_index=True)

    # 2015 / 2025 / change / rank per definition
    wide = regional.pivot_table(index=['indicator', 'definition', 'region'],
                                columns='year', values='value')
    table = pd.DataFrame({
        'value_2015': wide[TREND_START],
        'value_2025': wide[TREND_END],
        'abs_change': wide[TREND_END] - wide[TREND_START],
    }).reset_index()
    table['rank_2025'] = (table.groupby(['indicator', 'definition'])['value_2025']
                          .rank(ascending=False, method='min').astype(int))
    table['rank_change'] = (table.groupby(['indicator', 'definition'])['abs_change']
                            .rank(ascending=False, method='min').astype(int))
    return regional, table.sort_values(['indicator', 'definition', 'rank_2025'], ignore_index=True)


def rank_stability(table):
    """where the definition changes the regional ordering — the load-bearing robustness check"""
    rows = []
    for ind in INDICATORS:
        block = table[table['indicator'] == ind]
        pivot = block.pivot_table(index='region', columns='definition',
                                  values=['rank_2025', 'value_2025', 'abs_change'])
        for region in pivot.index:
            ranks = {d: int(pivot[('rank_2025', d)][region]) for d in DEFINITIONS}
            values = {d: float(pivot[('value_2025', d)][region]) for d in DEFINITIONS}
            rows.append({
                'indicator': ind, 'region': region,
                **{f'rank_{d}': ranks[d] for d in DEFINITIONS},
                **{f'value_{d}': values[d] for d in DEFINITIONS},
                'rank_changes': len(set(ranks.values())) > 1,
                'max_rank_shift': max(ranks.values()) - min(ranks.values()),
                'max_value_spread': max(values.values()) - min(values.values()),
            })
    return pd.DataFrame(rows).sort_values(['indicator', 'rank_direct'], ignore_index=True)


# ====================================================================================================
def main():
    stamp = f'{datetime.now():%Y%m%d}'

    annual_all = load_country_year()
    annual, dropped = common_sample(annual_all)
    print(f'common sample: {annual["gid_0"].nunique()} countries '
          f'({len(dropped)} dropped for incomplete coverage: {dropped[:8]}{"..." if len(dropped) > 8 else ""})')

    scatter, scatter_stats = build_scatter(annual)
    diffs, diff_summary = build_differences(annual)
    selection = select_trend_countries(annual)
    trends = build_trends(annual, selection)
    regional, regional_table = build_regional(annual)
    stability = rank_stability(regional_table)

    outputs = {
        'scatter': scatter, 'scatter_stats': scatter_stats,
        'differences': diffs, 'difference_summary': diff_summary,
        'trend_selection': selection, 'trends': trends,
        'regional': regional, 'regional_table': regional_table,
        'rank_stability': stability,
    }
    for name, frame in outputs.items():
        path = OUTDIR / f'{PREFIX}_{name}_{stamp}.csv'
        frame.to_csv(path, index=False)
        print(f'  {path.relative_to(params.ROOT)}  ({len(frame)} rows)')

    print()
    print(scatter_stats.round(4).to_string(index=False))
    print()
    print('regional rank stability (2025):')
    print(stability.round(4).to_string(index=False))



if __name__ == '__main__':
    main()

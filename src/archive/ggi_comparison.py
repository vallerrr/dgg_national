"""
# Created by valler at 05/08/2026
Feature: figure-ready datasets for the coherent-vs-direct GGI appendix.

Builds every table the appendix notebook plots, so the analysis is reproducible without running a
notebook and the plotting code contains no data preparation. `src/analysis_notebooks/05_coherent_ggi_comparison.ipynb`
reads these and draws.

Country selection for the trend panels follows a rule fixed here, applied before any figure is
inspected — see `select_trend_countries`.

    python src/14_coherent_ggi_comparison.py
"""
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

import coherent_ggi
import params

CFG = params.COHERENT_GGI
INDICATORS = CFG['indicators']
# the notebook that owns this module's output, so the folder names its producer
NOTEBOOK = '05_coherent_ggi_comparison'
OUTDIR = params.table_dir(NOTEBOOK)   # what this notebook writes — the folder names its producer
# What it reads: the coherent-GGI series, built by 04_coherent_ggi_figures. An input from an
# earlier stage is read from disk; this notebook's own results are built, not read back.
INDIR = params.table_dir('04_coherent_ggi_figures')
PREFIX = 'ggi_comparison'

SNAPSHOT_YEARS = CFG['focus_years']          # 2015, 2020, 2025
TREND_START, TREND_END = 2015, 2025          # last complete year; 2026 is partial (D18)
N_PER_GROUP = 3                              # countries per discrepancy group in panel C

DEFINITIONS = {
    'coherent_parity': '{ind}_ggi_coherent_parity',
    'coherent_raw': '{ind}_ggi_coherent_raw',
    'direct': '{ind}_ggi_direct',
}

# How countries are aggregated for the robustness check in section D/E. `region` is the World Bank's
# and is the default, because it is what the main regional figure uses; `continent` is plain
# geography and is what `06` and `08` group by (D56). Both are built on every run and stacked into
# one table tagged by a `grouping` column, so the notebook switches with a parameter rather than a
# rerun — the same arrangement ggi_decomposition.GROUPINGS uses.
GROUPINGS = ['region', 'continent']


# ====================================================================================================
# inputs
# ====================================================================================================
def latest(stem):
    matches = sorted(INDIR.glob(f'{stem}_*.csv'))
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
# the discrepancy filter (D43)
# ====================================================================================================
def apply_threshold(annual, cut=None):
    """
    drop country-years where the two constructions of the GGI disagree by more than `cut`

    Where they disagree sharply at least one of them is wrong about that country-year, and nothing
    in either series says which — so the point carries a construction disagreement rather than a
    measured gender gap. Section C of 05_coherent_ggi_comparison is the trial sweep behind the
    default cut (`params.COHERENT_GGI['discrepancy_threshold']`).

    The row is blanked under **every** definition for that indicator, not just the coherent ones:
    a regional mean is only a robustness check if the definitions are compared over the same
    country-years. Indicators are filtered independently, so a country-year can survive for mobile
    and not for internet — the columns are separate.

    Returns (filtered copy, per-indicator report).
    """
    cut = CFG['discrepancy_threshold'] if cut is None else cut
    out, rows = annual.copy(), []
    for ind in INDICATORS:
        d = coherent_ggi.discrepancy(annual, ind)
        cols = [t.format(ind=ind) for t in DEFINITIONS.values()]
        drop = d > cut
        before = annual[cols[0]].notna()
        out.loc[drop, cols] = np.nan
        kept_countries = set(out.loc[out[cols[0]].notna(), 'gid_0'])
        rows.append({
            'indicator': ind, 'cut': cut,
            'n': int(before.sum()), 'dropped': int((drop & before).sum()),
            'kept': int((~drop & before).sum()),
            'kept_pct': 100 * (~drop & before).sum() / max(before.sum(), 1),
            'countries': annual.loc[before, 'gid_0'].nunique(),
            'countries_kept': len(kept_countries),
            'countries_lost_entirely': annual.loc[before, 'gid_0'].nunique() - len(kept_countries),
            'max_disc_kept': float(d[~drop].max()),
        })
    return out, pd.DataFrame(rows)


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
             # a country the discrepancy filter emptied has no mean to rank on; without the drop
             # it sorts to the tail and would be picked as a `largest_negative`
             .dropna(subset=['diff'])
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
def build_regional(annual, grouping='region'):
    """
    group means under each definition, on an identical country sample and aggregation rule

    Unweighted country means within the group — the same aggregation the main regional figure uses,
    so the only thing varying across the three panels is the definition.

    `grouping` is 'region' (the World Bank's, the default and what the main figure uses) or
    'continent'. The grouping column keeps its own name in the output, so the default call returns
    exactly the frame it always did; `main` stacks both under a `grouping` tag.
    """
    if grouping not in GROUPINGS:
        raise ValueError(f'grouping must be one of {GROUPINGS}, got {grouping!r}')
    rows = []
    for ind in INDICATORS:
        for name, template in DEFINITIONS.items():
            col = template.format(ind=ind)
            # count the countries that actually carry a value, not the rows in the block: after
            # the discrepancy filter those differ, and the mean is over the former
            g = (annual.assign(_id=annual['gid_0'].where(annual[col].notna()))
                 .groupby([grouping, 'year'], as_index=False)
                 .agg(value=(col, 'mean'), n_countries=('_id', 'nunique')))
            rows.append(g.assign(indicator=ind, definition=name))
    regional = pd.concat(rows, ignore_index=True)

    # 2015 / 2025 / change / rank per definition
    wide = regional.pivot_table(index=['indicator', 'definition', grouping],
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


def rank_stability(table, grouping='region'):
    """
    where the definition changes the group ordering — the load-bearing robustness check

    `max_value_spread` is the quantity the appendix paragraph quotes: the largest gap between the
    three definitions' 2025 means for one group. `rank_changes` is the stricter test, because a
    ranking can flip on a spread far too small to matter in level terms.
    """
    if grouping not in GROUPINGS:
        raise ValueError(f'grouping must be one of {GROUPINGS}, got {grouping!r}')
    rows = []
    for ind in INDICATORS:
        block = table[table['indicator'] == ind]
        pivot = block.pivot_table(index=grouping, columns='definition',
                                  values=['rank_2025', 'value_2025', 'abs_change'])
        for group in pivot.index:
            ranks = {d: int(pivot[('rank_2025', d)][group]) for d in DEFINITIONS}
            values = {d: float(pivot[('value_2025', d)][group]) for d in DEFINITIONS}
            rows.append({
                'indicator': ind, grouping: group,
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

    # Sections A-B motivate the threshold, so they must not be subject to it; sections C onward are
    # the robustness check, which is only like-for-like on country-years the definitions agree about
    # (D43). The notebook splits exactly here, and main() did not — it ran the regional check on the
    # unfiltered sample, so `ggi_comparison_rank_stability_*.csv` meant one thing when the script
    # wrote it and another when the notebook did, under the same filename.
    filtered, threshold_report = apply_threshold(annual)
    print(f"  discrepancy filter (D43, cut {CFG['discrepancy_threshold']}): "
          + ' | '.join(f"{r.indicator} {r.kept_pct:.1f}% kept, "
                       f"{int(r.countries_lost_entirely)} countries lost"
                       for r in threshold_report.itertuples()))

    scatter, scatter_stats = build_scatter(annual)
    diffs, diff_summary = build_differences(annual)
    selection = select_trend_countries(filtered)
    trends = build_trends(filtered, selection)
    # Both groupings on every run (D56's arrangement): the region tables keep their existing names
    # and schema so nothing downstream moves, and the continent ones are stacked beside them under a
    # `grouping` tag for the notebook to select.
    regional, regional_table = build_regional(filtered)
    stability = rank_stability(regional_table)
    by_grouping = []
    for g in GROUPINGS:
        _, tbl = build_regional(filtered, grouping=g)
        st = rank_stability(tbl, grouping=g)
        by_grouping.append(st.rename(columns={g: 'group'}).assign(grouping=g))
    stability_by_grouping = pd.concat(by_grouping, ignore_index=True)

    outputs = {
        'scatter': scatter, 'scatter_stats': scatter_stats,
        'differences': diffs, 'difference_summary': diff_summary,
        'trend_selection': selection, 'trends': trends,
        'regional': regional, 'regional_table': regional_table,
        'rank_stability': stability,
        'rank_stability_by_grouping': stability_by_grouping,
        'threshold_report': threshold_report,
    }
    for name, frame in outputs.items():
        path = OUTDIR / f'{PREFIX}_{name}_{stamp}.csv'
        frame.to_csv(path, index=False)
        print(f'  {path.relative_to(params.ROOT)}  ({len(frame)} rows)')

    print()
    print(scatter_stats.round(4).to_string(index=False))
    print()
    print('rank stability (2025), both groupings:')
    for g in GROUPINGS:
        blk = stability_by_grouping[stability_by_grouping['grouping'] == g]
        print(f'\n  -- {g} --')
        print(blk.round(4).to_string(index=False))
        for ind in INDICATORS:
            sub = blk[blk['indicator'] == ind]
            print(f'  {ind}: max |spread across definitions| = {sub["max_value_spread"].max():.4f}'
                  f'  |  ranking changes for {int(sub["rank_changes"].sum())} of {len(sub)} groups')



if __name__ == '__main__':
    main()

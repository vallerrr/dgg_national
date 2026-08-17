"""
# Created by valler at 03/08/2026
Feature: coherent digital gender-gap variables derived from the sex-specific predictions.

The paper models female levels, male levels and the GGI as three separate regressions, so the
directly predicted GGI is not constrained to equal predicted-female / predicted-male. This module
derives that ratio explicitly, quantifies how far the two definitions diverge, and ships a clean
analytical file carrying both.

    {ind}_ggi_coherent_raw    = predicted_female / predicted_male     (NaN where the divide is unsafe)
    {ind}_ggi_coherent_parity = min(raw, 1)                           (primary measure)
    {ind}_ggi_direct          = the separately modelled GGI           (retained, never overwritten)

Nothing is divided when the male prediction is missing, zero or near-zero; those rows are flagged
instead. Female and male predictions are never clipped here — the clipping already applied upstream
is measured and reported (see `clipping_report`).

Reads dgg_pipeline's published series read-only; writes only to outputs/tables/.

    python src/13_coherent_ggi.py
"""
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats

import params
import utils

CFG = params.COHERENT_GGI
INDICATORS = CFG['indicators']
STAGE = 'coherent_ggi'
OUTDIR = params.table_dir(STAGE)  # outputs/tables/<stage>/ — the path names the step

NEAR_ZERO = CFG['near_zero']
TOL = CFG['tolerance']
GRAIN = ['gid_0', 'date', 'age_group', 'outcome']
FOCUS_YEARS = CFG['focus_years']


# ====================================================================================================
# 1. load and validate
# ====================================================================================================
def load_predictions():
    """
    the published country-date predictions in long form, validated for uniqueness

    The national models are fitted on the 18+ Facebook variables only, so there is a single age
    group. It is carried as an explicit column so the grain is stated rather than assumed, and so
    an age-disaggregated series can slot in later without changing the key.
    """
    long = pd.read_csv(CFG['source'])
    long = long.drop(columns=[c for c in long.columns if c.startswith('Unnamed')])
    long['age_group'] = CFG['age_group']

    # Yemen's level predictions are degenerate — female internet sits on the zero floor throughout,
    # so every ratio derived from it is meaningless. Dropped here rather than per-figure, matching
    # the upstream convergence analysis and trend.qmd. See params.excluded_countries, D21.
    excluded = long['gid_0'].isin(params.excluded_countries)
    if excluded.any():
        print(f'excluding {sorted(long.loc[excluded, "gid_0"].unique())}: '
              f'{int(excluded.sum())} rows ({long.loc[excluded, "gid_0"].nunique()} countries)')
    return long[~excluded].copy()


def resolve_duplicates(long):
    """
    enforce uniqueness at country-date-age-outcome

    Exact duplicates (every column identical) are dropped silently — they carry no information.
    Rows that share a key but disagree on a value are a data problem, not a tidying problem: they
    are reported and left in place rather than averaged away.
    """
    report = {'rows_in': len(long)}

    exact = long.duplicated(keep='first')
    report['exact_duplicates_dropped'] = int(exact.sum())
    long = long[~exact].copy()

    conflicting = long.duplicated(GRAIN, keep=False)
    report['conflicting_duplicates'] = int(conflicting.sum())
    report['rows_out'] = len(long)

    if conflicting.any():
        print('WARNING: keys with non-identical duplicate rows — NOT aggregated, left as-is:')
        print(long[conflicting].sort_values(GRAIN).to_string(index=False))

    return long, report


def to_wide(long):
    """one row per country-date-age, one column per outcome"""
    wide = long.pivot_table(index=['gid_0', 'date', 'age_group'],
                            columns='outcome', values='predicted').reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={f'{ind}_fm_ratio': f'{ind}_ggi_direct' for ind in INDICATORS})
    wide['year'] = wide['date'].str.slice(0, 4).astype(int)
    wide['month'] = wide['date'].str.slice(5, 7).astype(int)
    return wide


def add_geography(df):
    """attach World Bank region, continent and country name"""
    df = df.merge(utils.load_regions(), left_on='gid_0', right_on='iso3', how='left').drop(columns='iso3')
    df['continent'] = [utils.get_continent_from_iso3(x) for x in df['gid_0']]
    df['country'] = [utils.get_country_name_from_iso3(x) for x in df['gid_0']]
    return df


# ====================================================================================================
# 2. coherent GGI and its flags
# ====================================================================================================
def add_flags(df, ind):
    """
    explicit, inspectable reasons a coherent GGI is absent or suspect

    Each condition gets its own boolean rather than a single categorical, so they can be counted
    independently and combined however an analysis needs.
    """
    female, male, direct = df[f'{ind}_women'], df[f'{ind}_men'], df[f'{ind}_ggi_direct']

    df[f'{ind}_flag_female_missing'] = female.isna()
    df[f'{ind}_flag_male_missing'] = male.isna()
    df[f'{ind}_flag_male_near_zero'] = male.notna() & (male.abs() < NEAR_ZERO)
    df[f'{ind}_flag_level_out_of_range'] = (
        (female.notna() & ((female < 0) | (female > 1))) |
        (male.notna() & ((male < 0) | (male > 1))))
    df[f'{ind}_flag_direct_out_of_range'] = direct.notna() & ((direct < 0) | (direct > 1))

    # the upstream pipeline clips levels into [0, 1]; a value sitting exactly on a bound is the
    # observable trace of that. It is a lower bound on how often clipping bound, since how far
    # past the boundary the unclipped value went is not recoverable from this source.
    df[f'{ind}_flag_level_at_bound'] = (
        female.isin([0.0, 1.0]) | male.isin([0.0, 1.0]))

    return df


def add_coherent_ggi(df, ind):
    """
    female / male, computed only where the divide is safe

    No epsilon substitution: where the male prediction is missing, zero or near-zero the result is
    NaN and the corresponding flag says why. Silently flooring the denominator would manufacture a
    number out of an absence.
    """
    female, male = df[f'{ind}_women'], df[f'{ind}_men']

    safe = female.notna() & male.notna() & (male.abs() >= NEAR_ZERO)
    raw = pd.Series(np.nan, index=df.index, dtype=float)
    raw[safe] = female[safe] / male[safe]

    df[f'{ind}_ggi_coherent_raw'] = raw
    df[f'{ind}_ggi_coherent_parity'] = raw.clip(upper=CFG['parity_cap'])
    df[f'{ind}_flag_raw_above_parity'] = raw > CFG['parity_cap']
    return df


def build_analytical(df):
    for ind in INDICATORS:
        df = add_flags(df, ind)
        df = add_coherent_ggi(df, ind)
    return df


# ====================================================================================================
# 3. quality control and discrepancy statistics
# ====================================================================================================
def quality_control(df):
    """per-indicator counts, central tendency and range of the raw coherent GGI"""
    rows = []
    for ind in INDICATORS:
        raw = df[f'{ind}_ggi_coherent_raw']
        rows.append({
            'indicator': ind,
            'n_obs': len(df),
            'n_valid_coherent_ggi': int(raw.notna().sum()),
            'n_missing_coherent_ggi': int(raw.isna().sum()),
            'n_missing_female': int(df[f'{ind}_flag_female_missing'].sum()),
            'n_missing_male': int(df[f'{ind}_flag_male_missing'].sum()),
            'n_male_near_zero': int(df[f'{ind}_flag_male_near_zero'].sum()),
            'n_levels_outside_0_1': int(df[f'{ind}_flag_level_out_of_range'].sum()),
            'n_direct_outside_0_1': int(df[f'{ind}_flag_direct_out_of_range'].sum()),
            'mean_raw_coherent_ggi': raw.mean(),
            'median_raw_coherent_ggi': raw.median(),
            'prop_above_1': float((raw > 1).sum() / raw.notna().sum()),
            'min_raw_coherent_ggi': raw.min(),
            'max_raw_coherent_ggi': raw.max(),
        })
    return pd.DataFrame(rows)


def clipping_report(df):
    """
    how often clipping binds, given the source is already clipped

    This module never clips the levels. The upstream pipeline does, so what is measurable here is
    how many predictions already sit on a bound — a lower bound on the true incidence.
    """
    rows = []
    for ind in INDICATORS:
        for sex in ['women', 'men']:
            s = df[f'{ind}_{sex}']
            rows.append({
                'indicator': ind, 'sex': sex, 'n_obs': len(s),
                'n_at_lower_bound_0': int((s == 0).sum()),
                'n_at_upper_bound_1': int((s == 1).sum()),
                'pct_at_a_bound': float(((s == 0) | (s == 1)).mean() * 100),
                'n_outside_0_1_in_source': int(((s < 0) | (s > 1)).sum()),
            })
        both = (df[f'{ind}_women'] == 1) & (df[f'{ind}_men'] == 1)
        rows.append({
            'indicator': ind, 'sex': 'both_at_upper_bound', 'n_obs': len(df),
            'n_at_lower_bound_0': np.nan, 'n_at_upper_bound_1': int(both.sum()),
            'pct_at_a_bound': float(both.mean() * 100), 'n_outside_0_1_in_source': 0,
        })
    return pd.DataFrame(rows)


def discrepancy_stats(df, ind, label_fields):
    """signed/absolute differences, RMSE, correlations and threshold shares for one subset"""
    coherent, direct = df[f'{ind}_ggi_coherent_parity'], df[f'{ind}_ggi_direct']
    ok = coherent.notna() & direct.notna()
    c, d = coherent[ok], direct[ok]
    diff = c - d
    absdiff = diff.abs()

    row = {**label_fields, 'indicator': ind, 'n': int(ok.sum())}
    if ok.sum() < 3:
        return {**row, **{k: np.nan for k in
                          ['mean_signed_diff', 'median_signed_diff', 'mean_abs_diff',
                           'median_abs_diff', 'rmse', 'pearson_r', 'spearman_r',
                           'pct_abs_diff_gt_0.01', 'pct_abs_diff_gt_0.05', 'pct_abs_diff_gt_0.10']}}

    row.update({
        'mean_signed_diff': diff.mean(),
        'median_signed_diff': diff.median(),
        'mean_abs_diff': absdiff.mean(),
        'median_abs_diff': absdiff.median(),
        'rmse': float(np.sqrt((diff ** 2).mean())),
        'pearson_r': stats.pearsonr(c, d)[0],
        'spearman_r': stats.spearmanr(c, d)[0],
        'pct_abs_diff_gt_0.01': float((absdiff > 0.01).mean() * 100),
        'pct_abs_diff_gt_0.05': float((absdiff > 0.05).mean() * 100),
        'pct_abs_diff_gt_0.10': float((absdiff > 0.10).mean() * 100),
    })
    return row


def build_discrepancy_tables(df):
    """overall, by continent, and for the focus years"""
    overall = pd.DataFrame([discrepancy_stats(df, ind, {'scope': 'overall', 'group': 'all'})
                            for ind in INDICATORS])

    by_continent = pd.DataFrame([
        discrepancy_stats(g, ind, {'scope': 'continent', 'group': continent})
        for continent, g in df.groupby('continent') for ind in INDICATORS
    ]).sort_values(['indicator', 'group'], ignore_index=True)

    by_year = pd.DataFrame([
        discrepancy_stats(df[df['year'] == y], ind, {'scope': 'year', 'group': str(y)})
        for y in FOCUS_YEARS for ind in INDICATORS
    ]).sort_values(['indicator', 'group'], ignore_index=True)

    return pd.concat([overall, by_continent, by_year], ignore_index=True)


# ====================================================================================================
# 4. country-year and regional summaries
# ====================================================================================================
def build_country_year(df):
    """
    annual series, aggregated both ways so the two definitions can be compared

    `mean_of_ratios` averages the monthly ratios — what the existing trend figures do.
    `ratio_of_means` averages the monthly levels first, then divides. They differ by Jensen's
    inequality, so both are emitted and the gap is reported in the diagnostics.
    """
    keys = ['gid_0', 'country', 'region', 'continent', 'age_group', 'year']
    levels = [f'{ind}_{s}' for ind in INDICATORS for s in ('women', 'men')]
    direct = [f'{ind}_ggi_direct' for ind in INDICATORS]
    ratios = [f'{ind}_ggi_coherent_raw' for ind in INDICATORS]

    annual = df.groupby(keys, as_index=False)[levels + direct + ratios].mean()
    annual = annual.rename(columns={c: c.replace('_coherent_raw', '_coherent_mean_of_ratios_raw')
                                    for c in ratios})

    for ind in INDICATORS:
        mor = annual[f'{ind}_ggi_coherent_mean_of_ratios_raw']
        annual[f'{ind}_ggi_coherent_mean_of_ratios'] = mor.clip(upper=CFG['parity_cap'])

        male = annual[f'{ind}_men']
        safe = annual[f'{ind}_women'].notna() & male.notna() & (male.abs() >= NEAR_ZERO)
        rom = pd.Series(np.nan, index=annual.index, dtype=float)
        rom[safe] = annual.loc[safe, f'{ind}_women'] / male[safe]
        annual[f'{ind}_ggi_coherent_ratio_of_means_raw'] = rom
        annual[f'{ind}_ggi_coherent_ratio_of_means'] = rom.clip(upper=CFG['parity_cap'])

    return annual


def load_adult_population():
    """
    18+ population by country-year, for population-weighted regional aggregates

    Returned as the UN file has it; the year a given row actually uses is decided by the shared
    rule in `build_region_year` (D32). See doc/data_updates.md: this file is a refresh candidate.
    """
    pop = pd.read_csv(params.UN_POP_PROCESSED,
                      usecols=['iso3', 'Year', '18_inf_m', '18_inf_f'])
    pop['adult_pop'] = pop['18_inf_m'] + pop['18_inf_f']
    return pop.rename(columns={'iso3': 'gid_0', 'Year': 'year'})[['gid_0', 'year', 'adult_pop']]


def build_region_year(annual):
    """regional means, unweighted and weighted by adult population"""
    # One year rule, shared with 02, 01 and 16 (D32): the year's own population when the UN file
    # has it, otherwise the latest earlier year. Supersedes the earlier choice to leave post-2024
    # weights absent — `pop_year` makes every carried weight visible instead of implicit.
    annual = utils.join_latest_available_year(annual, load_adult_population(),
                                              key='gid_0', used_year_col='pop_year')
    carried = utils.carried_year_report(annual, used_year_col='pop_year')
    if len(carried):
        print(f'adult population carried forward for {len(carried)} country-years '
              f'({sorted(carried["year"].unique())})')
    measures = [f'{ind}_ggi_{v}' for ind in INDICATORS
                for v in ('direct', 'coherent_mean_of_ratios', 'coherent_ratio_of_means')]

    rows = []
    for (region, year), grp in annual.groupby(['region', 'year']):
        row = {'region': region, 'year': year, 'n_countries': len(grp),
               'n_countries_with_pop': int(grp['adult_pop'].notna().sum())}
        weights = grp['adult_pop']
        for m in measures:
            row[f'{m}_unweighted'] = grp[m].mean()
            ok = weights.notna() & grp[m].notna()
            row[f'{m}_pop_weighted'] = (np.average(grp.loc[ok, m], weights=weights[ok])
                                        if ok.any() else np.nan)
        rows.append(row)

    return pd.DataFrame(rows).sort_values(['region', 'year'], ignore_index=True)


def aggregation_order_diagnostics(annual):
    """how far the two annual aggregation orders actually diverge"""
    rows = []
    for ind in INDICATORS:
        mor = annual[f'{ind}_ggi_coherent_mean_of_ratios']
        rom = annual[f'{ind}_ggi_coherent_ratio_of_means']
        ok = mor.notna() & rom.notna()
        diff = (mor[ok] - rom[ok]).abs()
        rows.append({'indicator': ind, 'n_country_years': int(ok.sum()),
                     'mean_abs_diff_mor_vs_rom': diff.mean(),
                     'median_abs_diff_mor_vs_rom': diff.median(),
                     'max_abs_diff_mor_vs_rom': diff.max()})
    return pd.DataFrame(rows)


# ====================================================================================================
# 5. assertions
# ====================================================================================================
def run_assertions(df, source_long, dup_report):
    """fail loudly rather than ship a table that quietly violates its own definition"""
    checks = []

    def check(name, condition, detail=''):
        checks.append({'check': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(f'{name} FAILED — {detail}')

    for ind in INDICATORS:
        raw = df[f'{ind}_ggi_coherent_raw']
        female, male = df[f'{ind}_women'], df[f'{ind}_men']
        safe = raw.notna()

        recomputed = female[safe] / male[safe]
        worst = float((raw[safe] - recomputed).abs().max()) if safe.any() else 0.0
        check(f'{ind}: coherent GGI equals female/male within {TOL}', worst <= TOL, f'max deviation {worst:.2e}')

        parity = df[f'{ind}_ggi_coherent_parity']
        check(f'{ind}: parity-capped GGI never exceeds 1',
              bool((parity.dropna() <= CFG['parity_cap'] + TOL).all()),
              f'max {parity.max()}')

        check(f'{ind}: coherent GGI absent exactly where the divide is unsafe',
              bool((raw.isna() == (female.isna() | male.isna() | (male.abs() < NEAR_ZERO))).all()))

        # the source is pre-clipped; confirm the originals survive untouched
        original = source_long.loc[source_long['outcome'] == f'{ind}_fm_ratio',
                                   ['gid_0', 'date', 'predicted']]
        merged = df[['gid_0', 'date', f'{ind}_ggi_direct']].merge(original, on=['gid_0', 'date'], how='inner')
        drift = float((merged[f'{ind}_ggi_direct'] - merged['predicted']).abs().max())
        check(f'{ind}: directly predicted GGI unchanged from source', drift <= TOL, f'max drift {drift:.2e}')

    expected = dup_report['rows_out'] // len(source_long['outcome'].unique())
    check('row count matches the country-date-age grain',
          len(df) == expected, f'{len(df)} rows vs {expected} expected')
    check('grain is unique', not df.duplicated(['gid_0', 'date', 'age_group']).any())

    return pd.DataFrame(checks)


# ====================================================================================================
# 5. output
# ====================================================================================================
def write(frame, name, stamp, note=None, markdown=True):
    """
    CSV for machines; Markdown alongside it for the write-up

    Markdown is skipped for the analytical file — a 19k-row table is not a publication table.
    """
    csv_path = OUTDIR / f'{name}_{stamp}.csv'
    frame.to_csv(csv_path, index=False)

    if markdown:
        header = f'# {name.replace("_", " ")}\n\nGenerated {stamp} by `src/13_coherent_ggi.py`.\n'
        if note:
            header += f'\n{note}\n'
        (OUTDIR / f'{name}_{stamp}.md').write_text(
            f'{header}\n{utils.to_markdown_table(frame)}\n')

    print(f'  {csv_path.relative_to(params.ROOT)}  ({len(frame)} rows)')
    return frame


def main():
    stamp = f'{datetime.now():%Y%m%d}'

    long = load_predictions()
    long, dup_report = resolve_duplicates(long)
    df = build_analytical(add_geography(to_wide(long)))

    assertions = run_assertions(df, long, dup_report)

    print('writing:')
    write(df, 'coherent_ggi_analytical', stamp, markdown=False)
    qc = write(quality_control(df), 'coherent_ggi_qc', stamp)
    clip = write(clipping_report(df), 'coherent_ggi_clipping_report', stamp,
                 note='This module never clips. The source series is already clipped to [0,1] '
                      'upstream, so these counts are a **lower bound** on how often clipping bound — '
                      'the distance past the boundary is not recoverable from the published data.')
    disc = write(build_discrepancy_tables(df), 'coherent_ggi_discrepancy', stamp)
    write(assertions, 'coherent_ggi_assertions', stamp)
    write(pd.DataFrame([dup_report]), 'coherent_ggi_duplicate_report', stamp)

    annual = build_country_year(df)
    write(annual, 'coherent_ggi_country_year', stamp, markdown=False)
    write(build_region_year(annual), 'coherent_ggi_region_year', stamp, markdown=False)
    agg = write(aggregation_order_diagnostics(annual), 'coherent_ggi_aggregation_order', stamp)

    print()
    print('date coverage:', df['date'].min(), '->', df['date'].max(),
          f"({df['date'].nunique()} dates, {df['gid_0'].nunique()} countries)")
    print(f'\nall {len(assertions)} assertions passed\n')
    print(qc.round(4).to_string(index=False))
    print()
    print(disc[disc['scope'] == 'overall'].round(4).to_string(index=False))
    print()
    print(clip.to_string(index=False))
    print()
    print(agg.round(6).to_string(index=False))


if __name__ == '__main__':
    main()

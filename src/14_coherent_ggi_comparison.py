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
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import params

CFG = params.COHERENT_GGI
INDICATORS = CFG['indicators']
OUTDIR = params.OUTPUTS / 'tables'
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
# F. validation against the survey ground truth
# ====================================================================================================
# The GGI series is a model output; the DHS/MICS surveys are the only direct observation of the
# same quantity. Two caveats travel with every number in this section:
#
#   1. AGE. The surveys measure 15-49; the models are fitted and predicted on 18+. Part of any gap
#      is that mismatch, not model error. It is not correctable here — no 15-49 prediction exists.
#   2. SAMPLE. Most of these country-years are *in* the fitted panel, so their agreement is fit,
#      not validation. `in_training` splits them, and the held-out rows (chiefly the 2023-2025
#      surveys added in the latest refresh) are the ones that carry evidential weight.
GT_DEFINITIONS = {
    'coherent_parity': '{ind}_ggi_coherent_parity',
    'coherent_raw': '{ind}_ggi_coherent_raw',
    'direct': '{ind}_ggi_direct',
}


def load_groundtruth():
    """the harmonised survey outcomes, newest dated run"""
    matches = sorted(glob(CFG['groundtruth_glob']))
    if not matches:
        raise FileNotFoundError(f'no file matching {CFG["groundtruth_glob"]} — run '
                                'src/02_ground_truth_data_calculation.qmd first')
    gt = pd.read_csv(matches[-1])
    print(f'ground truth: {Path(matches[-1]).name} ({len(gt)} country-years)')
    return gt


def load_training_keys():
    """
    what the final models actually saw, per indicator

    Two levels of "unseen", and they answer different questions. A country-year absent from the
    panel is still an easy prediction if an earlier survey from the same country was fitted — the
    model has seen that country's level. A country absent entirely is the harder test.

    Returns (country_year_keys, country_keys) keyed by indicator.
    """
    pairs, countries = {}, {}
    for ind in INDICATORS:
        panel = pd.read_csv(CFG['training_panels'] / ind / CFG['training_file'],
                            usecols=['iso3', f'{ind}_year'])
        pairs[ind] = set(zip(panel['iso3'], panel[f'{ind}_year'].astype(int)))
        countries[ind] = set(panel['iso3'])
    return pairs, countries


def _flags(gid_0, year, ind, training_keys, training_countries):
    """the two membership flags, computed the same way for the GGI and the level tables"""
    return (
        [(i, y) in training_keys[ind] for i, y in zip(gid_0, year)],
        [i in training_countries[ind] for i in gid_0],
    )


def build_groundtruth_comparison(annual, groundtruth, training_keys, training_countries):
    """one row per indicator-definition-country-year, observed against predicted"""
    rows = []
    for ind in INDICATORS:
        obs_col = f'{ind}_fm_ratio'
        block = groundtruth[['iso3', 'country', 'year', 'survey_type', obs_col]].dropna(subset=[obs_col])
        merged = block.merge(annual, left_on=['iso3', 'year'], right_on=['gid_0', 'year'],
                             how='inner', suffixes=('_gt', ''))
        # Where the predicted female level is exactly on the zero floor, every coherent definition
        # is 0/male = 0 by construction — the clip, not a gap (D19). Flagged, not dropped here, so
        # the tables downstream can report the sample both ways (params.delete_coherent_zero_ctrl).
        zero_floor = merged[f'{ind}_women'] <= CFG['near_zero']
        seen_year, seen_country = _flags(merged['gid_0'], merged['year'], ind,
                                         training_keys, training_countries)
        for name, template in GT_DEFINITIONS.items():
            rows.append(pd.DataFrame({
                'indicator': ind, 'definition': name,
                'gid_0': merged['gid_0'], 'country': merged['country'],
                'continent': merged['continent'], 'region': merged['region'],
                'year': merged['year'], 'survey_type': merged['survey_type'],
                'observed': merged[obs_col], 'predicted': merged[template.format(ind=ind)],
                'in_training': seen_year, 'country_in_training': seen_country,
                'zero_floor': zero_floor.to_numpy(),
            }))
    out = pd.concat(rows, ignore_index=True)
    out['error'] = out['predicted'] - out['observed']
    return out.dropna(subset=['predicted', 'observed'])


def build_groundtruth_levels(annual, groundtruth, training_keys, training_countries):
    """the same comparison on the female and male levels, which is where any ratio error comes from"""
    rows = []
    for ind in INDICATORS:
        # `{ind}_men` names the observed column in the ground truth and the predicted column in
        # the annual series, so the survey side is renamed before merging rather than relying on
        # merge suffixes — a collision here reads as perfect agreement.
        pairs = {'women': (f'{ind}_wom', f'{ind}_women'), 'men': (f'{ind}_men', f'{ind}_men')}
        block = (groundtruth[['iso3', 'country', 'year', 'survey_type']
                             + [g for g, _ in pairs.values()]]
                 .rename(columns={g: f'obs_{sex}' for sex, (g, _) in pairs.items()}))
        pred_cols = ['gid_0', 'year', 'continent'] + [p for _, p in pairs.values()]
        merged = block.merge(annual[pred_cols].drop_duplicates(['gid_0', 'year']),
                             left_on=['iso3', 'year'], right_on=['gid_0', 'year'], how='inner')
        seen_year, seen_country = _flags(merged['gid_0'], merged['year'], ind,
                                         training_keys, training_countries)
        for sex, (_, pred_col) in pairs.items():
            rows.append(pd.DataFrame({
                'indicator': ind, 'sex': sex,
                'gid_0': merged['gid_0'], 'country': merged['country'],
                'continent': merged['continent'], 'year': merged['year'],
                'survey_type': merged['survey_type'],
                'observed': merged[f'obs_{sex}'], 'predicted': merged[pred_col],
                'in_training': seen_year, 'country_in_training': seen_country,
            }))
    out = pd.concat(rows, ignore_index=True)
    out['error'] = out['predicted'] - out['observed']
    return out.dropna(subset=['predicted', 'observed'])


def _accuracy(frame, value='error', obs='observed', pred='predicted'):
    """the metric set used for every validation table here"""
    e = frame[value]
    # R2 against the 1:1 line, not against a fitted line: 1 - SS(pred - obs) / SS(obs - mean obs).
    # This is the one that answers "does the model beat just predicting the sample mean", and it
    # penalises bias and scale error, so it can go negative. `pearson_r` below is the association
    # alone and stays high even when the predictions are systematically off.
    ss_res = float((e ** 2).sum())
    ss_tot = float(((frame[obs] - frame[obs].mean()) ** 2).sum())
    row = {
        'n': len(frame),
        'bias': e.mean(),                       # signed: positive = model above the survey
        'mae': e.abs().mean(),
        'rmse': float(np.sqrt((e ** 2).mean())),
        'r2': 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        'within_0.05': (e.abs() <= 0.05).mean() * 100,
        'within_0.10': (e.abs() <= 0.10).mean() * 100,
    }
    if len(frame) >= 3 and frame[obs].nunique() > 1 and frame[pred].nunique() > 1:
        row['pearson_r'] = stats.pearsonr(frame[obs], frame[pred])[0]
        row['spearman_r'] = stats.spearmanr(frame[obs], frame[pred])[0]
    else:
        row['pearson_r'] = row['spearman_r'] = np.nan
    return row


def summarise_groundtruth(comparison):
    """
    accuracy by indicator and definition, over each sample the reader might reasonably want

    `sample` crosses the training split with the zero-floor filter, and carries the same labels as
    `summarise_groundtruth_levels()` so the two can be charted side by side. `excl_zero` variants
    appear only when params.delete_coherent_zero_ctrl is on, and are the answer to "how does the
    coherent GGI do where it is defined at all" — not to "which definition is better" (D34).
    """
    splits = [('all', lambda g: g),
              ('in_training', lambda g: g[g['in_training']]),
              ('held_out', lambda g: g[~g['in_training']]),
              ('held_out_new_country', lambda g: g[~g['country_in_training']])]
    if params.delete_coherent_zero_ctrl:
        splits += [('all_excl_zero', lambda g: g[~g['zero_floor']]),
                   ('in_training_excl_zero', lambda g: g[g['in_training'] & ~g['zero_floor']]),
                   ('held_out_excl_zero', lambda g: g[~g['in_training'] & ~g['zero_floor']])]

    rows = []
    for (ind, defn), g in comparison.groupby(['indicator', 'definition']):
        for label, select in splits:
            sub = select(g)
            if len(sub):
                rows.append({'indicator': ind, 'definition': defn, 'sample': label,
                             'n_dropped_zero_floor': int(g['zero_floor'].sum())
                             if label.endswith('_excl_zero') else 0,
                             **_accuracy(sub)})
    return pd.DataFrame(rows)


def summarise_groundtruth_levels(levels):
    """accuracy of the predicted levels, by indicator, sex and sample"""
    splits = [('all', lambda g: g),
              ('in_training', lambda g: g[g['in_training']]),
              ('held_out', lambda g: g[~g['in_training']]),
              ('held_out_new_country', lambda g: g[~g['country_in_training']])]
    rows = []
    for (ind, sex), g in levels.groupby(['indicator', 'sex']):
        for label, select in splits:
            sub = select(g)
            if len(sub):
                rows.append({'indicator': ind, 'sex': sex, 'sample': label, **_accuracy(sub)})
    return pd.DataFrame(rows)


def build_unseen_surveys(comparison, levels):
    """
    one row per unseen survey: both indicators' levels and every GGI definition, side by side

    The surveys added in the 2026-08 refresh are the only rows the models never saw, so this is
    the sheet a reader wants when asking "how did it do on the new data" — country by country,
    rather than as a summary statistic over 17 points.
    """
    held = comparison[~comparison['in_training']]
    wide = held.pivot_table(index=['indicator', 'gid_0', 'country', 'region', 'year',
                                   'survey_type', 'country_in_training'],
                            columns='definition',
                            values=['observed', 'predicted', 'error']).reset_index()
    wide.columns = [c[0] if not c[1] else f'{c[0]}_{c[1]}' for c in wide.columns]
    # `observed` is the survey GGI and does not vary by definition; keep one copy
    obs = [c for c in wide.columns if c.startswith('observed_')]
    wide['observed'] = wide[obs[0]]
    wide = wide.drop(columns=obs)

    lv = (levels[~levels['in_training']]
          .pivot_table(index=['indicator', 'gid_0', 'year'], columns='sex',
                       values=['observed', 'predicted']).reset_index())
    lv.columns = [c[0] if not c[1] else f'{c[0]}_level_{c[1]}' for c in lv.columns]

    out = wide.merge(lv, on=['indicator', 'gid_0', 'year'], how='left')
    return out.sort_values(['indicator', 'year', 'gid_0'], ignore_index=True)


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
    OUTDIR.mkdir(parents=True, exist_ok=True)
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

    # Validation runs on `annual_all`, not the common sample: restricting to countries complete
    # under every definition would drop surveys for no reason that applies here.
    groundtruth = load_groundtruth()
    training_keys, training_countries = load_training_keys()
    gt_comparison = build_groundtruth_comparison(annual_all, groundtruth, training_keys,
                                                 training_countries)
    gt_stats = summarise_groundtruth(gt_comparison)
    gt_levels = build_groundtruth_levels(annual_all, groundtruth, training_keys,
                                         training_countries)
    gt_level_stats = summarise_groundtruth_levels(gt_levels)
    unseen = build_unseen_surveys(gt_comparison, gt_levels)

    outputs = {
        'scatter': scatter, 'scatter_stats': scatter_stats,
        'differences': diffs, 'difference_summary': diff_summary,
        'trend_selection': selection, 'trends': trends,
        'regional': regional, 'regional_table': regional_table,
        'rank_stability': stability,
        'groundtruth': gt_comparison, 'groundtruth_stats': gt_stats,
        'groundtruth_levels': gt_levels, 'groundtruth_level_stats': gt_level_stats,
        'unseen_surveys': unseen,
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

    print()
    print('GGI against the survey ground truth (surveys measure 15-49, models predict 18+):')
    print(gt_stats.round(4).to_string(index=False))
    print()
    print('predicted levels against the survey levels:')
    print(gt_level_stats.round(4).to_string(index=False))


if __name__ == '__main__':
    main()

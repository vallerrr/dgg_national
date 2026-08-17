"""
# Created by valler at 17/08/2026
Feature: leave-one-country-out performance for every national model variant, and a check that each
one reproduces its shipped pickle.

This is the audit that answers "which model are we actually using, on what data, and do our numbers
match the ones published online". It exists because five model variants ship under names that do not
describe what separates them — see params.MODEL_VARIANTS and doc/modelling.md.

Two outputs, both date-stamped:
  * model_performance_<date>.csv   — n, in-sample R2, LOCO R2 and LOCO MAE for every
                                     indicator x variant x outcome
  * model_verification_<date>.csv  — max |coefficient difference| against each shipped pickle,
                                     so a drift in the training data is visible immediately

Read-only with respect to the models: it refits in memory and compares, and writes nothing into
any model directory.

    python src/11_01_model_performance.py
    python src/11_01_model_performance.py --variants online_with_CIS combined_with_CIS
"""
import argparse
from datetime import datetime
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

import params
import utils

CFG = params.FINAL_MODEL
# the notebook that owns this module's output, so the folder names its producer
NOTEBOOK = '02_model_performance'
OUTDIR = params.table_dir(NOTEBOOK)  # outputs/tables/<stage>/ — the path names the step



# ====================================================================================================
# validation against the survey ground truth
# ====================================================================================================
# Moved here from the coherent-GGI stage: predicted-against-observed is model performance, so it
# belongs to this step. Keeping it in stage 04 made stage 03's notebook read stage 04's tables,
# which is a backwards dependency in a pipeline whose numbers are supposed to be its run order.
INDICATORS = CFG['indicators']
CGGI = params.COHERENT_GGI


# The models dgg_pipeline publishes from. Ours are compared against these, never written over.
ONLINE_MODELS = params.EXTERNAL / 'pipeline/model_fit/national/models/OLS'


def load_annual_series():
    """the coherent-GGI country-year table, which carries the predicted levels and every GGI"""
    matches = sorted(params.table_dir('04_coherent_ggi_figures').glob('coherent_ggi_country_year_*.csv'))
    if not matches:
        raise FileNotFoundError('no coherent_ggi_country_year_<date>.csv — run the coherent GGI '
                                'notebook (04_01) first')
    annual = pd.read_csv(matches[-1])
    annual = annual.rename(columns={
        f'{ind}_ggi_coherent_mean_of_ratios': f'{ind}_ggi_coherent_parity' for ind in INDICATORS})
    return annual.rename(columns={
        f'{ind}_ggi_coherent_mean_of_ratios_raw': f'{ind}_ggi_coherent_raw' for ind in INDICATORS})


def load_panel(indicator, keep_itu):
    """
    the fitting panel for one indicator

    `year` is centred on FINAL_MODEL['year_origin'] so the intercept is interpretable at the start
    of the study period. `keep_itu` is the entire sample difference between the model variants: the
    panel carries ITU rows despite its `_itu_deleted` name (D37).
    """
    path = params.PROCESSED / 'combined_data/updated_ground_truth_and_fb' / indicator / CFG['dataset']
    data = pd.read_csv(path)
    data['year'] = data[f'{indicator}_year'] - CFG['year_origin']
    if not keep_itu:
        is_itu = data[f'{indicator}_survey_type'].astype(str).str.contains('itu', case=False, na=False)
        data = data[~is_itu]
    return data


def pooled_r2(y, pred):
    """
    1 - RSS/TSS on the pooled held-out predictions

    Pooled, not the average of per-country R2: with one to three rows per country a per-fold R2 is
    undefined or meaningless. This matches utils.fit_ols_loco's formula, and it is what the
    published performance figure reports.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    return 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def loco(data, spec, outcome, leave_column):
    """
    leave-one-country-out predictions

    Refits with one country held out and predicts that country's rows, so no prediction comes from
    a model that saw the country. The whole point of the model is countries with no survey, which
    is why this is the headline number rather than in-sample R2.
    """
    est = data.dropna(subset=list(spec) + [outcome]).copy()
    preds = pd.Series(index=est.index, dtype=float)
    for held in est[leave_column].unique():
        is_held = est[leave_column] == held
        train, test = est[~is_held], est[is_held]
        reg = sm.OLS(train[outcome], sm.add_constant(train[spec])).fit()
        preds.loc[test.index] = reg.predict(sm.add_constant(test[spec], has_constant='add'))
    return est[outcome], preds


def shipped_pickle(indicator, variant, outcome_var, root):
    return root / params.FINAL_MODEL_FILENAME.format(
        indicator=indicator, model_type=variant, outcome_var=outcome_var)


def evaluate(variants):
    perf, checks = [], []

    for indicator in CFG['indicators']:
        for variant in variants:
            cfg = params.MODEL_VARIANTS[variant]
            data = load_panel(indicator, cfg['keep_itu'])

            for outcome_var in CFG['outcome_vars']:
                # `_align` variants change regressor with the outcome; everything else is fixed
                spec = params.resolve_spec(variant, outcome_var)
                outcome = f'{indicator}_{outcome_var}'
                model = sm.OLS(data[outcome], sm.add_constant(data[spec]), missing='drop').fit()
                y, pred = loco(data, spec, outcome, CFG['leave_column'])

                perf.append({
                    'indicator': indicator, 'variant': variant, 'outcome': outcome_var,
                    'n': int(model.nobs), 'k': len(spec),
                    'keep_itu': cfg['keep_itu'], 'spec': ' + '.join(spec),
                    'r2': float(model.rsquared), 'adj_r2': float(model.rsquared_adj),
                    'loco_r2': float(pooled_r2(y, pred)),
                    'loco_mae': float((y - pred).abs().mean()),
                    'is_production': variant == CFG['model_type'],
                })

                # every variant that ships as a pickle must reproduce it; a non-trivial difference
                # means the training data has moved under the published estimates
                for label, root in (('repo', params.MODELS / CFG['model_folder']),
                                    ('online', ONLINE_MODELS)):
                    path = shipped_pickle(indicator, variant, outcome_var, root)
                    shipped = utils.load_model(path) if path.exists() else None
                    if shipped is None:
                        continue
                    # The registry is an inference from the pickles, so a spec disagreement is a
                    # finding, not a crash: report it and move on.
                    missing = [c for c in shipped.params.index if c not in model.params.index]
                    checks.append({
                        'indicator': indicator, 'variant': variant, 'outcome': outcome_var,
                        'against': label, 'shipped_n': int(shipped.nobs), 'refit_n': int(model.nobs),
                        'n_matches': int(shipped.nobs) == int(model.nobs),
                        'spec_matches': not missing,
                        'shipped_only_terms': ' + '.join(missing),
                        'max_abs_coef_diff': np.nan if missing else float(
                            np.abs(model.params[shipped.params.index] - shipped.params).max()),
                    })

    return pd.DataFrame(perf), pd.DataFrame(checks)


def figure_table(perf):
    """the published performance figure, as a table: LOCO R2 by outcome x variant"""
    block = perf[perf['variant'].isin(params.FIGURE_VARIANTS)]
    wide = block.pivot_table(index=['indicator', 'outcome'], columns='variant', values='loco_r2')
    return wide.reindex(columns=params.FIGURE_VARIANTS).round(3)


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
    matches = sorted(glob(CGGI['groundtruth_glob']))
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
        panel = pd.read_csv(CGGI['training_panels'] / ind / CGGI['training_file'],
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
        zero_floor = merged[f'{ind}_women'] <= CGGI['near_zero']
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


def main(variants):
    perf, checks = evaluate(variants)
    stamp = f'{datetime.now():%Y%m%d}'

    print('=' * 100)
    print('PERFORMANCE — LOCO R2 is the headline; in-sample R2 is shown only for contrast')
    print('=' * 100)
    print(perf[['indicator', 'variant', 'outcome', 'n', 'k', 'r2', 'loco_r2', 'loco_mae']]
          .round(3).to_string(index=False))

    if len(checks):
        print()
        print('=' * 100)
        print('VERIFICATION against the shipped pickles')
        print('=' * 100)
        worst = checks['max_abs_coef_diff'].max()
        print(f"{len(checks)} comparisons | sample sizes match: {bool(checks['n_matches'].all())} "
              f"| specs match: {bool(checks['spec_matches'].all())} "
              f"| max |coefficient difference|: {worst:.2e}")
        bad = checks[(~checks['n_matches']) | (~checks['spec_matches'])
                     | (checks['max_abs_coef_diff'] > 1e-8)]
        if len(bad):
            print('\nDISCREPANCIES — the training data has moved under these models:')
            print(bad.to_string(index=False))

    print()
    print('=' * 100)
    print('THE PUBLISHED FIGURE, AS A TABLE (LOCO R2)')
    print('=' * 100)
    print(figure_table(perf).to_string())

    # the survey validation: predicted against observed, and the unseen-survey sheet
    annual = load_annual_series()
    groundtruth = load_groundtruth()
    training_keys, training_countries = load_training_keys()
    gt = build_groundtruth_comparison(annual, groundtruth, training_keys, training_countries)
    gt_stats = summarise_groundtruth(gt)
    gt_levels = build_groundtruth_levels(annual, groundtruth, training_keys, training_countries)
    gt_level_stats = summarise_groundtruth_levels(gt_levels)
    unseen = build_unseen_surveys(gt, gt_levels)

    print()
    print('=' * 100)
    print('AGAINST THE SURVEY GROUND TRUTH (surveys measure 15-49, models predict 18+)')
    print('=' * 100)
    print(gt_stats.round(4).to_string(index=False))
    print()
    print('predicted levels against the survey levels:')
    print(gt_level_stats.round(4).to_string(index=False))

    outputs = (('model_performance', perf), ('model_verification', checks),
               ('model_performance_groundtruth', gt),
               ('model_performance_groundtruth_stats', gt_stats),
               ('model_performance_groundtruth_levels', gt_levels),
               ('model_performance_groundtruth_level_stats', gt_level_stats),
               ('model_performance_unseen_surveys', unseen))
    for name, frame in outputs:
        if len(frame):
            path = OUTDIR / f'{name}_{stamp}.csv'
            frame.to_csv(path, index=False)
            print(f'\nwrote {path.relative_to(params.ROOT)}  ({len(frame)} rows)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variants', nargs='+', default=list(params.MODEL_VARIANTS),
                        choices=list(params.MODEL_VARIANTS),
                        help='model variants to evaluate (default: all)')
    main(parser.parse_args().variants)

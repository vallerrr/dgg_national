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
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import params
import utils

CFG = params.FINAL_MODEL
OUTDIR = params.OUTPUTS / 'tables'

# The models dgg_pipeline publishes from. Ours are compared against these, never written over.
ONLINE_MODELS = params.EXTERNAL / 'pipeline/model_fit/national/models/OLS'


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


def main(variants):
    perf, checks = evaluate(variants)
    OUTDIR.mkdir(parents=True, exist_ok=True)
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

    for name, frame in (('model_performance', perf), ('model_verification', checks)):
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

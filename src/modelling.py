"""
# Created by valler at 02/08/2026
Feature: fit the final (production) national DGG models from scratch.

This is the model dgg_pipeline/src/modelling/national_model.py loads to produce the published
estimates: an OLS of the gender-gap outcome on the 18+ Facebook ratios plus HDI, GDI, logged GDP
per capita and a linear year term, fitted on the ITU-deleted pooled panel.

Alongside the six fitted models it produces the two artefacts the pipeline consumes:
  * leave-one-country-out (LOCO) predictions — the headline validation, since the model exists to
    predict countries with no survey;
  * the error-estimation betas, a non-negative least squares fit of |LOCO error| used downstream
    to attach an uncertainty band to each prediction.

Writes to a date-stamped directory and never overwrites the shipped models. Run with --verify to
check the refit against them instead of writing anything.

    python src/05_fit_final_models.py --verify
    python src/05_fit_final_models.py
"""
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import nnls
from sklearn.metrics import r2_score

import params
import utils

CFG = params.FINAL_MODEL
SPEC = CFG['spec']


def load_training_data(indicator):
    """
    the fitting panel for one indicator, with the `year` regressor built

    Rows are country-survey observations; a country contributes one row per survey year, so the
    panel is unbalanced. `year` is centred on params.FINAL_MODEL['year_origin'] so the intercept
    is interpretable at the start of the study period.
    """
    path = params.PROCESSED / 'combined_data/updated_ground_truth_and_fb' / indicator / CFG['dataset']
    data = pd.read_csv(path)
    data['year'] = data[f'{indicator}_year'] - CFG['year_origin']
    return data


def fit_full_model(data, indicator, outcome_var):
    """OLS on the complete cases for this outcome"""
    y = data[f'{indicator}_{outcome_var}']
    X = sm.add_constant(data[SPEC])
    return sm.OLS(y, X, missing='drop').fit()


def fit_loco(data, indicator, outcome_var):
    """
    leave-one-country-out predictions

    Refits the model with one country held out and predicts that country's rows, so every
    prediction comes from a model that never saw the country. Returns the held-out predictions
    aligned to the estimation sample.
    """
    leave_column = CFG['leave_column']
    est = data.dropna(subset=SPEC + [f'{indicator}_{outcome_var}']).copy()

    records = []
    for held_out in est[leave_column].unique():
        is_held = est[leave_column] == held_out
        train, test = est[~is_held], est[is_held]

        reg = sm.OLS(train[f'{indicator}_{outcome_var}'], sm.add_constant(train[SPEC])).fit()
        pred = reg.predict(sm.add_constant(test[SPEC], has_constant='add'))

        for idx, value in pred.items():
            records.append({'index': idx,
                            leave_column: held_out,
                            'true': est.loc[idx, f'{indicator}_{outcome_var}'],
                            'pred': value})

    loco = pd.DataFrame(records).set_index('index')
    return est.join(loco[['true', 'pred']])


def fit_error_betas(loco):
    """
    non-negative least squares of the absolute LOCO error on the model spec

    Non-negativity keeps the fitted uncertainty band positive for every country, which a plain OLS
    on absolute errors would not guarantee.
    """
    frame = loco.dropna(subset=['pred']).copy()
    frame['abs_diff'] = np.abs(frame['true'] - frame['pred'])

    coefficients, _ = nnls(frame[SPEC].to_numpy(dtype=float), frame['abs_diff'].to_numpy(dtype=float))
    predicted_error = frame[SPEC].to_numpy(dtype=float) @ coefficients

    return coefficients, {
        'r2': r2_score(frame['abs_diff'], predicted_error),
        'mean_abs_diff': frame['abs_diff'].mean(),
        'loco_r2': r2_score(frame['true'], frame['pred']),
    }


def shipped_model_path(indicator, outcome_var):
    filename = params.FINAL_MODEL_FILENAME.format(
        indicator=indicator, model_type=CFG['model_type'], outcome_var=outcome_var)
    return params.MODELS / CFG['model_folder'] / filename


def main(verify_only):
    outdir = params.MODELS / f"{CFG['model_folder']}_refit_{datetime.now():%Y%m%d}"
    if not verify_only:
        outdir.mkdir(parents=True, exist_ok=True)

    summary, betas = [], []

    for indicator in CFG['indicators']:
        data = load_training_data(indicator)

        for outcome_var in CFG['outcome_vars']:
            model = fit_full_model(data, indicator, outcome_var)
            loco = fit_loco(data, indicator, outcome_var)
            coefficients, stats = fit_error_betas(loco)

            row = {'model': f'{indicator}_{outcome_var}',
                   'n': int(model.nobs),
                   'r2': model.rsquared,
                   'adj_r2': model.rsquared_adj,
                   'loco_r2': stats['loco_r2'],
                   'mean_abs_diff': stats['mean_abs_diff']}

            # compare against the shipped pickle so a drift in the inputs is visible immediately
            shipped = utils.load_model(shipped_model_path(indicator, outcome_var))
            if shipped is not None:
                row['max_abs_coef_diff'] = float(np.abs(model.params[shipped.params.index] - shipped.params).max())
                row['shipped_r2'] = shipped.rsquared
            summary.append(row)

            betas.append({'model': f'{indicator}_{outcome_var}',
                          'r2': stats['r2'],
                          'mean_abs_diff': stats['mean_abs_diff'],
                          **dict(zip(SPEC, coefficients))})

            if not verify_only:
                model.save(outdir / shipped_model_path(indicator, outcome_var).name)

    df_summary = pd.DataFrame(summary)
    df_betas = pd.DataFrame(betas)

    print(df_summary.to_string(index=False))
    print()
    print(df_betas.to_string(index=False))

    if verify_only:
        print(f'\n[verify] nothing written. Refit vs shipped models: '
              f"max coefficient difference {df_summary.get('max_abs_coef_diff', pd.Series([np.nan])).max():.2e}")
        return

    stamp = f'{datetime.now():%Y%m%d}'
    df_summary.to_csv(outdir / f'model_summary_{stamp}.csv', index=False)
    df_betas.to_csv(outdir / f"{CFG['model_type']}_{CFG['leave_column']}_model_error_estimation_betas_{stamp}.csv", index=False)
    print(f'\nwritten to {outdir}')
    print('These are a refit, not the shipped models. Promote them into '
          f"{params.MODELS / CFG['model_folder']} deliberately, not by default.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true',
                        help='compare the refit against the shipped models without writing anything')
    main(parser.parse_args().verify)

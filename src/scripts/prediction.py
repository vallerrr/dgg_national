"""
# Created by valler at 27/11/2024
Feature: predict the monthly data with the 18+ models

"""
import os
from operator import index

import pandas as pd
import params
from params import latest_contrl,missing_contrl,rolling_contrl,del_country_contrl,model_folder
import utils

df_fb_data = pd.read_csv(params.PROCESSED / f'fb_data/fb_all_sd_monthly{"_rolling" if rolling_contrl else ""}.csv')


# The final model, matching dgg_pipeline/src/modelling/national_model.py — an OLS
# (`combined_with_CIS`) in models/OLS, not the archived random forests in models/18_plus.
# dgg_pipeline is canonical for the published series; this script is the local equivalent.
model_folder = params.FINAL_MODEL['model_folder']
model_type = params.FINAL_MODEL['model_type']
leave_column = params.FINAL_MODEL['leave_column']

for year in range(2015, 2025):

    for month in range(1, 13):
        date = f'{year}-{str(month).zfill(2)}'
        if not os.path.isfile(params.RESULTS / f'pred_by_year_and_month/combined_with_CIS/{date}.csv'):

            if year == 2024 and month > 6:
                break
            else:

                # the 18_plus datasets carry only the 18+ bands; params.fb_vars is the full-age set
                fb_cols = params.fb_vars_18_plus

                df_result = pd.DataFrame(columns=['gid_0', 'outcome', 'true', 'predicted', 'date'])


                for indicator in ['internet', 'mobile']:
                    for outcome_var in ['ggi', 'wom', 'men']:

                        outcome = f'{indicator}_{outcome_var}'
                        # read file based on year
                        df_model = pd.read_csv(params.PROCESSED / f'combined_data/updated_ground_truth_and_fb/{indicator}/year_align/combined_multiple_years_no_missing_fb_aligned_{round(year)}.csv')
                        if year >= 2019:
                            df_model.drop(columns=fb_cols, inplace=True)
                            df_fb_data_selected = df_fb_data.loc[(df_fb_data['survey_year'] == year) & (df_fb_data['month'] == month)]
                            df_model = df_model.merge(df_fb_data_selected, left_on=['iso3', 'align_year'], right_on=['iso3', 'survey_year'], how='left')


                        # the `year` regressor the model was fitted with (see params.FINAL_MODEL)
                        df_model['year'] = df_model['align_year'] - params.FINAL_MODEL['year_origin']

                        # load the model
                        model_filename = params.FINAL_MODEL_FILENAME.format(
                            indicator=indicator, model_type=model_type, outcome_var=outcome_var)
                        model_filepath = params.MODELS / f'{model_folder}' / model_filename

                        temp = pd.DataFrame(columns=['gid_0','outcome','true', 'predicted','date'])
                        model = utils.load_model(model_filepath)
                        if model is None:
                            raise FileNotFoundError(f'model not found: {model_filepath}')

                        # statsmodels results carry the spec (incl. the intercept) on .params,
                        # unlike sklearn's .feature_names_in_ the random forests exposed
                        model_spec = model.params.index.to_list()
                        df_model['const'] = 1

                        df_model = df_model.groupby(['iso3']).mean(numeric_only=True).reset_index()  # some countries have multiple entries but the results are the same
                        df_model = df_model.dropna(subset=model_spec).reset_index(drop=True)
                        countries = df_model['iso3'].tolist()
                        X = df_model[model_spec]
                        y_pred = model.predict(X)

                        for ind in range(len(y_pred)):
                            true_value = df_model.loc[(df_model[f'{indicator}_year'] == year) & (df_model[f'iso3'] == countries[ind]), f'{indicator}_{outcome_var}']
                            true_value = true_value.values[0] if len(true_value) > 0 else None
                            temp.loc[len(temp), :] = [countries[ind],outcome, true_value, y_pred[ind], date]


                        # now get the predicted errors
                        df_error_beta = pd.read_csv(params.RESULTS / f'ols_predicted_by_year/{model_type}_country_model_error_estimation_betas.csv')
                        coefficients = df_error_beta.loc[(df_error_beta['model'] == outcome)].values[0][3:]

                        error_spec = df_error_beta.columns.to_list()[3:]
                        temp['predicted_error'] = df_model[error_spec] @ coefficients


                        df_result = pd.concat([df_result, temp], axis=0)

                df_result = df_result[['gid_0', 'outcome', 'predicted', 'predicted_error', 'date']]

                df_result['outcome'] = df_result['outcome'].str.replace('ggi', 'fm_ratio')
                for col in ['predicted', 'predicted_error']:
                    # only three decimal places
                    df_result[col] = df_result[col].apply(lambda x: round(x, 3))
                sanction_countries = params.sactioned_countries
                df_result.to_csv(params.RESULTS / f'pred_by_year_and_month/combined_with_CIS/{date}.csv', index=False)

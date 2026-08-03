"""
# Created by valler at 18/09/2024
Feature: 

"""

import pandas as pd
import params
from pathlib import Path
import warnings
import re
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import utils
warnings.filterwarnings('ignore')



def find_aligned_year(background_var,row):
    """
    find the aligned year based on the country & outcome var year
    """
    iso3 = row['iso3']
    year = align_year

    c = df['iso3']==iso3
    if f'{background_var}_{year}' in list(df.columns):
        target = df[c][f'{background_var}_{year}'].drop_duplicates()
        if len(target) > 0:
            if pd.notnull(target.values[0]):
                return [target.values[0], int(year)]
            else:
                return [None, None]
        else:
            return [None, None]
    else:
        return [None, None]

def find_latest_year(background_var, row):
    """
    if the value in the df is None, find the latest_year (relative to the variable year)
    """
    iso3 = row['iso3']
    year = align_year

    c = df['iso3'] == iso3

    latest_val = None
    years = list(range(2015, year+1))
    years.reverse()

    for latest_year in years:
        latest_year = str(round(latest_year))
        if f'{background_var}_{latest_year}' in df.columns:
            target = df[c][f'{background_var}_{latest_year}'].drop_duplicates()
            if len(target) > 0:
                if pd.notnull(target.values[0]):
                    latest_val = target.values[0]
                    # print(latest_year,latest_val)
                    return [latest_val, int(latest_year)]

    return [None, None]

def find_latest_year_upwards(background_var, row):
    """
    last step, if the value in the df is None, find the most recent year (relative to the variable year, bigger than most recent year)
    """
    iso3 = row['iso3']
    year = align_year
    c = df['iso3'] == iso3

    latest_val = None
    years = list(range(year, 2025))

    for latest_year in years:
        latest_year = str(round(latest_year))
        if f'{background_var}_{latest_year}' in df.columns:
            target = df[c][f'{background_var}_{latest_year}'].drop_duplicates()
            if len(target) > 0:
                if pd.notnull(target.values[0]):
                    latest_val = target.values[0]
                    # print(latest_year,latest_val)
                    return [latest_val, int(latest_year)]


    return [None, None]




# ====================================================================================================
# 1. load the data
# ====================================================================================================
latest_contrl = params.latest_contrl
start_year = 2015
end_year = 2025
# read the background variables

data_path = params.RAW / 'offline_predictors.csv'
df = pd.read_csv(data_path)
df_wide = df.pivot(index='iso3',columns='year',values=[x for x in df.columns if x not in ['country','iso3','year']])

# transform to wide format to fit with the most recent matching algorithms in the pipeline
df_wide.columns = [f'{col}_{round(year)}' for col, year in df_wide.columns]
df_wide.reset_index(inplace=True)

df_fb = pd.read_csv(params.FB_PREPROCESSED / 'fb_national_sd_rolling_std_202606.csv')
df_fb = df_fb.groupby(['iso3','year'],group_keys=False).mean(numeric_only=True).reset_index()
df_fb_wide = df_fb.pivot(index='iso3',columns='year',values=[x for x in df_fb.columns if x not in ['iso3','year','month']])
df_fb_wide.columns = [f'{col}_{round(year)}' for col, year in df_fb_wide.columns]

# merge df_fb with df
df = df_wide.merge(df_fb_wide, left_on=['iso3'], right_on=['iso3'], how='left')

df.rename(columns={'country.x': "country"}, inplace=True)
df_outcome = pd.read_csv(params.PROCESSED / f'outcome_vars{"_latest" if latest_contrl else "_multiple_years"}.csv')

all_countries = list(df.iso3.unique())  # 232 countries

indicator = 'mobile'
for align_year in range(start_year, end_year):
    print(align_year)

    bg_cols = params.bg_cols
    background_vars = params.background_vars+params.fb_vars_18_plus
    # -----------------------------------------------------------------------------------------------

    # initialise the df_outcome_align
    df_outcome_align = pd.DataFrame({'iso3': all_countries})  # this code add all the countries
    df_outcome_align = df_outcome_align.merge(df_outcome[['country', 'iso3', f'{indicator}_year', f'{indicator}_survey_type', f'{indicator}_ggi',f'{indicator}_men',f'{indicator}_wom']],left_on='iso3',right_on='iso3',how='left')
    # drop duplicates
    df_outcome_align.drop_duplicates(inplace=True)
    df_outcome_align['align_year'] = [align_year]*len(df_outcome_align)

    # step 1: get the aligned years for all background variables
    for background_var in background_vars:
        df_outcome_align[[background_var, f"{background_var}_year"]] = pd.DataFrame(
            df_outcome_align.apply(lambda x: find_aligned_year(background_var, x), axis=1).tolist(),
            index=df_outcome_align.index)

    if align_year > 2015:

        # step 2: get the most recent year (relative to the var year, smaller than the current year) for None vals
        for background_var in background_vars:
            df_outcome_align[[background_var, f"{background_var}_year"]] = df_outcome_align.apply(
                lambda x: (x[background_var], x[f"{background_var}_year"]) if pd.notnull(x[background_var]) and pd.notnull(x[f"{background_var}_year"])
                else find_latest_year(background_var, x), axis=1).apply(pd.Series)

    # step 3: if the value in the df is None, find the most recent year (relative to the variable year, bigger than most recent year)
    for background_var in background_vars:
        df_outcome_align[[background_var, f"{background_var}_year"]] = df_outcome_align.apply(
                lambda x: (x[background_var], x[f"{background_var}_year"]) if pd.notnull(x[background_var]) and pd.notnull(x[f"{background_var}_year"])
                else find_latest_year_upwards(background_var, x), axis=1).apply(pd.Series)



    df_outcome_align['gdp_pcap'] = np.log(df_outcome_align['gdp_pcap'])
    df_outcome_align['educ_hdi_r'] = np.average(df_outcome_align[['eys_r', 'mys_r']], axis=1)
    df_outcome_align['conti'] = [utils.get_continent_from_iso3(x) for x in df_outcome_align['iso3']]  # continent column


    missing_contrl = True


    # fill by missing values
    if missing_contrl:
        df_imputed = utils.filling_missing_by_continent_mean(df_outcome_align, params.background_vars+['educ_hdi_r'], indicator,del_country=False)

    # order-preserving: see the same fix in 02_00_background_data_for_model.py (§9)
    column_differences = [x for x in df_outcome_align.columns if x not in set(df_imputed.columns)]
    merge_cols = list(dict.fromkeys(column_differences+['iso3',f'{indicator}_year','conti']))
    df_imputed = pd.merge(df_imputed.drop(columns = ['conti']),df_outcome_align[merge_cols],
                          left_on=['iso3',f'{indicator}_year'],
                          right_on=['iso3',f'{indicator}_year'],how='outer')

    # temp = df_imputed[[f'{indicator}_ggi',f'{indicator}_wom',f'{indicator}_men','iso3']]

    for col in background_vars:
        # print(col, df_outcome_align[col].isnull().sum(), df_align_most_recent.dropna(subset=f'{indicator}_year')[col].isnull().sum())
        print(col, df_imputed[col].isnull().sum())


    cols = [f'{indicator}_year','align_year']+[x for x in df_outcome_align.columns if '_year' not in x]
    df_model = df_imputed[cols]

    utils.check_critical_info(df_model, indicator)

    df_imputed.to_csv(params.PROCESSED / f'combined_data/updated_ground_truth_and_fb/{indicator}/year_align/combined_multiple_years_no_missing_fb_aligned_{round(align_year)}_with_year.csv', index=False)
    df_model.to_csv(params.PROCESSED / f'combined_data/updated_ground_truth_and_fb/{indicator}/year_align/combined_multiple_years_no_missing_fb_aligned_{round(align_year)}.csv', index=False)


# # =============================================================================
# check the information
# # =============================================================================
'''
fb_vars = params.fb_vars

def check_critical_info(df_model,indicator):

    coverage = {}
    coverage["online fit coverage"] =len(df_model[df_model[fb_vars + [f'{indicator}_ggi']].notnull()][fb_vars + [f'{indicator}_ggi']].dropna())
    coverage["online pred coverage"] = len(df_model[df_model[fb_vars].notnull()][fb_vars].dropna())

    coverage["offline fit coverage fit"] = len(df_model[df_model[params.bg_cols + [f'{indicator}_ggi']].notnull()][params.bg_cols + [f'{indicator}_ggi']].dropna())
    coverage["offline pred coverage"] = len(df_model[df_model[params.bg_cols].notnull()][params.bg_cols].dropna())

    coverage["combined fit coverage"] = len(df_model[df_model[params.bg_cols + fb_vars + [f'{indicator}_ggi']].notnull()][params.bg_cols + fb_vars + [f'{indicator}_ggi']].dropna())
    coverage["combined pred coverage"] = len(df_model[df_model[params.bg_cols+fb_vars].notnull()][params.bg_cols+fb_vars].dropna())

    coverage["online fit unique country coverage"] = len(df_model[df_model[fb_vars + [f'{indicator}_ggi', 'iso3']].notnull()][fb_vars + [f'{indicator}_ggi', 'iso3']].dropna()['iso3'].unique())
    coverage["online pred coverage unique country coverage"] = len(df_model[df_model[fb_vars+['iso3']].notnull()][fb_vars+['iso3']].dropna()['iso3'].unique())

    coverage["offline fit coverage fit unique country coverage"] = len(df_model[df_model[params.bg_cols + [f'{indicator}_ggi', 'iso3']].notnull()][params.bg_cols + [f'{indicator}_ggi', 'iso3']].dropna()['iso3'].unique())
    coverage["offline pred coverage unique country coverage"] = len(df_model[df_model[params.bg_cols+['iso3']].notnull()][params.bg_cols+['iso3']].dropna()['iso3'].unique())

    coverage["combined fit coverage unique country coverage"] = len(df_model[df_model[params.bg_cols + fb_vars + [f'{indicator}_ggi', 'iso3']].notnull()][params.bg_cols + fb_vars + [f'{indicator}_ggi', 'iso3']].dropna()['iso3'].unique())
    coverage["combined pred coverage unique country coverage"] = len(df_model[df_model[params.bg_cols + fb_vars+['iso3']].notnull()][params.bg_cols + fb_vars+['iso3']].dropna()['iso3'].unique())


    return coverage
df_coverage = pd.DataFrame(columns=['indicator','align_year','online fit coverage', 'online pred coverage', 'offline fit coverage fit', 'offline pred coverage', 'combined fit coverage', 'combined pred coverage', 'online fit unique country coverage', 'online pred coverage unique country coverage', 'offline fit coverage fit unique country coverage', 'offline pred coverage unique country coverage', 'combined fit coverage unique country coverage', 'combined pred coverage unique country coverage'])

for indicator in ['mobile','internet']:
    for year in range(start_year,end_year):
        df_model = pd.read_csv(params.PROCESSED / f'combined_data/updated_ground_truth_and_fb/{indicator}/year_align/combined_multiple_years_no_missing_fb_aligned_{round(year)}.csv')
        coverage_dict = check_critical_info(df_model, indicator)
        df_coverage.loc[len(df_coverage),] = [indicator, year]+list(coverage_dict.values())
'''

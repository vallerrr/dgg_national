"""
# Created by valler at 18/07/2024
Feature: check and generate report about the background data
@28 Dec 2024: we substitue the fb data with the latest version (imputed value for year 2015-2019) and rolling window for data after 2019
"""

import pandas as pd
import params
from pathlib import Path
import warnings
import re
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def find_aligned_year(background_var,row):
    """
    find the aligned year based on the country & outcome var year
    """
    iso3 = row['iso3']
    if pd.isnull(row[f'{indicator}_year']):
        year = 2022  # use the most recent year as the default outcome year
    else:
        year = str(round(row[f'{indicator}_year']))

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
    if pd.isnull(row[f'{indicator}_year']):
        year = 2022  # use the most recent year as the default outcome year
    else:
        year = int(round(row[f'{indicator}_year']))

    c = df['iso3'] == iso3

    latest_val = None
    years = list(range(start_year, year+1))
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
    if pd.isnull(row[f'{indicator}_year']):
        year = 2022  # use the most recent year as the default outcome year
    else:
        year = int(round(row[f'{indicator}_year']))

    c = df['iso3'] == iso3

    latest_val = None
    years = list(range(year, 2023))

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

# read the background variables

data_path = params.RAW / 'offline_predictors.csv'
df = pd.read_csv(data_path)
df_wide = df.pivot(index='iso3',columns='year',values=[x for x in df.columns if x not in ['country','iso3','year']])

# transform to wide format to fit with the most recent matching algorithms in the pipeline
df_wide.columns = [f'{col}_{round(year)}' for col, year in df_wide.columns]
df_wide.reset_index(inplace=True)


#df_fb = pd.read_csv(params.PROCESSED / f'fb_data/fb_all_sd_wide.csv')
df_fb = pd.read_csv(params.FB_PREPROCESSED / 'fb_national_sd_rolling_std_202606.csv')
df_fb = df_fb.groupby(['iso3','year'],group_keys=False).mean(numeric_only=True).reset_index()
df_fb_wide = df_fb.pivot(index='iso3',columns='year',values=[x for x in df_fb.columns if x not in ['iso3','year','month']])
df_fb_wide.columns = [f'{col}_{round(year)}' for col, year in df_fb_wide.columns]

# merge df_fb with df
df = df_wide.merge(df_fb_wide, left_on=['iso3'], right_on=['iso3'], how='left')

df.rename(columns={'country.x': "country"}, inplace=True)
df_outcome = pd.read_csv(params.PROCESSED / f'outcome_vars{"_latest" if latest_contrl else "_multiple_years"}.csv')
background_vars = params.background_vars+params.fb_vars_18_plus


all_countries = list(df.iso3.unique())  # 232 countries
start_year = 2015  # the year to start do downward search

# ====================================================================================================
# 2. check the missing values based on different indicator (outcome)
# ====================================================================================================
# get the missing information of each parameter for each country and indicator
# create a df that records all the latest year for each of the factor and country

# df_outcome_align_internet[['internet_year','internet_survey_type']].value_counts().reset_index().sort_values(by=['internet_year','internet_survey_type'],ignore_index=True)

# ---------------------------------------------------------------------------------------------------------------------
# 2.3 align method 3 (the final version): mix method 1 & 2
# ---------------------------------------------------------------------------------------------------------------------
# create a function that check/use the aligned year with the outcome variable,
# if it doesn't exist, use the latest available year (before 2015)
# do we wish to use future data? (data available after the outcome var)
indicator = 'mobile'
df_outcome_align = pd.DataFrame({'iso3': all_countries})  # this code add all the countries
df_outcome_align = df_outcome_align.merge(df_outcome[['country', 'iso3', f'{indicator}_year', f'{indicator}_survey_type', f'{indicator}_ggi',f'{indicator}_men',f'{indicator}_wom']],left_on='iso3',right_on='iso3',how='left')
# drop duplicates
df_outcome_align.drop_duplicates(inplace=True)
#df_outcome_align.dropna(subset=[f'{indicator}_year'],inplace=True)
# note that if a country has no data for the outcome variable, we will try to align the background variables to the most recent year (2022)
df_outcome_align.reset_index(drop=True,inplace=True)

# step 1: get the aligned years for all background variables
for background_var in background_vars:
    df_outcome_align[background_var] = df_outcome_align.apply(lambda x: find_aligned_year(background_var,x)[0], axis=1)
    df_outcome_align[f"{background_var}_year"] = df_outcome_align.apply(lambda x: find_aligned_year(background_var, x)[1], axis=1)


# step 2: get the most recent year (relative to the var year) for None vals
for background_var in background_vars:
    df_outcome_align[background_var] = df_outcome_align.apply(lambda x: x[background_var] if pd.notnull(x[background_var]) else find_latest_year(background_var,x)[0],axis=1)
    df_outcome_align[f"{background_var}_year"] = df_outcome_align.apply(lambda x: x[f'{background_var}_year'] if pd.notnull(x[f'{background_var}_year']) else find_latest_year(background_var, x)[1], axis=1)

import numpy as np
missing_rates = {}
for col in [x for x in df_outcome_align.columns if 'year' not in x]:
    missing_rates[col] = df_outcome_align[col].isnull().sum()/len(df_outcome_align)
print(np.mean(list(missing_rates.values())))
# average ,0.0864 missing rate without upward matching in internet and mobile

# step 3: if the value in the df is None, find the most recent year (relative to the variable year, bigger than most recent year)
for background_var in background_vars:
    df_outcome_align[background_var] = df_outcome_align.apply(lambda x: x[background_var] if pd.notnull(x[background_var]) else find_latest_year_upwards(background_var,x)[0],axis=1)
    df_outcome_align[f"{background_var}_year"] = df_outcome_align.apply(lambda x: x[f'{background_var}_year'] if pd.notnull(x[f'{background_var}_year']) else find_latest_year_upwards(background_var, x)[1], axis=1)
# average xxx,0.067 missing rate without upward matching in internet and mobile

# 17 countries without FB data ['CHI', 'CSK', 'CUB', 'DDR', 'IRN', 'PRK', 'SAS', 'SCG', 'SDN', 'SUN', 'SYR', 'VDR', 'XTI', 'XXK', 'YMD', 'YUG']
# check missing

for col in background_vars:
    # print(col, df_outcome_align[col].isnull().sum(), df_align_most_recent.dropna(subset=f'{indicator}_year')[col].isnull().sum())
    print(col, df_outcome_align[col].isnull().sum())

# ====================================================================================================
# missing imputation
# ====================================================================================================


missing_contrl = True
del_country_contrl =False
import numpy as np
import utils
# 2.0 new vars
df_outcome_align['gdp_pcap'] = np.log(df_outcome_align['gdp_pcap'])
df_outcome_align['educ_hdi_r'] = np.average(df_outcome_align[['eys_r', 'mys_r']], axis=1)
df_outcome_align['conti'] = [utils.get_continent_from_iso3(x) for x in df_outcome_align['iso3']]  # continent column

# check if every continent,year has a value for all variables
# temp = df_outcome_align.groupby(['conti',f'{indicator}_year']).count()


# fill by missing values
if missing_contrl:
    df_imputed = utils.filling_missing_by_continent_mean(df_outcome_align, params.background_vars+['educ_hdi_r'], indicator, del_country=del_country_contrl)

# order-preserving: a raw set() is both rejected as a pandas indexer and orders columns
# differently every run, which would make the saved csv non-reproducible (§9)
column_differences = [x for x in df_outcome_align.columns if x not in set(df_imputed.columns)]
merge_cols = list(dict.fromkeys(['iso3',f'{indicator}_year','conti']+column_differences))
df_imputed = pd.merge(df_outcome_align[merge_cols],df_imputed.drop(columns=['conti']),
                      left_on=['iso3', f'{indicator}_year'],
                      right_on=['iso3', f'{indicator}_year'],how='left')

# temp = df_imputed[[f'{indicator}_ggi',f'{indicator}_wom',f'{indicator}_men','iso3']]

for col in background_vars:
    # print(col, df_outcome_align[col].isnull().sum(), df_align_most_recent.dropna(subset=f'{indicator}_year')[col].isnull().sum())
    print(col, df_imputed[col].isnull().sum())

# there are 4 rows with missing values: CUB IRN RUS SDN

basic_cols = ['iso3', 'country', f'{indicator}_year', f'{indicator}_survey_type', f'{indicator}_ggi', f'{indicator}_men', f'{indicator}_wom']
cols = basic_cols+[x for x in df_outcome_align.columns if '_year' not in x and x not in set(basic_cols)]
df_model = df_imputed[cols]
df_model.drop(df_model.loc[df_model['iso3'].isin(['CUB', 'IRN', 'RUS', 'SDN']),].index,inplace=True)
utils.check_critical_info(df_model, indicator)
# fit unique country for combined model 215 internet, 127 mobile
# predicted unique country for combined model 215 internet, 215 mobile
"""
df_imputed.to_csv(params.PROCESSED/f'combined_data/updated_ground_truth_and_fb/{indicator}/combined{"_latest" if latest_contrl else "_multiple_years"}{"_no_missing" if missing_contrl else ""}{"_keep_countries" if not del_country_contrl else ""}_fb_aligned_with_year.csv',index=False)
df_model.to_csv(params.PROCESSED/f'combined_data/updated_ground_truth_and_fb/{indicator}/combined{"_latest" if latest_contrl else "_multiple_years"}{"_no_missing" if missing_contrl else ""}{"_keep_countries" if not del_country_contrl else ""}_fb_aligned.csv',index=False)
"""

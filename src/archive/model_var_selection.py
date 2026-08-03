"""
# Created by valler at 13/08/2024
Feature: this script is used to select the features for the DGG model

"""


import utils
import params
import numpy as np

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os

latest_contrl = params.latest_contrl

# ====================================================================================================
# 1. linear regression validation
#       as the linear regression of offline and combine models of Internet and Mobile are the best, we validate them here
# ====================================================================================================
# read data
model_datasets = params.model_datasets
model_specs = {'internet':{'offline':{"var_set":['se_r', 'wdi_fertility', 'pov_hr', 'gdi', 'gggi_ggi', 'ineq_inc', 'wdi_internet', 'wdi_acel', 'wdi_unempr', 'lfpr_r', 'abr', 'hcpi_a', 'bicc_gmi', 'le_r', 'une_girlgpt', 'vdem_gender', 'wbgi_pve', 'hdi_r', 'mys_r', 'une_girlglsr', 'gdp_pcap', 'hdi', 'eys', 'ihdi', 'gni_pc', 'educ_hdi_r', 'wdi_unemp', 'ineq_edu', 'mys', 'gni_pc_r', 'gggi_pos'],
                                      "missing":False,
                                      "year": 2022},
                           'combined':{"var_set":['se_r', 'wdi_fertility', 'wdi_litrad', 'pov_hr', 'fb_30_34', 'fb_25_29', 'une_girlglsr', 'hcpi_a', 'wdi_internet', 'mys', 'wdi_litradr', 'wdi_acel', 'fb_ios_device_users_ratio', 'fb_25_49', 'fb_16_17', 'fb_15_16', 'une_girlgpt', 'bicc_gmi', 'wbgi_pve', 'une_girlgpr', 'vdem_gender', 'fb_40_44', 'fb_20_64', 'fb_18_999', 'wdi_gert'],
                                       "missing":True,
                                       "year": 2024}},

               "mobile":  {'offline':{"var_set":['wdi_litrad', 'wbgi_pve', 'wdi_unemp', 'wdi_internet', 'abr', 'gdi', 'pov_hr', 'hcpi_a', 'gggi_pes', 'une_girlgpr', 'une_girlgpt', 'wdi_fertility', 'ineq_inc', 'gii', 'le_r', 'gni_pc', 'gdp_pcap', 'bicc_gmi', 'wdi_litradr', 'une_girlglst', 'lfpr_r', 'le', 'ineq_le', 'pr_r', 'mmr', 'hdi_r', 'mys_r', 'se_r', 'gggi_eas', 'mys'],
                                      "missing":False,
                                      "year": 2022},

                           'combined':{"var_set":['wdi_litrad', 'wbgi_pve', 'wdi_unemp', 'wdi_internet', 'fb_android_device_users_ratio', 'hcpi_a', 'gii', 'gggi_pes', 'fb_all', 'pov_hr', 'wdi_litradr', 'fb_60_64', 'une_girlgpt', 'wdi_fertility', 'fb_25_999', 'ineq_le', 'wdi_gertr', 'une_girlglst', 'fb_18_999', 'abr', 'une_girlgpr', 'le', 'vdem_gender', 'fb_65_999', 'fb_45_49', 'fb_20_64', 'bicc_gmi', 'lfpr_r', 'hdi_r', 'le_r', 'ineq_inc', 'gggi_eas', 'fb_25_29', 'fb_55_59', 'fb_50_54', 'fb_35_39', 'gdi'],
                                       "missing":False,
                                       "year": 2022}}}


# model fitting and validation
loco_r2_dict = {}
reg_dict = {}
res = "r2a"
for indicator in ['internet', 'mobile']:
    for model_type in ["offline", "combined"]:
        indep_vars = model_specs[indicator][model_type]['var_set']
        filename = f"{params.PROCESSED}/combined_data/updated_ground_truth_and_fb/{indicator}/combined{'_latest' if latest_contrl else '_multiple_years'}{'_no_missing' if model_specs[indicator][model_type]['missing'] else ''}_{model_specs[indicator][model_type]['year']}-06.csv"
        data = pd.read_csv(filename)

        """
        # those lines are trying to align the 2024 prediction of internet combine to the 2022 prediction, as the fb_ios_device_users_ratio is missing in 2022
        if indicator=='internet' and model_type == 'combined':
            temp = pd.read_csv(params.PROCESSED / "combined_data/updated_ground_truth_and_fb/internet/combined_multiple_years_2024-06.csv")
            data.drop(columns=['fb_ios_device_users_ratio'],inplace=True)
            data=data.merge(left_on=['iso3','country',f'{indicator}_year'],right=temp[['iso3','country',f'{indicator}_year',"fb_ios_device_users_ratio"]],right_on=['iso3','country',f'{indicator}_year'],how='left')
        """
        # record sample years
        temp = data[f'{indicator}_year']
        data = data.dropna(subset=[f'{indicator}_ggi'] + indep_vars)
        if 'fb_all' in indep_vars:
            # standardise the fb_all
            data['fb_all'] = (data['fb_all']-min(data['fb_all']))/(max(data['fb_all'])-min(data['fb_all']))
        reg_dict[f"{indicator}_{model_type}"] = utils.fit_ols(data, indep_vars, indicator)

        loco_r2_dict[f"{indicator}_{model_type}"] = utils.fit_ols_loco(data,indep_vars,indicator,result = res)

        if res!="r2":
            loco_r2_dict[f"{indicator}_{model_type}"]['survey_year'] = temp

# confirm the results
# utils.print_models_with_loco(models=list(reg_dict.values()), model_names=list(reg_dict.keys()), loco_r2_dict=loco_r2_dict)

# organise the loco_r2_dict into one dataframe
df_loc_res = pd.DataFrame()
for key, df in loco_r2_dict.items():
    df['name'] = [key.replace('_',' ')]*len(df)
    df_loc_res = pd.concat([df_loc_res, df], axis=0)

df_loc_res['method'] = ['linear regression']*len(df_loc_res)

# ====================================================================================================
# 2. Combine Random Forest results for online models
# ====================================================================================================
df_rf_res = pd.read_csv(params.RESULTS / 'logs/random_forest_loco_results.csv')
df_rf_res = df_rf_res[df_rf_res['name'].str.contains('online')]

# select years
df_rf_res = df_rf_res[((df_rf_res['year'].isin([2022]))&(df_rf_res['name']=='mobile online'))|((df_rf_res['year'].isin([2021]))&(df_rf_res['name']=='internet online'))]
df_rf_res['method'] = ['random forest']*len(df_rf_res)

# add the survey year information back to the df
for indicator,year in [['mobile',2022],['internet',2021]]:

    missing_imputed = False
    model_type = 'online'
    filename = f"{params.PROCESSED}/combined_data/updated_ground_truth_and_fb/{indicator}/combined{'_latest' if latest_contrl else '_multiple_years'}{'_no_missing' if missing_imputed else ''}_{year}-06.csv"
    data = pd.read_csv(filename)
    data.rename(columns={f'{indicator}_ggi': 'true'}, inplace=True)
    df_rf_res = df_rf_res.merge(left_on=['country','true'],right=data[['country',f'{indicator}_year','true']],right_on=['country','true'],how='left')

# rename columns
df_rf_res['survey_year'] = [x if pd.notnull(x) else y for x, y in zip(df_rf_res['mobile_year'], df_rf_res['internet_year'])]

# combine them together
df_res = pd.concat([df_loc_res[['name','country','true','pred_val','method','survey_year']].drop_duplicates().rename(columns={'pred_val':'pred'}),df_rf_res[['name','country','true','pred','method','survey_year']]],axis=0)

# group by model name and country (average countries within the same model name )
# df_res = pd.merge(left=df_res.groupby(['name','country'], as_index=False).mean(),left_on=['name','country'],
#                  right=df_res[['name','country','method']], right_on=['name','country'])  # 865 to 746


df_res.to_csv(params.RESULTS / 'logs/best_models_pred_and_true.csv',index=False)

# ====================================================================================================
# 3. plot
# ====================================================================================================
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

model_performance = {}
# Define a color palette based on the range of survey years
years = sorted(df_res['survey_year'].unique())
palette = sns.color_palette("mako", len(years))
year_color_map = dict(zip(years, palette))

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
fig.suptitle('Best Models: True vs Predicted Values (LOCO design)', fontsize=16)

for i, indicator in enumerate(['internet', 'mobile']):
    for j, model_type in enumerate(["online", "offline", "combined"]):
        df_temp = df_res[(df_res['name'] == f"{indicator} {model_type}")].drop_duplicates()
        ax[i, j].plot([0, 1], [0, 1], ls="--", c="grey")
        scatter = ax[i, j].scatter(x=df_temp['true'], y=df_temp['pred'],
                                   c=df_temp['survey_year'].map(year_color_map),
                                   alpha=0.8)
        ax[i, j].set_title(f"{indicator} {model_type}\n{df_temp['method'].unique()[0]}")
        ax[i, j].set_xlabel('true')
        ax[i, j].set_ylabel('pred')

# Adjust the layout to include the legend
fig.tight_layout(rect=[0, 0.1, 1, 0.97])

# Create custom legend as dots

n_cols = (len(years) + 1) // 2  # Calculate number of columns needed for 2 rows
legend_elements = [mpatches.Patch(color=year_color_map[year], label=str(round(year))) for year in years]
fig.legend(handles=legend_elements, loc='lower center', ncol=n_cols, title='Survey Year', bbox_to_anchor=(0.5, 0.01),frameon=False)

#plt.show()
plt.savefig(params.GRAPHS / 'best_models_pred_and_true.pdf')



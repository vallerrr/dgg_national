"""
# Created by valler at 10/06/2024
Feature: check the new national data file
"""

from pathlib import Path
import pandas as pd
import warnings
import pycountry as pc
import params
import utils
import statsmodels.api as sm
latest_contrl = params.latest_contrl  # use inflated data
cut_off_year = 2015
warnings.filterwarnings('ignore')
data_path = params.RAW / 'groundtruth_offline_predictors.csv'
df = pd.read_csv(data_path)
df.rename(columns={'country.x': "country"}, inplace=True)
country_iso3_code_dict = {country:iso3 for country,iso3 in zip(df['country'],df['iso3'])}
# ====================================================================================================
# 1. preprocess outcome vars
# ====================================================================================================

# ---------------------------------------------------------------------------------------------------
# 1.1 internet outcomes
# for the internet outcomes, please refer to the src/00_
# ---------------------------------------------------------------------------------------------------
# 1.1.0 read data: data is pre-created
df_outcome = pd.read_csv(params.RAW / 'internet_mobile_indicator_clean.csv')
df_internet = df_outcome[[x for x in df_outcome.columns if 'mob' not in x]]
columns_to_exclude = ['n_age_15_to_49_men_valid_internet','n_age_15_to_49_wom_valid_internet','survey_duration']
df_internet.drop(columns=columns_to_exclude, inplace=True)
df_internet.dropna(inplace=True)

# 1.1.1 no gsma
df_internet.drop(df_internet.loc[df_internet['survey_type'].str.contains('gsma')].index, inplace=True)  # 170 data points

# 1.1.2 save same year and same country rows into temp (zambia 2018 and Guinea 2018)
temp = df_internet.groupby(['country', 'survey_start']).count().sort_values('survey_type',ascending=False).reset_index()
for country,year in temp.loc[temp['survey_type']>1,['country','survey_start']].values:

    survey_year = temp.loc[temp['country']==country,'survey_start'].max()
    temp_check = df_internet.loc[(df_internet['country']==country) & (df_internet['survey_start']==survey_year),]
    avg = temp_check.mean(numeric_only=True)
    df_internet.drop(df_internet.loc[(df_internet['country']==country) & (df_internet['survey_start']==survey_year)].index, inplace=True)
    df_internet = df_internet.reset_index(drop=True)
    df_internet.loc[len(df_internet)] = [country, temp_check['iso3'].values[0], avg[0],'+'.join(temp_check['survey_type']) , avg[1], avg[2], avg[3]]


# no data before cut_off_year
df_internet = df_internet.loc[df_internet['survey_start']>=cut_off_year,]  # 192 data points


# ---------------------------------------------------------------------------------------------------
# 1.2 mobile outcomes
# ---------------------------------------------------------------------------------------------------
# there are 2 types of mobile outcomes based on the age groups: 15-49 and all age groups
# 1. 15-49 includes results in DHS, MICS and GSMA where the age group in GSMA is harmonized
#     perc_owns_mobile_telephone_wght_age_15_to_49_men (no ITU data)
#     perc_owns_mobile_telephone_wght_age_15_to_49_wom (no ITU data)
#     n_owns_mobile_telephone_age_15_to_49_men (includes ITU data)
#     n_owns_mobile_telephone_age_15_to_49_wom (includes ITU data)
# 2. all includes DHS, MICS and ITU data, where the ITU data is non-adjustable to the age group
#     perc_owns_mobile_telephone_wght_fm_perc_ratio
#     perc_owns_mobile_telephone_wght_fm_count_ratio

# relationships
# perc_owns_mobile_telephone_wght_fm_perc_ratio = perc_owns_mobile_telephone_wght_age_15_to_49_wom/perc_owns_mobile_telephone_wght_age_15_to_49_men + ITU columns
# perc_owns_mobile_telephone_wght_fm_count_ratio = n_owns_mobile_telephone_age_15_to_49_wom/n_owns_mobile_telephone_age_15_to_49_men # the relationship is valid

# codes to exam
# temp = df[['country','iso3','survey_start','survey_duration','survey_type']+selected_mobile_cols]
# temp['constructed_count']=df['n_owns_mobile_telephone_age_15_to_49_wom']/df['n_owns_mobile_telephone_age_15_to_49_men']
# temp['constructed_perc']=df['perc_owns_mobile_telephone_wght_age_15_to_49_wom']/df['perc_owns_mobile_telephone_wght_age_15_to_49_men']
# ---------------------------------------------------------------------------------------------------

# @ 2 sep 2024, use the file organised by jiaxuan
df_mobile = df_outcome[[x for x in df_outcome.columns if 'int' not in x]]
columns_to_exclude = ['n_age_15_to_49_men_valid_mobile','n_age_15_to_49_wom_valid_mobile','survey_duration']

df_mobile.drop(columns=columns_to_exclude, inplace=True)
df_mobile.dropna(inplace=True)


# 1.2.1 delete GSMA data and nans in the outcome vars
df_mobile.drop(df_mobile.loc[df_mobile['survey_type'].str.contains('gsma')].index, inplace=True)

# 1.2.2 save same year and same country rows into temp (same, Zambia 2018 and Guinea 2018)
temp = df_mobile.groupby(['country', 'survey_start']).count().sort_values('survey_type',ascending=False).reset_index()
for country,year in temp.loc[temp['survey_type']>1,['country','survey_start']].values:

    survey_year = temp.loc[temp['country']==country,'survey_start'].max()
    temp_check = df_mobile.loc[(df_mobile['country']==country) & (df_mobile['survey_start']==survey_year),]
    avg = temp_check.mean(numeric_only=True)
    df_mobile.drop(df_mobile.loc[(df_mobile['country']==country) & (df_mobile['survey_start']==survey_year)].index, inplace=True)
    df_mobile = df_mobile.reset_index(drop=True)
    df_mobile.loc[len(df_mobile)] = [country, temp_check['iso3'].values[0], avg[0],'+'.join(temp_check['survey_type']) , avg[1], avg[2], avg[3]]

# no data before cut_off_year
df_mobile = df_mobile.loc[df_mobile['survey_start']>=cut_off_year,]  # 159 data points

temp = df_mobile.groupby(['iso3']).count().sort_values('survey_start',ascending=False).reset_index()
temp=temp.loc[temp['country']>1]
temp['country'].sum()


# ---------------------------------------------------------------------------------------------------
#  1.3 visualise the gender gap in a map
# ---------------------------------------------------------------------------------------------------
import geopandas as gpd
import matplotlib.pyplot as plt

for indicator in ['internet','mobile']:

    if indicator == 'internet':
        temp = df_internet.copy()
        outcome_var = 'used_internet_past12months_fm_perc_ratio'
    else:
        temp = df_mobile.copy()
        outcome_var = 'owns_mobile_phone_fm_perc_ratio'

    world = utils.load_world()
    merged = world.merge(temp, left_on='iso_a3', right_on='iso3')  # only 128 countries left

    fig, ax = plt.subplots(1, 1, figsize=(20, 12))

    world.plot(ax=ax, color='grey')  # Default color for countries not in df
    merged.plot(ax=ax, column=outcome_var, cmap='viridis', legend=True, legend_kwds={'shrink': 0.5})
    plt.title(f'{indicator} coverage map', fontsize=15)
    ax.axis('off')
    plt.show()


# ====================================================================================================
# 2. merge outcome variables
# ====================================================================================================

# merge two outcome_vars
df_outcome = pd.merge(left=df_mobile, right=df_internet, on=['iso3', 'country'], how='outer', suffixes=('_mobile', '_internet'))

df_outcome.rename(columns = {'used_internet_past12months_fm_perc_ratio':'internet_ggi','owns_mobile_phone_fm_perc_ratio':'mobile_ggi',
                             "owns_mobile_phone_men":"mobile_men","owns_mobile_phone_wom":"mobile_wom",
                             "internet_use_in_12_months_men":"internet_men","internet_use_in_12_months_wom":"internet_wom",
                             'survey_start_mobile':'mobile_year','survey_start_internet':'internet_year',
                             'survey_type_internet':'internet_survey_type','survey_type_mobile':'mobile_survey_type'},inplace=True)

df_outcome.to_csv(params.PROCESSED/f'outcome_vars{"_latest" if latest_contrl else "_multiple_years"}.csv',index=False)


# ====================================================================================================
# 3. check distributions of the outcome variables
# ====================================================================================================
df_outcome = pd.read_csv(params.PROCESSED/f'outcome_vars_multiple_years.csv')
df_diff = df_outcome.loc[df_outcome['mobile_year']==df_outcome['internet_year'],]  # 136
for type in ['ggi','men','wom']:
    df_diff[f'diff_{type}'] = df_diff[f'mobile_{type}'] - df_diff[f'internet_{type}']

# plot the distribution of the differences
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

years = sorted(df_diff['internet_year'].unique())
palette = sns.color_palette("mako", len(years))
year_color_map = dict(zip(years, palette))


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, type in enumerate(['ggi','men','wom']):

    scatter = ax[i].scatter(df_diff[f'mobile_{type}'],df_diff[f'internet_{type}'],c=df_diff['internet_year'].map(year_color_map),alpha=0.8)

    ax[i].set_xlabel(f'mobile {type}')
    ax[i].set_ylabel(f'internet {type}')
    x_min = -0.1
    y_max = 1.3
    ax[i].plot([x_min, 1.2], [x_min, 1.2], ls="--", c="grey")
    ax[i].set_xlim(x_min,y_max)
    ax[i].set_ylim(x_min,y_max)
    #ax[i].text(0.05, 1.2, f"mean diff: {df_diff[f'diff_{type}'].mean():.2f}", fontsize=12)
    ax[i].text(0.05,1.14,f'mobile-internet\npositive portion: {len(df_diff.loc[df_diff[f"diff_{type}"]>0])/len(df_diff):.2f}',fontsize=10)
    for spine in ['top', 'right']:
        ax[i].spines[spine].set_visible(False)

n_cols = (len(years))  # Calculate number of columns needed for 2 rows
legend_elements = [mpatches.Patch(color=year_color_map[year], label=str(round(year))) for year in years]
fig.legend(handles=legend_elements, loc='lower center', ncol=n_cols, title='Survey Year', bbox_to_anchor=(0.5, 0.01),frameon=False)
fig.tight_layout(rect=[0, 0.1, 1, 0.97])
#plt.show()

plt.savefig(params.GRAPHS / 'outcome_diff_distribution.pdf')


df_outcome[['internet_survey_type','internet_year']].value_counts().reset_index().sort_values(by=['internet_year','internet_survey_type'],ignore_index=True)
df_outcome[['mobile_survey_type','mobile_year']].value_counts().reset_index().sort_values(by=['mobile_year','mobile_survey_type'],ignore_index=True)


# total variance of the two outcomes
for indicator in ['internet','mobile']:
    for type in ['ggi','men','wom']:
        print(f'{indicator} {type} variance: {sum((df_diff[f"{indicator}_{type}"]-df_diff[f"{indicator}_{type}"].mean())**2)}')

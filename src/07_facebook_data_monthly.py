"""
# Created by valler at 17/10/2024
Feature:
"""

import utils
import params
import pandas as pd
import pycountry as pc
import warnings

warnings.filterwarnings("ignore")
# ====================================================================================================
# 1. facebook data organise
# ====================================================================================================
#  read the latest facebook data (by month)
age_cols_dict = params.age_cols_dict
pop_cols = params.pop_cols

df_fb_all = utils.read_fb_data(years=range(2019, 2025))
df_fb_all.sort_values(by=['iso3','Year','Month'], inplace=True)

df_fb_all=df_fb_all[['iso3','Year', 'Month','Audience_Type']+[x for x in df_fb_all.columns if x not in {'Country', 'Year', 'Month', 'Audience_Type', 'iso3'}]]


# ====================================================================================================
# 2. UN population data preprocess
# ====================================================================================================

df_pop_all= pd.read_csv(params.RAW / 'un_1950_2023_processed.csv')

# 2.3 merge back to the fb data
pop_cols = pop_cols+[x.replace('_r','_m') for x in pop_cols]+[x.replace('_r','_f') for x in pop_cols]
temp = df_pop_all[['iso3', 'Year']+pop_cols]
df = pd.merge(left=df_fb_all, right=df_pop_all[['iso3','Year']+pop_cols], left_on=['iso3','Year'], right_on=['iso3','Year'], how='left')

# missing rate
df.isnull().sum().sum()/df.shape[0]/df.shape[1]  # 0.073


# ====================================================================================================
# 3. standardise the fb data
# ====================================================================================================

# 3.0 define standardising columns
age_cols = [f'fb_{x}' for x in age_cols_dict.keys()]
df_fb_sd = pd.DataFrame()
# def standardise_df_pop(age_range):
df_fb_sd[['iso3', 'survey_year','month']] = df[['iso3','Year','Month']]

# 3.1 standardise by age groups
for age_group in age_cols_dict.keys():

    fb_col, pop_col = age_cols_dict[age_group][0], age_cols_dict[age_group][1]
    fb_male_col,fb_femal_col = fb_col.replace("_ratio",'_men'),fb_col.replace("_ratio",'_women')
    male_col,female_col = pop_col.replace("_r",'_m'),pop_col.replace("_r",'_f')

    df_fb_sd[f'fb_{age_group}_r'] = df[fb_col] / df[pop_col]
    df_fb_sd[f'fb_{age_group}_wom'] = df[fb_femal_col]/df[female_col]
    df_fb_sd[f'fb_{age_group}_men'] = df[fb_male_col] / df[male_col]

df_fb_sd.isnull().sum().sum()/df_fb_sd.shape[0]/df_fb_sd.shape[1]# 0.06
# relationships valid?

# 3.2 behaviour types
cols = ['FB_all', 'FB_android_device_users_ratio', 'FB_iOS_device_users_ratio', 'FB_mobile_device_users_ratio']
for col in cols:
    if col in df.columns.to_list():

        col_without_ratio = col.replace('_ratio','')
        df_fb_sd[f'{col_without_ratio}_r'] = df[col]/df['18_inf_r']  # standardise by 18+ ratio
        if f'{col_without_ratio}_women' in df.columns.to_list():
            df_fb_sd[f'{col_without_ratio}_wom'] = df[f'{col_without_ratio}_women']/df['18_inf_f']
        if f'{col_without_ratio}_men' in df.columns.to_list():
            df_fb_sd[f'{col_without_ratio}_men'] = df[f'{col_without_ratio}_men'] / df['18_inf_m']

df_fb_sd.isnull().sum().sum()/df_fb_sd.shape[0]/df_fb_sd.shape[1]  # 0.082
df_fb_sd.dropna(subset=[x for x in df_fb_sd.columns if 'fb' in x], how='all', inplace=True)
# relationships valid, check the missing rate
for col in df_fb_sd.columns:
    print(col,df_fb_sd[col].isnull().sum()/len(df_fb_sd))

# ====================================================================================================
# 4. rolling window imputation
# ====================================================================================================
rolling_contrl = True
rolling_window_size = 3

df_fb_sd_rolling = df_fb_sd[['iso3', 'survey_year', 'month']].copy()
df_fb_sd_rolling['timestamp'] = pd.to_datetime([x.replace('.0','') for x in df_fb_sd['survey_year'].astype(str)+ '-' + df_fb_sd['month'].astype(str)+ '-01'])

for col in [x for x in df_fb_sd.columns if 'fb' in x.lower() ]:
    df_fb_sd_rolling[col] = df_fb_sd.groupby('iso3')[col].apply(lambda group: group.rolling(window=rolling_window_size).mean())

# replace the 2019-01 and 2019-02 in df_fb_sd_rolling with the df_fb_sd values
df_fb_sd_rolling.drop(df_fb_sd_rolling.loc[(df_fb_sd_rolling['survey_year'] == 2019) & (df_fb_sd_rolling['month'].isin([1,2])),].index, inplace=True)
df_fb_sd_rolling= pd.concat([df_fb_sd_rolling,df_fb_sd.loc[(df_fb_sd['survey_year'] == 2019) & (df_fb_sd['month'].isin([1,2])),]], axis=0)

df_fb_sd_rolling['timestamp'] = pd.to_datetime([x.replace('.0','') for x in df_fb_sd_rolling['survey_year'].astype(str)+ '-' + df_fb_sd_rolling['month'].astype(str)+ '-01'])
df_fb_sd_rolling.sort_values(by=['iso3', 'timestamp'], inplace=True)



for col in df_fb_sd_rolling.columns:
    print(col,df_fb_sd_rolling[col].isnull().sum()/len(df_fb_sd_rolling))


# ====================================================================================================
# 5. missing imputation
# the missing imputation of fb monthly data is seperated from the background data as the granularity is different
# ====================================================================================================

if rolling_contrl:
    df_fb_sd = df_fb_sd_rolling.copy()

df_fb_sd = df_fb_sd.sort_values(by=['iso3', 'survey_year', 'month'])

# make up year months if there is no data
for country in df_fb_sd['iso3'].unique():
    for year in range(2019, 2025):
        for month in range(1, 13):
            if year == 2024 and month > 6:
                continue
            else:
                if len(df_fb_sd.loc[(df_fb_sd['iso3'] == country) & (df_fb_sd['survey_year'] == year) & (df_fb_sd['month'] == month),]) == 0:
                    df_fb_sd = df_fb_sd.append({'iso3': country, 'survey_year': year, 'month': month}, ignore_index=True)

df_fb_sd.columns = [x.lower() for x in df_fb_sd.columns]
fb_vars = [x for x in df_fb_sd.columns if 'fb' in x.lower()]

def forward_fill_by_group(df, group_col, date_cols, value_cols):
    # Group by country
    df_filled = df.copy()
    df = df.sort_values(by=[group_col]+date_cols)
    for col in value_cols:
        df_filled[col] = df.groupby(group_col)[col].apply(lambda group: group.bfill())
    return df_filled


def interpolate_fill_by_group(df, group_col, date_cols, value_cols):
    # Group by country
    df_filled = df.copy()
    df_filled = df_filled.sort_values(by=[group_col] + date_cols)

    for col in value_cols:
        # Apply interpolation within each group (i.e., for each country)
        df_filled[col] = df_filled.groupby(group_col)[col].apply(lambda group: group.interpolate(method='linear'))

    return df_filled

# set the observation of 2023-08 and 2023-12 to be missing
# df_fb_sd.loc[(df_fb_sd['survey_year'] == 2023) & (df_fb_sd['month'].isin([7,8,10,11])), fb_vars] = None
# firstly filled by interpolation and then forward fill
#df_fb_sd_filled = interpolate_fill_by_group(df_fb_sd, 'iso3', ['survey_year', 'month'], fb_vars)
df_fb_sd_filled = forward_fill_by_group(df_fb_sd, 'iso3', ['survey_year', 'month'], fb_vars)


# drop RUS and fully null rows
df_fb_sd_filled = df_fb_sd_filled.loc[df_fb_sd_filled['iso3'] != 'RUS',]
# we've checked the null values in df_fb_sd_filled, and the missing rows have ALL missing in both year and month, across all fb_vars, delete them
df_fb_sd_filled = df_fb_sd_filled.dropna(subset=fb_vars, how='all')

df_fb_sd_filled['iso3'].nunique()  # 230

# ====================================================================================================
# 5. save the data
# ====================================================================================================

# convert the df_fb_sd to fat format by year
df_fb_sd_wide = df_fb_sd_filled.pivot(index='iso3', columns=['survey_year','month'],values=[x for x in df_fb_sd_filled.columns if x not in ['iso3','survey_year','month']])
df_fb_sd_wide = df_fb_sd_wide.T.bfill().T
df_fb_sd_wide.isnull().sum().sum()/df_fb_sd_wide.shape[0]/df_fb_sd_wide.shape[1]

# transform to wide format to fit with the most recent matching algorithms in the pipeline
df_fb_sd_wide.columns = [f'{col}_{round(year)}_{round(month)}' for col, year,month in df_fb_sd_wide.columns]
df_fb_sd_wide.reset_index(inplace=True)

# df_fb_sd_wide.to_csv(params.PROCESSED / f'fb_data/fb_all_sd_wide_monthly{"_rolling" if rolling_contrl else ""}.csv', index=False)
# df_fb_sd_filled.to_csv(params.PROCESSED / f'fb_data/fb_all_sd_monthly{"_rolling" if rolling_contrl else ""}.csv', index=False)


""" 
import numpy as np
# df_fb_sd_filled = pd.read_csv(params.PROCESSED / f'fb_data/fb_all_sd_monthly.csv')
df_plot = df_fb_sd_filled.copy()
income_year = 2022

df_plot['continent'] = [utils.get_continent_from_iso3(iso3) for iso3 in df_plot['iso3']]

# income type
df_income_level = pd.read_csv(params.RAW / 'income_classification.csv')
df_income_level.columns = df_income_level.loc[4,:].values

rows_to_remove = range(0,10)
df_income_level.drop(rows_to_remove, inplace=True)
df_income_level.replace('..', None, inplace=True)
df_income_level.rename(columns = {'Data for calendar year':'country',np.nan:'iso3' },inplace=True)
df_income_level.loc[df_income_level['iso3']=='VEN',f'{income_year}'] = 'UM'
df_plot = df_plot.merge(df_income_level[['iso3', f'{income_year}']],on='iso3',how='left')

# sub-region
df_sub_region = pd.read_csv(params.RAW / 'definition_of_regions.csv',delimiter=';')
df_plot = df_plot.merge(df_sub_region[['ISO-alpha3 Code','Sub-region Name']],left_on='iso3',right_on='ISO-alpha3 Code',how='left')
df_plot.rename(columns = {'Sub-region Name':'sub_region'},inplace=True)





import seaborn as sns
import matplotlib.pyplot as plt
def show_column(df,col):
    df['year_month'] = df['survey_year'].astype(str)+'-'+df['month'].astype(str).str.zfill(2)
    df.sort_values(by = 'year_month',inplace=True)
    sns.lineplot(data=df,x='year_month',y=col,hue='iso3',legend=False)
    plt.suptitle(col)
    plt.show()




# only keep
# 'LM','L'
df_plot[f'{income_year}'].unique()

temp = df_plot.loc[df_plot[f'{income_year}'].isin(['H','UM']),]
temp = df_plot.copy()
for col in params.fb_vars:

    show_column(temp,col)
    
    
    
# to generate pop_preprocessed.csv 

# 2.0 read df_pop (ratio and total count) @ 2024-10-25,deal with the latest UN population data
df_pop = pd.read_csv(params.RAW / 'un_1950_2023.csv')  # 236 countries
df_pop.dropna(subset=['ISO3_code'],inplace=True)
df_pop.drop(columns = ['SortOrder','LocID','Notes','SDMX_code','LocTypeID','LocTypeName','ParentID','VarID','AgeGrp','AgeGrpSpan'],inplace = True)

df_pop = df_pop.pivot(index=['ISO3_code','Time'],columns='AgeGrpStart',values=['PopTotal','PopFemale','PopMale'])
df_pop.reset_index(inplace=True)
gender_dict = {'PopTotal':'_t','PopFemale':'_f','PopMale':'_m'}
df_pop.columns = [f"{age}{gender_dict[gender] if gender in gender_dict.keys() else gender}" for gender,age in df_pop.columns]

age_cols_dict = params.age_cols_dict

# 2.1 calculate the total population by age group
for gender in ['_t','_f','_m']:
    temp = df_pop[['ISO3_code','Time']+[x for x in df_pop.columns if gender in x]]
    for age_group in age_cols_dict.keys():
        age_min, age_max = age_group.split('_')
        cols = [f'{age}{gender}' for age in range(int(age_min),int(age_max)+1 if age_max!='999' else 100+1)]
        df_pop[f'{age_group}{gender}'] = temp[cols].sum(axis=1)
df_pop_all = df_pop[['ISO3_code','Time']+[f'{age_group}{gender}' for age_group in age_cols_dict.keys() for gender in  ['_t','_f','_m']]].copy().rename(columns={'ISO3_code':'iso3','Time':'Year'})

# 2.2 calculate the ratio and multiply by 1000

for col in [x for x in df_pop_all.columns if '_' in x ]:
    df_pop_all[col] = df_pop_all[col]*1000
for age_group in age_cols_dict.keys():
    df_pop_all[f'{age_group}_r'] = df_pop_all[f'{age_group}_f']/df_pop_all[f'{age_group}_m']

df_pop_all.columns = [x.replace('999','inf') for x in df_pop_all.columns]

df_pop_all['Year'] = df_pop_all['Year'].astype(int)
temp = df_pop_all.loc[df_pop_all['Year'] == 2023,]
for year in [2024]:
    temp['Year'] = [year] * len(temp)
    df_pop_all = pd.concat([df_pop_all, temp], axis=0)
del temp

"""

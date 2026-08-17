"""
# Created by valler at 06/07/2024, modified in 2024-09-02
Feature: extract and get the facebook data prepared
standardise the fb ratios with the un pop ratios (the most recent year)
results are stored in the dropbox_path/data/fb_{date}_sd.csv
"""

import params
import pandas as pd
import pycountry as pc
import warnings

# from national.pipeline.src.ins.ins_download import age_groups
# from subnational.pipeline.src.params import gender_dict

warnings.filterwarnings("ignore")
# ====================================================================================================
# 1. facebook data organise
# ====================================================================================================
#  read the latest facebook data (by month)

# @ 2024.09.02
# to log the changes at the monthly level

age_cols_dict = params.age_cols_dict
pop_cols = params.pop_cols

years = range(2019, 2025)
months = range(1, 13)

# read data from 2019-01 to 2024-06 and concat them
df_fb_all = pd.DataFrame()
for year in years:
    for month in months:
        if year == 2024 and month > 11:
            continue
        date = f'{year}-{str(month).zfill(2)}'
        year_month = f'{year}{str(month).zfill(2)}'
        if year <=2021:
            if date == '2021-12':

                fb_data_path = params.FB_AVERAGED / f'mau_upper_{year_month}_averaged.csv'
            else:
                fb_data_path = params.FB_AVERAGED / f'mau_{year_month}_averaged.csv'
        else:
            fb_data_path = params.FB_AVERAGED / f'mau_upper_{year_month}_averaged.csv'

        temp = pd.read_csv(fb_data_path)
        df_fb_all = pd.concat([df_fb_all,temp],axis=0)

# mark iso3 to the data
df_fb_all['Country']=df_fb_all['Country'].str.upper()
df_fb_all['Country'].fillna('NA',inplace=True)

df_fb_all.dropna(thresh=5, axis=0, inplace=True)
df_fb_all['iso3'] = [None if pd.isnull(x) else pc.countries.get(alpha_2=x).alpha_3 if x not in ['AN', 'XK'] else {'AN': 'ANT', 'XK': 'XKX'}[x] for x in df_fb_all['Country']]

# average by iso3
# df_fb_all = df_fb_all.groupby(['iso3','Year','Month']).mean().reset_index()


# ====================================================================================================
# 2. backward calculation of features
# criteria: if the value is missing, then calculate it by linear regression on the country and the column
# only calculate the missing values in the year 2015-2018
# only models with performance > 0.9 are considered
# ====================================================================================================

import statsmodels.api as sm

threshold = 0
years_to_cal = range(2015, 2019)
cols = [x for x in df_fb_all.columns if 'FB' in x]
df_fb_all = df_fb_all[['Year','Month','iso3','Audience_Type']+cols]
df_fb_cal = pd.DataFrame(columns=['country']+cols)
non_calculation_countries = ['ERI','ESH','DJI']



for country in df_fb_all['iso3'].unique():
    if country in non_calculation_countries:
        continue

    df_fb_cal.loc[len(df_fb_cal),] = [country]+[None]*len(cols)
    for col in [x for x in cols if ('FB' in x) and ('ratio' not in x)]:

        df_sub = df_fb_all.loc[df_fb_all['iso3'] == country,['Year',col]].groupby('Year').mean().reset_index()
        # fit a linear model

        df_sub.dropna(inplace=True)
        if len(df_sub)==0:
            r2 = None
        else:
            X = df_sub[['Year']]
            X = sm.add_constant(X)
            y = df_sub[col].astype(float)
            # if y are all identical value: skip it
            if len(y.unique())==1:
                r2 = 1
                # record the value to the df_fb_all
                for cal_year in years_to_cal:
                    if len(df_fb_all.loc[(df_fb_all['iso3']==country)&(df_fb_all['Year']==cal_year)])==0:
                        df_fb_all.loc[len(df_fb_all)] = [cal_year,1,country, None]+[None]*len(cols)
                    df_fb_all.loc[(df_fb_all['iso3']==country)&(df_fb_all['Year']==cal_year),col] = y.unique()[0]
            else:
                model = sm.GLM(y, X, family=sm.families.Gaussian(sm.families.links.log()))
                result = model.fit()
                r2 = round(result.pseudo_rsquared(kind="cs"),3)
                if r2>=threshold:
                    for cal_year in years_to_cal:
                        if len(df_fb_all.loc[(df_fb_all['iso3'] == country) & (df_fb_all['Year'] == cal_year)]) == 0:
                            df_fb_all.loc[len(df_fb_all)] =  [cal_year,1,country, None] + [None] * len(cols)
                        df_fb_all.loc[(df_fb_all['iso3'] == country) & (df_fb_all['Year'] == cal_year), col] = result.predict([1, cal_year])[0]


        df_fb_cal.loc[df_fb_cal['country']==country,col] = r2



# check df_fb_cal for columns with 18+

print(threshold)
for col in ['FB_age_18_plus_men','FB_age_18_plus_women','FB_age_18_plus_ratio']:
    #print(df_fb_cal[col].astype(float).describe())
    print(col, df_fb_cal.loc[df_fb_cal[col]>=threshold].shape[0]/len(df_fb_cal))
    print('empty ratio',df_fb_all[col].isnull().sum()/len(df_fb_all))


# putting the missing values back to the df_fb_all (2015,2019)
for country in df_fb_all['iso3'].unique():
    country_data = df_fb_all.loc[df_fb_all['iso3']==country]
    for year in range(2015,2019):
        if year not in country_data['Year'].unique():
            df_fb_all.loc[len(df_fb_all)] = [year,1,country,None]+[None]*len(cols)

# now fill the missing values
# firstly interpolate fill the missing values
# then forward fill
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


df_fb_all = interpolate_fill_by_group(df_fb_all, 'iso3', ['Year','Month'], [x for x in df_fb_all.columns if ('FB' in x) and ('ratio' not in x)])
df_fb_all = forward_fill_by_group(df_fb_all, 'iso3', ['Year','Month'], [x for x in df_fb_all.columns if ('FB' in x) and ('ratio' not in x)])


# sort the data
df_fb_all.sort_values(['iso3','Year'],inplace=True)
df_fb_all.reset_index(drop=True,inplace=True)
for col in cols:
    df_fb_all[col] = df_fb_all[col].astype(float)


# putting the missing values back to the df_fb_all (2015,2019) and interpolate fill them
for country in df_fb_all['iso3'].unique():
    country_data = df_fb_all.loc[df_fb_all['iso3']==country]
    for year in range(2019,2025):
        if year != 2024:
            for month in range(1,13):
                if len(country_data.loc[(country_data['Year']==year)&(country_data['Month']==month)])==0:
                    df_fb_all.loc[len(df_fb_all)] = [year,month,country,None]+[None]*len(cols)
        else:
            for month in range(1,7):
                if len(country_data.loc[(country_data['Year']==year)&(country_data['Month']==month)])==0:
                    df_fb_all.loc[len(df_fb_all)] = [year,month,country,None]+[None]*len(cols)



df_fb_all = interpolate_fill_by_group(df_fb_all, 'iso3', ['Year','Month'], [x for x in df_fb_all.columns if ('FB' in x) and ('ratio' not in x)])
df_fb_all = forward_fill_by_group(df_fb_all, 'iso3', ['Year','Month'], [x for x in df_fb_all.columns if ('FB' in x) and ('ratio' not in x)])


print(threshold)
for col in ['FB_age_18_plus_men','FB_age_18_plus_women','FB_age_18_plus_ratio']:
    #print(df_fb_cal[col].astype(float).describe())
    print(col, df_fb_cal.loc[df_fb_cal[col]>=threshold].shape[0]/len(df_fb_cal))
    print('empty ratio',df_fb_all[col].isnull().sum()/len(df_fb_all))


# lastly calculate the ratio
for col in [x for x in cols if 'ratio' in x]:
    women_col = col.replace('_ratio','_women')
    men_col = col.replace('_ratio','_men')
    df_fb_all[col] = [x for x in df_fb_all[women_col]/df_fb_all[men_col]]

print(threshold)
print('after calculating the ratios')
for col in ['FB_age_18_plus_men','FB_age_18_plus_women','FB_age_18_plus_ratio']:
    #print(df_fb_cal[col].astype(float).describe())
    print(col, df_fb_cal.loc[df_fb_cal[col]>=threshold].shape[0]/len(df_fb_cal))
    print('empty ratio',df_fb_all[col].isnull().sum()/len(df_fb_all))

# delete RUS data
sanc_countries = ['RUS']
df_fb_all.drop(df_fb_all[df_fb_all['iso3'].isin(sanc_countries)].index, inplace=True)


import matplotlib.pyplot as plt


df = df_fb_all.copy()
for col in ['FB_age_18_plus_ratio', 'FB_age_18_plus_women', 'FB_age_18_plus_men']:
    # Combine Year and Month into a datetime column for the x-axis
    df = df.groupby(['iso3', 'Year']).mean().reset_index()
    #df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(Day=1))

    # Get unique countries
    countries = df['iso3'].unique()

    # Set up the figure
    num_countries = len(countries)
    plots_per_row = 15
    rows = (num_countries + plots_per_row - 1) // plots_per_row  # Round up for rows
    fig, axes = plt.subplots(rows, plots_per_row, figsize=(30, rows * 1.5))
    axes = axes.flatten()

    # Plot each country's data
    for i, country in enumerate(countries):
        country_data = df[df['iso3'] == country]
        ax = axes[i]
        ax.plot(country_data['Year'], country_data[col], marker='o', markersize=2, linestyle='-')
        ax.set_title(country, fontsize=8)
        ax.tick_params(axis='x', labelsize=6)
        ax.tick_params(axis='y', labelsize=6)
        ax.set_ylim(0, max(1.2*country_data[col].max(), 1))

    # Hide unused axes
    for i in range(num_countries, len(axes)):
        fig.delaxes(axes[i])

    # Adjust layout
    fig.suptitle(f'imputed {col}')
    #fig.tight_layout()
    fig.subplots_adjust(top=0.95, right=0.98,left=0.02,bottom=0.02, hspace=0.5, wspace=0.5)
    plt.show()


df_fb_all.to_csv(params.FB_PRE_PIPELINE / 'fb_all.csv',index=False)


# before calculation:
# df_fb_all shape: 1420,103
# after calculation: 2376,103

# ====================================================================================================
# 2. UN population data preprocess
# ====================================================================================================
# 2.0 read df_pop (ratio and total count)
df_pop = pd.read_csv(params.RAW / 'un_pop_2001_2021.csv')  # 236 countries
df_pop.rename(columns={'survey_start': 'Year'},inplace=True)  # 2022-2021

# 2.1 merge female and male data
df_pop_all = pd.read_csv(params.RAW / 'population_count.csv')
df_pop_all.rename(columns={'ISO3 Alpha-code':'iso3','survey_start':'Year'},inplace=True)  # 2022-2021
# female and male counts should time 1000 (counting unit)
for col in [x for x in df_pop_all.columns if '_' in x ]:
    df_pop_all[col] = df_pop_all[col]*1000

df_pop = df_pop_all.merge(df_pop, left_on=['iso3', 'Year'], right_on=['iso3','Year'], how='left')


# 2.2 copy the 2021 data and paste them to 2022-2024
df_pop['Year'] = df_pop['Year'].astype(int)
temp = df_pop.loc[df_pop['Year']==2021,]
for year in range(2022,2025):
    temp['Year'] = [year]*len(temp)
    df_pop=pd.concat([df_pop,temp],axis=0)

# 2.3 merge back to the fb data
pop_cols = pop_cols+[x.replace('_r','_m') for x in pop_cols]+[x.replace('_r','_f') for x in pop_cols]
temp = df_pop[['iso3', 'Year']+pop_cols]
df = pd.merge(left=df_fb_all, right=df_pop[['country','iso3','Year']+pop_cols], left_on=['iso3','Year'], right_on=['iso3','Year'], how='left')


# {'SJM', 'IOT', 'ATA', 'SGS', 'PCN', 'UMI', 'VAT', 'NFK', 'ANT', 'CXR'} are only available in FB
# {'PRK', 'RUS', 'CUB', 'SYR', 'NAM', 'IRN', 'SDN'} only available in df_pop (countries in sanction list)

# ====================================================================================================
# 3. standardise the fb data
# ====================================================================================================

# 3.0 define standardising columns
age_cols = [f'fb_{x}' for x in age_cols_dict.keys()]
df_fb_sd = pd.DataFrame()
# def standardise_df_pop(age_range):
df_fb_sd[['country', 'iso3', 'survey_year']] = df[['country','iso3','Year']]

# 3.1 standardise by age groups
for age_group in age_cols_dict.keys():

    fb_col, pop_col = age_cols_dict[age_group][0], age_cols_dict[age_group][1]
    fb_male_col,fb_femal_col = fb_col.replace("_ratio",'_men'),fb_col.replace("_ratio",'_women')
    male_col,female_col = pop_col.replace("_r",'_m'),pop_col.replace("_r",'_f')

    df_fb_sd[f'fb_{age_group}_r'] = df[fb_col] / df[pop_col]
    df_fb_sd[f'fb_{age_group}_wom'] = df[fb_femal_col]/df[female_col]
    df_fb_sd[f'fb_{age_group}_men'] = df[fb_male_col] / df[male_col]
# relationships valid

# 3.2 behaviour types
cols = ['FB_all','FB_android_device_users_ratio', 'FB_iOS_device_users_ratio', 'FB_mobile_device_users_ratio']
for col in cols:
    if col in df.columns.to_list():

        col_without_ratio = col.replace('_ratio','')
        df_fb_sd[f'{col_without_ratio}_r'] = df[col]/df['18_inf_r']  # standardise by 18+ ratio
        if f'{col_without_ratio}_women' in df.columns.to_list():
            df_fb_sd[f'{col_without_ratio}_wom'] = df[f'{col_without_ratio}_women']/df['18_inf_f']
        if f'{col_without_ratio}_men' in df.columns.to_list():
            df_fb_sd[f'{col_without_ratio}_men'] = df[f'{col_without_ratio}_men'] / df['18_inf_m']
# relationships valid

# 3.3 save the data
df_fb_sd.columns = df_fb_sd.columns.str.lower()
# convert the df_fb_sd to fat format by year
df_fb_sd_wide = df_fb_sd.pivot(index='iso3',columns='survey_year',values=[x for x in df_fb_sd.columns if x not in ['country','iso3','survey_year']])

# transform to wide format to fit with the most recent matching algorithms in the pipeline
df_fb_sd_wide.columns = [f'{col}_{round(year)}' for col, year in df_fb_sd_wide.columns]
df_fb_sd_wide.reset_index(inplace=True)
df_fb_sd_wide.to_csv(params.PROCESSED / f'fb_data/fb_all_sd_wide.csv', index=False)


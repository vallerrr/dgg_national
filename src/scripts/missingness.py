"""
# Created by valler at 18/09/2024
Feature: this script generate the list of countries that should not be included in the imputation, separately for internet and mobile

"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import params
import utils


# imputation check

indicator = 'internet'
data_path = params.PROCESSED / f"combined_data/updated_ground_truth_and_fb/{indicator}/combined_multiple_years_no_missing_keep_countries_fb_aligned_with_year.csv"
df = pd.read_csv(data_path)

# copy, and drop 'conti' only if present — params.bg_cols no longer carries it, and
# list.remove on the shared list mutated params for every later import
imputed_features = [x for x in params.bg_cols if x != 'conti']

# check the year missing rate
df_missing_recorder =pd.DataFrame(columns=['indicator','missing_type']+imputed_features)


# total missing count
for feature in imputed_features:
    if feature =='educ_hdi_r':
        df_feature_name = 'eys_r_year'
    else:
        df_feature_name = f'{feature}_year'

    if 'total' not in df_missing_recorder['missing_type'].to_list():
        df_missing_recorder.loc[len(df_missing_recorder),] = [indicator,'total']+[None]*len(imputed_features)

    df_missing_recorder.loc[df_missing_recorder['missing_type'] == 'total', feature] = df[df_feature_name].isna().sum()/len(df)

# by country

for country in df['iso3'].unique():
    for feature in imputed_features:
        if feature =='educ_hdi_r':
            df_feature_name = 'eys_r_year'
        else:
            df_feature_name = f'{feature}_year'

        if country not in df_missing_recorder['missing_type'].to_list():
            df_missing_recorder.loc[len(df_missing_recorder),] = [indicator,country]+[None]*len(imputed_features)

        df_missing_recorder.loc[df_missing_recorder['missing_type']==country,feature] = df.loc[df['iso3']==country,df_feature_name].isna().sum()

df_missing_recorder['total_missing_count_by_country'] = df_missing_recorder[imputed_features].sum(axis=1)

# with indicator/not
df_missing_recorder['indicator_missing'] = [0 if x in df.loc[df[f'{indicator}_ggi'].notnull(),'iso3'].to_list() else 1 for x in df_missing_recorder['missing_type']]

for col in ['indicator_missing','total_missing_count_by_country']:

    df_missing_recorder.loc[df_missing_recorder['missing_type']=='total',col] = None

df_threshold = pd.DataFrame(columns=['threshold','countries_excluded','indicators_excluded'])

for threshold in range(15,40):
    df_threshold.loc[len(df_threshold),] = [threshold,
                                            len(df_missing_recorder.loc[df_missing_recorder['total_missing_count_by_country']>threshold,'total_missing_count_by_country']),
                                            len(df_missing_recorder.loc[(df_missing_recorder['total_missing_count_by_country']>threshold) & (df_missing_recorder['indicator_missing']==0)])]


# df_missing_recorder.to_csv(params.RESULTS / f'logs/{indicator}_missing_rate.csv',index=False)


# map visualisation based on the iso3 code in column missing_type

'''
for threshold in [15,16,17,18,19,20,23,24,25,28,31]:
    world = utils.load_world()
    df_missing_by_country = df_missing_recorder.loc[df_missing_recorder['missing_type']!='total',['missing_type','total_missing_count_by_country','indicator_missing']]
    df_missing_by_country = df_missing_by_country.loc[df_missing_by_country['total_missing_count_by_country']>threshold,]
    world = world.merge(df_missing_by_country, left_on='iso_a3', right_on='missing_type', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    world.boundary.plot(ax=ax)
    world.plot(column='total_missing_count_by_country', ax=ax, legend=False)
    plt.title(f'{indicator} missing rate by country, threshold: {threshold}')
    ax.set_axis_off()
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    plt.show()
'''
threshold = {'mobile':19,'internet':20}

df_missing_by_country = df_missing_recorder.loc[df_missing_recorder['missing_type']!='total',['missing_type','total_missing_count_by_country','indicator_missing']]
countries_to_exclude = df_missing_by_country.loc[df_missing_by_country['total_missing_count_by_country']>threshold[indicator],'missing_type'].to_list()

# mobile : ['ABW', 'AND', 'ASM', 'ATG', 'BMU', 'CHI', 'CSK', 'CUW', 'CYM', 'DDR', 'DMA', 'FRO', 'GIB', 'GRL', 'GUM', 'HKG', 'IMN', 'KNA', 'MAC', 'MAF', 'MCO', 'MNP', 'MSR', 'MTQ', 'NCL', 'NIU', 'NRU', 'PRI', 'PRK', 'PSE', 'PYF', 'SAS', 'SCG', 'SHN', 'SUN', 'SXM', 'TCA', 'TWN', 'VDR', 'VGB', 'VIR', 'XKX', 'XTI', 'XXK', 'YMD', 'YUG']
# internet :['ABW', 'ASM', 'ATG', 'BMU', 'CHI', 'CSK', 'CUW', 'CYM', 'DDR', 'DMA', 'FRO', 'GIB', 'GRL', 'GUM', 'HKG', 'IMN', 'KNA', 'MAC', 'MAF', 'MCO', 'MNP', 'MSR', 'MTQ', 'NCL', 'NIU', 'PRI', 'PRK', 'PSE', 'PYF', 'SAS', 'SCG', 'SHN', 'SUN', 'SXM', 'TCA', 'TWN', 'VDR', 'VGB', 'VIR', 'XKX', 'XTI', 'XXK', 'YMD', 'YUG']

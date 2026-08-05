"""
# Created by valler at 13/08/2024
Feature:
"""

import pycountry
import pycountry_convert as pc
import pandas as pd
import statsmodels.api as sm
import params
import numpy as np
from statsmodels.iolib.summary2 import summary_col
import joblib
import os


latest_contrl = params.latest_contrl

def get_continent_from_iso3(iso3):
    """
     convert ISO3 country codes to continent names
    """
    iso3_dict = {'TLS': 'Asia', 'SXM': 'North America', 'XKX': 'Europe', 'ESH': 'Africa', 'VAT': 'Europe', 'PCN': 'Oceania', 'CHI': 'Asia', 'CSK': 'Europe', 'DDR': 'Europe', 'SAS': 'Asia', 'SCG': 'Europe', 'SUN': 'Europe and Asia', 'VDR': 'Asia', 'XTI': 'Asia', 'XXK': 'Europe', 'YMD': 'Asia', 'YUG': 'Europe'}

    if iso3 in iso3_dict.keys():
        return iso3_dict[iso3]
    try:
        country = pycountry.countries.get(alpha_3=iso3)
        if not country:
            return None
        iso2 = country.alpha_2
        continent_code = pc.country_alpha2_to_continent_code(iso2)
        continent_name = pc.convert_continent_code_to_continent_name(continent_code)
        return continent_name
    except KeyError:
            return None


def filling_missing_by_continent_mean(df_model, indep_vars, indicator,del_country=True):
    """
    fill missing values by the mean of the continent
    if del_country == True, it will exclude the countries in the params.countries_to_exclude_for_imputation, as those countries having more than 50% missing values
    """
    data_impute = df_model[indep_vars + ['conti', f'{indicator}_ggi','iso3',f'{indicator}_year',f'{indicator}_wom',f'{indicator}_men']].copy()
    if del_country:
        excluding_countries = params.countries_to_exclude_for_imputation[indicator]
        data_impute = data_impute[~data_impute['iso3'].isin(excluding_countries)]

    ind = data_impute[f'{indicator}_ggi'].copy()
    ind_wom = data_impute[f'{indicator}_wom'].copy()
    ind_men = data_impute[f'{indicator}_men'].copy()
    year = data_impute[f'{indicator}_year'].copy()

    means = data_impute.groupby('conti', as_index=False).mean(numeric_only=True) # using mean

    for col in indep_vars:
        data_impute[col] = [x if pd.notnull(x) else means.loc[means['conti'] == y, col].values[0] for x, y in zip(data_impute[col], data_impute['conti'])]
    data_impute[f'{indicator}_ggi']=ind.values
    data_impute[f'{indicator}_year']=year.values
    data_impute[f'{indicator}_wom']=ind_wom.values
    data_impute[f'{indicator}_men']=ind_men.values
    return data_impute

def fit_ols(data, indep_vars,indicator):

    X = data[indep_vars]
    y = data[f'{indicator}_ggi']
    X = sm.add_constant(X)
    reg = sm.OLS(y, X).fit()
    return reg




def fit_ols_loco(data,indep_vars,indicator,result='r2'):
    """
    leave one country out r2
    if result!='r2', it returns the dataframe with the true and predicted values
    """
    def compute_r_squared(y, pred_y):

        y = np.array(y)
        pred_y = np.array(pred_y)
        y_mean = np.mean(y)

        tss = np.sum((y - y_mean) * (y - y_mean))
        rss = np.sum((y - pred_y) * (y - pred_y))

        r_squared = 1 - (rss / tss)
        return r_squared

    countries = data.country.unique()
    indep_vars_ = indep_vars+['const']
    if latest_contrl: # only use the latest dataset for the control
        df_r2 = pd.DataFrame({'country':countries, 'true':data[f'{indicator}_ggi']})
    else:
        df_r2 = pd.DataFrame({'country':data['country'],'true':data[f'{indicator}_ggi']})

    for country in countries:
        c1 = data['country'] == country
        data['const'] = [1]*data.shape[0]

        temp = data.drop(data[c1].index)
        X = temp[indep_vars_]
        y = temp[f'{indicator}_ggi']

        reg = sm.OLS(y, X).fit()
        r2 = reg.rsquared
        r2_adj = reg.rsquared_adj
        pred_val = reg.predict(data.loc[c1, indep_vars_])

        c2 = df_r2['country'] == country
        df_r2.loc[c1, 'pred_val'] = pred_val
        df_r2.loc[c2,'fitted_r2']=r2
        df_r2.loc[c1, 'fitted_r2_adj'] = r2_adj
    if result == 'r2':
        return compute_r_squared(df_r2['true'], df_r2['pred_val'])
    else:
        return df_r2



def print_models_with_loco(models,model_names,loco_r2_dict=None):


    if loco_r2_dict:
        def custom_r2(model, model_name):
            return "{:.3f}".format(loco_r2_dict[model_name])

        # Create the info_dict
        info_dict = {
            'loco R2': lambda x: custom_r2(x, model_names[models.index(x)]),
             'N': lambda x: "{0:d}".format(int(x.nobs))
        }

        print(summary_col(models,
                          stars=True, float_format='%0.3f',
                          model_names=model_names,
                          info_dict=info_dict,
                          regressor_order=params.fb_cols+params.bg_cols))
    else:
        print(summary_col(models,
                          stars=True, float_format='%0.3f',
                          model_names=model_names,
                          info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))},
                          regressor_order=params.fb_cols+params.bg_cols))



def check_critical_info(df_model,indicator):

    coverage = {}
    coverage["online fit coverage"] =len(df_model[df_model[params.fb_vars_18_plus + [f'{indicator}_ggi']].notnull()][params.fb_vars_18_plus + [f'{indicator}_ggi']].dropna())
    coverage["online pred coverage"] = len(df_model[df_model[params.fb_vars_18_plus].notnull()][params.fb_vars_18_plus].dropna())

    coverage["offline fit coverage fit"] = len(df_model[df_model[params.bg_cols + [f'{indicator}_ggi']].notnull()][params.bg_cols + [f'{indicator}_ggi']].dropna())
    coverage["offline pred coverage"] = len(df_model[df_model[params.bg_cols].notnull()][params.bg_cols].dropna())

    coverage["combined fit coverage"] = len(df_model[df_model[params.bg_cols + params.fb_vars_18_plus + [f'{indicator}_ggi']].notnull()][params.bg_cols + params.fb_vars_18_plus + [f'{indicator}_ggi']].dropna())
    coverage["combined pred coverage"] = len(df_model[df_model[params.bg_cols+params.fb_vars_18_plus].notnull()][params.bg_cols+params.fb_vars_18_plus].dropna())



    coverage["online fit unique country coverage"] = len(df_model[df_model[params.fb_vars_18_plus + [f'{indicator}_ggi', 'iso3']].notnull()][params.fb_vars_18_plus + [f'{indicator}_ggi', 'iso3']].dropna()['iso3'].unique())
    coverage["online pred coverage unique country coverage"] = len(df_model[df_model[params.fb_vars_18_plus+['iso3']].notnull()][params.fb_vars_18_plus+['iso3']].dropna()['iso3'].unique())

    coverage["offline fit coverage fit unique country coverage"] = len(df_model[df_model[params.bg_cols + [f'{indicator}_ggi', 'iso3']].notnull()][params.bg_cols + [f'{indicator}_ggi', 'iso3']].dropna()['iso3'].unique())
    coverage["offline pred coverage unique country coverage"] = len(df_model[df_model[params.bg_cols+['iso3']].notnull()][params.bg_cols+['iso3']].dropna()['iso3'].unique())

    coverage["combined fit coverage unique country coverage"] = len(df_model[df_model[params.bg_cols + params.fb_vars_18_plus + [f'{indicator}_ggi', 'iso3']].notnull()][params.bg_cols + params.fb_vars_18_plus + [f'{indicator}_ggi', 'iso3']].dropna()['iso3'].unique())
    coverage["combined pred coverage unique country coverage"] = len(df_model[df_model[params.bg_cols + params.fb_vars_18_plus+['iso3']].notnull()][params.bg_cols + params.fb_vars_18_plus+['iso3']].dropna()['iso3'].unique())


    for key, value in coverage.items():
        print(key, value)

    return coverage

import pycountry as pcy
def read_fb_data(years):
    """
    read the facebook data by the year range sent in
    """
    months = range(1, 13)
    # read data from 2019-01 to 2024-06 and concat them
    df_fb_all = pd.DataFrame()
    for year in years:
        for month in months:
            if year == 2024 and month > 6:
                continue
            date = f'{year}-{str(month).zfill(2)}'

            if year <= 2021:
                if date == '2021-12':
                    fb_data_path = params.FB_COUNTS / f'mau_upper_counts_{date}.csv'
                else:
                    fb_data_path = params.FB_COUNTS / f'mau_counts_{date}.csv'
            else:
                fb_data_path = params.FB_COUNTS / f'mau_upper_counts_{date}.csv'

            temp = pd.read_csv(fb_data_path)
            df_fb_all = pd.concat([df_fb_all, temp], axis=0)

    # mark iso3 to the data
    df_fb_all.dropna(thresh=5, axis=0, inplace=True)
    df_fb_all['iso3'] = [None if pd.isnull(x) else pcy.countries.get(alpha_2=x).alpha_3 if x not in ['AN', 'XK'] else {'AN': 'ANT', 'XK': 'XKX'}[x] for x in df_fb_all['Country']]
    return df_fb_all


def load_model(file_path):
    return joblib.load(file_path) if os.path.exists(file_path) else None


def to_markdown_table(df, float_fmt='{:.4f}'):
    """
    render a DataFrame as a GitHub-flavoured markdown table

    Hand-rolled rather than pandas' `.to_markdown()`, which needs `tabulate` — not worth adding a
    dependency to a shared conda environment for a few summary tables.
    """
    def cell(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ''
        return float_fmt.format(v) if isinstance(v, (float, np.floating)) else str(v)

    header = '| ' + ' | '.join(str(c) for c in df.columns) + ' |'
    rule = '| ' + ' | '.join('---' for _ in df.columns) + ' |'
    rows = ['| ' + ' | '.join(cell(v) for v in row) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join([header, rule, *rows])


def mark_regions(df, iso3_col='iso3'):
    """
    add `conti` (continent) and `subregion` (UN M49) columns from an iso3 column

    Ported from dgg_research's `origin/pipeline` branch, where the convergence and adolescent
    notebooks group by these. Reads the UN M49 definitions from data/raw rather than an absolute
    Dropbox path; TWN and XKX are absent from that file and are set by hand, as upstream.
    """
    # UN M49 subregion
    m49 = pd.read_csv(params.RAW / 'definition_of_regions.csv', delimiter=';')
    df['conti'] = df[iso3_col].apply(get_continent_from_iso3)
    df['subregion'] = df[iso3_col].map(dict(zip(m49['ISO-alpha3 Code'], m49['Region Name'])))
    df.loc[df[iso3_col] == 'TWN', 'subregion'] = 'Eastern Asia'
    df.loc[df[iso3_col] == 'XKX', 'subregion'] = 'Southern Europe'

    # World Bank region — the scheme the trend and convergence figures group by
    wb = pd.read_csv(params.COHERENT_GGI['regions'])
    df['region'] = df[iso3_col].map(dict(zip(wb['iso3'], wb['region'])))
    df.loc[df[iso3_col] == 'SHN', 'region'] = 'Sub-Saharan Africa'   # St Helena
    df.loc[df[iso3_col] == 'XKX', 'region'] = 'Europe & Central Asia'  # Kosovo
    return df


def load_regions():
    """
    World Bank region per iso3 ('East Asia & Pacific', 'Europe & Central Asia', …)

    The scheme the trend figures already use. Covers all 214 predicted countries.
    """
    return pd.read_csv(params.COHERENT_GGI['regions'])


def get_country_name_from_iso3(iso3):
    """
    readable country name, mirroring get_continent_from_iso3's fallback style

    pycountry returns full official names ('Congo, The Democratic Republic of the'), which are too
    long to direct-label a chart with. The overrides give the short conventional form.
    """
    iso3_dict = {
        'XKX': 'Kosovo', 'CHI': 'Channel Islands', 'ANT': 'Netherlands Antilles',
        'COD': 'DR Congo', 'COG': 'Congo', 'TZA': 'Tanzania', 'VEN': 'Venezuela',
        'BOL': 'Bolivia', 'IRN': 'Iran', 'KOR': 'South Korea', 'PRK': 'North Korea',
        'LAO': 'Laos', 'SYR': 'Syria', 'MDA': 'Moldova', 'RUS': 'Russia', 'VNM': 'Vietnam',
        'TWN': 'Taiwan', 'CIV': "Côte d'Ivoire", 'FSM': 'Micronesia', 'GBR': 'United Kingdom',
        'USA': 'United States', 'ARE': 'UAE', 'CAF': 'Central African Rep.', 'BRN': 'Brunei',
        'PSE': 'Palestine', 'MKD': 'North Macedonia', 'SSD': 'South Sudan', 'CPV': 'Cabo Verde',
        'VCT': 'St Vincent & Grenadines', 'KNA': 'St Kitts & Nevis', 'LCA': 'St Lucia',
        'TTO': 'Trinidad & Tobago', 'ATG': 'Antigua & Barbuda', 'BIH': 'Bosnia & Herzegovina',
        'STP': 'São Tomé & Príncipe', 'GNB': 'Guinea-Bissau', 'GNQ': 'Equatorial Guinea',
    }
    if iso3 in iso3_dict:
        return iso3_dict[iso3]
    country = pycountry.countries.get(alpha_3=iso3)
    if not country:
        return None
    return getattr(country, 'common_name', country.name)


def load_world():
    """
    world country polygons for the coverage maps

    Drop-in replacement for `gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))`,
    which GeoPandas 1.0 removed. Reads the same Natural Earth 110m admin-0 layer from
    params.WORLD_SHP and exposes the `iso_a3` column the old dataset provided, falling
    back to ADM0_A3 for the handful of territories Natural Earth codes as '-99'.
    """
    import geopandas as gpd

    world = gpd.read_file(params.WORLD_SHP)
    world['iso_a3'] = [adm if iso == '-99' else iso for iso, adm in zip(world['ISO_A3'], world['ADM0_A3'])]
    return world

"""
# Created by valler at 29/07/2024
Feature: 

"""
import matplotlib.pyplot as plt

import params
import utils
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.iolib.summary2 import summary_col


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
                          regressor_order=fb_cols+lst_last_model_ind))
    else:
        print(summary_col(models,
                          stars=True, float_format='%0.3f',
                          model_names=model_names,
                          info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))},
                          regressor_order=fb_cols + lst_last_model_ind))
def predict_val(reg,df_model,indep_vars):
    """
    calculate the predicted value for sent regression
    """
    data = df_model[indep_vars + ['iso3',f'{indicator}_ggi']].dropna(subset=indep_vars).copy()
    data = sm.add_constant(data)
    data['pred'] = reg.predict(data[['const']+ indep_vars])

    return data[['iso3','pred']]


# ====================================================================================================
#  run both mobile and internet at once and present the regression table
# ====================================================================================================

latest_contrl = True
date= '2020-06'

loco_r2_dict = {}

for indicator in ['internet','mobile']:
    # ====================================================================================================
    # 1. read data
    # ====================================================================================================
    df_background = pd.read_csv(params.PROCESSED/f'{indicator}_aligned_final{"_latest" if latest_contrl else "_multiple_years"}.csv')  # contains outcome vars
    df_fb = pd.read_csv(params.PROCESSED/f'fb_data/fb_{date}_sd.csv')

    df_fb.dropna(thresh=28,axis=0,inplace=True)

    # ====================================================================================================
    # 2. var define
    # ====================================================================================================

    # 2.0 new vars
    df_background['gdp_pcap'] = np.log(df_background['gdp_pcap'])
    df_background['educ_hdi_r'] = np.average(df_background[['eys_r','mys_r']],axis=1)
    # continent

    # 2.1 concatenating data
    lst_last_model_ind = ['gggi_eas','gdp_pcap','le_r','wdi_litradr','wdi_gertr','gggi_ggi','wdi_unempr','educ_hdi_r','wdi_internet']
    fb_cols = ['fb_18_999', 'fb_25_29', 'fb_21_999','fb_18_23','fb_20_64', 'fb_android_device_users_ratio']

    df_model = df_background[['iso3', f'{indicator}_ggi']+lst_last_model_ind].merge(df_fb[['iso3']+fb_cols], left_on='iso3', right_on='iso3', how='outer')  # 127 countries
    df_model['conti']=[utils.get_continent_from_iso3(x) for x in df_model['iso3']]

    # deleting ATA
    df_model.dropna()
    # for col in lst_last_model_ind:
    #    print(col, df_model[col].isnull().sum())

    features = {}
    features['internet']={'online':['fb_18_999'], 'offline':['gggi_eas','gdp_pcap','le_r','educ_hdi_r','wdi_internet'],'combined':['wdi_gertr','gdp_pcap','fb_18_999','fb_18_23','fb_20_64']}
    features['mobile']={'online':['fb_18_999'], 'offline':['wdi_litradr','gdp_pcap','le_r','gggi_eas','wdi_unempr'], 'combined':['gggi_eas','gggi_ggi','fb_21_999','fb_25_29','fb_android_device_users_ratio']}


    # 2.2 alignment check
    # internet indicator
    # align regardless of the outcome availability
    print(f'all indicators {len(df_model[lst_last_model_ind].dropna())}')  # 103 countries when deleting na in all indicators
    for model in features[indicator].keys():
        temp = df_model[features[indicator][model]].dropna()
        print(indicator, model, f"available countries {len(temp)}")

    df_results = df_model[['iso3', 'conti', f'{indicator}_ggi']]
    # ====================================================================================================
    # 3. run simple regression model on the indicators aligned with the last version
    # ====================================================================================================



    if indicator=="internet":
        # 1. only fb
        indep_vars = features[indicator]['online']
        data = df_model[indep_vars + ['iso3',f'{indicator}_ggi']].dropna()
        reg_fb = utils.fit_ols(data, indep_vars, indicator)
        loco_r2_dict['fb'] = utils.fit_ols_loco(data,indep_vars,indicator)
        # df_results = df_results.merge(predict_val(reg=reg_fb,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'fb_pred'}),left_on='iso3',right_on='iso3',how='left')

        data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
        reg_fb_impute = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
        loco_r2_dict['fb_impute'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)
        # df_results = df_results.merge(predict_val(reg=reg_fb_impute,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'fb_impute_pred'}),left_on='iso3',right_on='iso3',how='left')
        # df_model=df_model or df_model = data_impute?

        # 2. model_only_bg
        indep_vars = features[indicator]['offline']
        data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
        reg_bg = utils.fit_ols(data, indep_vars, indicator)
        loco_r2_dict['bg'] = utils.fit_ols_loco(data, indep_vars,indicator)
        df_results = df_results.merge(predict_val(reg=reg_bg,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'bg_pred'}),left_on='iso3',right_on='iso3',how='left')

        data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
        reg_bg_impute = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
        loco_r2_dict['bg_impute'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)
        # df_results = df_results.merge(predict_val(reg=reg_bg_impute,df_model=data_impute,indep_vars=indep_vars).rename(columns={'pred':'bg_impute_pred'}),left_on='iso3',right_on='iso3',how='left')

        # 3. comb
        indep_vars = features[indicator]['combined']
        data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
        reg_comb = utils.fit_ols(data, indep_vars, indicator)
        loco_r2_dict['comb'] = utils.fit_ols_loco(data,indep_vars,indicator)
        temp = utils.fit_ols_loco(data,indep_vars,indicator,result='f')
        df_results = df_results.merge(predict_val(reg=reg_comb,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'comb_pred'}),left_on='iso3',right_on='iso3',how='left')

        # missing imputation
        data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
        reg_comb_impute = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
        loco_r2_dict['comb_impute'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)
        # df_results = df_results.merge(predict_val(reg=reg_comb_impute,df_model=data_impute,indep_vars=indep_vars).rename(columns={'pred':'comb_impute_pred'}),left_on='iso3',right_on='iso3',how='left')

        # print out only internet regressions
        #  models = [reg_fb, reg_fb_impute, reg_bg, reg_bg_impute, reg_comb, reg_comb_impute]
        #  model_names = ['fb', 'fb_impute', 'bg', 'bg_impute', 'comb', 'comb_impute']
        #  print_models_with_loco(models,model_names,loco_r2_dict)



    # those codes are used to put the regressions of both indicator together
    if indicator=="mobile":

        indep_vars = features[indicator]['online']
        data = df_model[indep_vars + ['iso3',f'{indicator}_ggi']].dropna()
        reg_fb_mob = utils.fit_ols(data, indep_vars, indicator)
        loco_r2_dict['fb_mob'] = utils.fit_ols_loco(data,indep_vars,indicator)

        data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
        reg_fb_impute_mob = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
        loco_r2_dict['fb_impute_mob'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)


        # 2. model_only_bg
        indep_vars = features[indicator]['offline']
        data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
        reg_bg_mob = utils.fit_ols(data, indep_vars, indicator)
        loco_r2_dict['bg_mob']  = utils.fit_ols_loco(data,indep_vars,indicator)

        data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
        reg_bg_impute_mob = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
        loco_r2_dict['bg_impute_mob']  = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)

        # 3. comb
        indep_vars = features[indicator]['combined']
        data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
        reg_comb_mob = utils.fit_ols(data, indep_vars, indicator)
        loco_r2_dict['comb_mob'] = utils.fit_ols_loco(data,indep_vars,indicator)

        # missing imputation
        data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
        reg_comb_impute_mob = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
        loco_r2_dict['comb_impute_mob']  = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)

        models = [reg_fb_mob, reg_fb_impute_mob, reg_bg_mob, reg_bg_impute_mob, reg_comb_mob, reg_comb_impute_mob]
        model_names = ['fb_mob', 'fb_impute_mob', 'bg_mob', 'bg_impute_mob', 'comb_mob', 'comb_impute_mob']
        print_models_with_loco(models,model_names,loco_r2_dict)

        models = [reg_fb, reg_fb_impute, reg_bg, reg_bg_impute, reg_comb, reg_comb_impute, reg_fb_mob, reg_fb_impute_mob, reg_bg_mob, reg_bg_impute_mob, reg_comb_mob, reg_comb_impute_mob]
        model_names = ['fb', 'fb_impute', 'bg', 'bg_impute', 'comb', 'comb_impute', 'fb_mob', 'fb_impute_mob', 'bg_mob', 'bg_impute_mob', 'comb_mob', 'comb_impute_mob']
        print_models_with_loco(models,model_names,loco_r2_dict)




# ====================================================================================================
#  test on new variables
# ====================================================================================================

latest_contrl = False
date= '2022-06'

df_result = pd.DataFrame(columns=['age_group','indicator','r2'])
age_group = 'fb_20_64'


for age_group in ['fb_18_999', 'fb_14_15', 'fb_15_16',
       'fb_16_17', 'fb_17_18', 'fb_18_19', 'fb_13_14', 'fb_15_19', 'fb_20_24',
       'fb_25_29', 'fb_30_34', 'fb_35_39', 'fb_40_44', 'fb_45_49', 'fb_50_54',
       'fb_55_59', 'fb_60_64', 'fb_18_23', 'fb_20_999', 'fb_20_64',
       'fb_21_999', 'fb_25_999', 'fb_25_49', 'fb_25_64', 'fb_50_999',
       'fb_60_999', 'fb_65_999', 'fb_all']:
    features = {}
    features['internet'] = {'online': [age_group], 'offline': ['gggi_eas', 'gdp_pcap', 'le_r', 'educ_hdi_r', 'wdi_internet'], 'combined': ['wdi_gertr', 'gdp_pcap', 'fb_18_999', 'fb_18_23', 'fb_20_64']}
    features['mobile'] = {'online': [age_group], 'offline': ['wdi_litradr', 'gdp_pcap', 'le_r', 'gggi_eas', 'wdi_unempr'], 'combined': ['gggi_eas', 'gggi_ggi', 'fb_21_999', 'fb_25_29', 'fb_android_device_users_ratio']}

    loco_r2_dict = {}

    for indicator in ['internet', 'mobile']:
        # ====================================================================================================
        # 1. read data
        # ====================================================================================================
        df_background = pd.read_csv(params.PROCESSED/f'{indicator}_aligned_final{"_latest" if latest_contrl else "_multiple_years"}.csv')  # contains outcome vars
        df_fb = pd.read_csv(params.PROCESSED/f'fb_data/fb_{date}_sd.csv')

        df_fb.dropna(thresh=28,axis=0,inplace=True)

        # ====================================================================================================
        # 2. var define
        # ====================================================================================================

        # 2.0 new vars
        df_background['gdp_pcap'] = np.log(df_background['gdp_pcap'])
        df_background['educ_hdi_r'] = np.average(df_background[['eys_r','mys_r']],axis=1)
        # continent

        # 2.1 concatenating data
        lst_last_model_ind = ['gggi_eas','gdp_pcap','le_r','wdi_litradr','wdi_gertr','gggi_ggi','wdi_unempr','educ_hdi_r','wdi_internet']
        fb_cols = list(dict.fromkeys(['fb_18_999', 'fb_25_29', 'fb_21_999','fb_18_23','fb_20_64', 'fb_android_device_users_ratio']+[x for x in df_fb.columns if "fb_" in x]))

        df_model = df_background[['iso3', f'{indicator}_ggi']+lst_last_model_ind].merge(df_fb[['iso3']+fb_cols], left_on='iso3', right_on='iso3', how='outer')  # 127 countries
        df_model['conti']=[utils.get_continent_from_iso3(x) for x in df_model['iso3']]

        # deleting ATA
        df_model.dropna()
        # for col in lst_last_model_ind:
        #    print(col, df_model[col].isnull().sum())



        # 2.2 alignment check
        # internet indicator
        # align regardless of the outcome availability
        print(f'all indicators {len(df_model[lst_last_model_ind].dropna())}')  # 103 countries when deleting na in all indicators
        for model in features[indicator].keys():
            temp = df_model[features[indicator][model]].dropna()
            print(indicator, model, f"available countries {len(temp)}")

        df_results = df_model[['iso3', 'conti', f'{indicator}_ggi']]
        # ====================================================================================================
        # 3. run simple regression model on the indicators aligned with the last version
        # ====================================================================================================



        if indicator=="internet":
            # 1. only fb
            indep_vars = features[indicator]['online']
            data = df_model[indep_vars + ['iso3',f'{indicator}_ggi']].dropna()
            reg_fb = utils.fit_ols(data, indep_vars, indicator)
            loco_r2_dict['fb'] = utils.fit_ols_loco(data,indep_vars,indicator)
            # df_results = df_results.merge(predict_val(reg=reg_fb,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'fb_pred'}),left_on='iso3',right_on='iso3',how='left')

            data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
            reg_fb_impute = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
            loco_r2_dict['fb_impute'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)
            # df_results = df_results.merge(predict_val(reg=reg_fb_impute,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'fb_impute_pred'}),left_on='iso3',right_on='iso3',how='left')
            # df_model=df_model or df_model = data_impute?

            # 2. model_only_bg
            indep_vars = features[indicator]['offline']
            data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
            reg_bg = utils.fit_ols(data, indep_vars, indicator)
            loco_r2_dict['bg'] = utils.fit_ols_loco(data, indep_vars,indicator)
            df_results = df_results.merge(predict_val(reg=reg_bg,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'bg_pred'}),left_on='iso3',right_on='iso3',how='left')

            data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
            reg_bg_impute = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
            loco_r2_dict['bg_impute'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)
            # df_results = df_results.merge(predict_val(reg=reg_bg_impute,df_model=data_impute,indep_vars=indep_vars).rename(columns={'pred':'bg_impute_pred'}),left_on='iso3',right_on='iso3',how='left')

            # 3. comb
            indep_vars = features[indicator]['combined']
            data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
            reg_comb = utils.fit_ols(data, indep_vars, indicator)
            loco_r2_dict['comb'] = utils.fit_ols_loco(data,indep_vars,indicator)
            temp = utils.fit_ols_loco(data,indep_vars,indicator,result='f')
            df_results = df_results.merge(predict_val(reg=reg_comb,df_model=df_model,indep_vars=indep_vars).rename(columns={'pred':'comb_pred'}),left_on='iso3',right_on='iso3',how='left')

            # missing imputation
            data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
            reg_comb_impute = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
            loco_r2_dict['comb_impute'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)
            # df_results = df_results.merge(predict_val(reg=reg_comb_impute,df_model=data_impute,indep_vars=indep_vars).rename(columns={'pred':'comb_impute_pred'}),left_on='iso3',right_on='iso3',how='left')

            # print out only internet regressions
            #  models = [reg_fb, reg_fb_impute, reg_bg, reg_bg_impute, reg_comb, reg_comb_impute]
            #  model_names = ['fb', 'fb_impute', 'bg', 'bg_impute', 'comb', 'comb_impute']
            #  print_models_with_loco(models,model_names,loco_r2_dict)


        # those codes are used to put the regressions of both indicator together
        if indicator=="mobile":

            indep_vars = features[indicator]['online']
            data = df_model[indep_vars + ['iso3',f'{indicator}_ggi']].dropna()
            reg_fb_mob = utils.fit_ols(data, indep_vars, indicator)
            loco_r2_dict['fb_mob'] = utils.fit_ols_loco(data,indep_vars,indicator)

            data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
            reg_fb_impute_mob = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
            loco_r2_dict['fb_impute_mob'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)


            # 2. model_only_bg
            indep_vars = features[indicator]['offline']
            data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
            reg_bg_mob = utils.fit_ols(data, indep_vars, indicator)
            loco_r2_dict['bg_mob']  = utils.fit_ols_loco(data,indep_vars,indicator)

            data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
            reg_bg_impute_mob = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
            loco_r2_dict['bg_impute_mob']  = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)

            # 3. comb
            indep_vars = features[indicator]['combined']
            data = df_model[indep_vars+[f'{indicator}_ggi','iso3']].dropna()
            reg_comb_mob = utils.fit_ols(data, indep_vars, indicator)
            loco_r2_dict['comb_mob'] = utils.fit_ols_loco(data,indep_vars,indicator)

            # missing imputation
            data_impute = utils.filling_missing_by_continent_mean(df_model,indep_vars,indicator)
            reg_comb_impute_mob = utils.fit_ols(data_impute.dropna(), indep_vars, indicator)
            loco_r2_dict['comb_impute_mob'] = utils.fit_ols_loco(data_impute.dropna(),indep_vars,indicator)

            models = [reg_fb_mob, reg_fb_impute_mob, reg_bg_mob, reg_bg_impute_mob, reg_comb_mob, reg_comb_impute_mob]
            model_names = ['fb_mob', 'fb_impute_mob', 'bg_mob', 'bg_impute_mob', 'comb_mob', 'comb_impute_mob']
            print_models_with_loco(models,model_names,loco_r2_dict)

            models = [reg_fb, reg_fb_impute, reg_bg, reg_bg_impute, reg_comb, reg_comb_impute, reg_fb_mob, reg_fb_impute_mob, reg_bg_mob, reg_bg_impute_mob, reg_comb_mob, reg_comb_impute_mob]
            model_names = ['fb', 'fb_impute', 'bg', 'bg_impute', 'comb', 'comb_impute', 'fb_mob', 'fb_impute_mob', 'bg_mob', 'bg_impute_mob', 'comb_mob', 'comb_impute_mob']
            # print_models_with_loco(models,model_names,loco_r2_dict)


            df_result.loc[len(df_result),] = [age_group,'internet',reg_fb.rsquared]
            df_result.loc[len(df_result),] = [age_group, 'mobile', reg_fb_mob.rsquared]


# ====================================================================================================
# 4. run simple regression model on new indicators and only one model for each outcome variable
# ====================================================================================================
'''
# 4.1 missing check
lst_last_model_ind = ['gggi_eas','gdp_pcap','le_r','hdi_r','ihdi','wdi_litradr','wdi_gertr','gggi_ggi','wdi_unempr','educ_hdi_r']
fb_cols = ['fb_18_999', 'fb_25_29', 'fb_21_999','fb_18_23','fb_20_64', 'fb_android_device_users_ratio']

temp = df_model[['iso3']+fb_cols].dropna()
print(indicator, 'fb', f"available countries {len(temp)}")

temp = df_model[['iso3']+lst_last_model_ind].dropna()
print(indicator, 'bg', f"available countries {len(temp)}")
countries_bg = temp.iso3.unique()


temp = df_model[['iso3']+fb_cols+lst_last_model_ind].dropna()
print(indicator, 'comb', f"available countries {len(temp)}")
countries_comb = temp.iso3.unique()


df_model = df_background[['iso3',f'{indicator}_ggi']+lst_last_model_ind].merge(df_fb[['iso3']+fb_cols], left_on='iso3', right_on='iso3', how='inner')
for col in lst_last_model_ind:
   print(col, df_model[col].isnull().sum())


# 4.3 fitting models


data = df_model[[f'{indicator}_ggi']+fb_cols].dropna()
X = data[fb_cols]
X = sm.add_constant(X)
y = data[f'{indicator}_ggi']

reg_only_fb = sm.OLS(y, X).fit()

# model_only_bg

data = df_model[lst_last_model_ind+[f'{indicator}_ggi']].dropna()
X = data[lst_last_model_ind]
X = sm.add_constant(X)
y = data[f'{indicator}_ggi']

reg_only_bg = sm.OLS(y, X).fit()


data = df_model[fb_cols+lst_last_model_ind+[f'{indicator}_ggi']].dropna()
X = data[fb_cols+lst_last_model_ind]
X = sm.add_constant(X)
y = data[f'{indicator}_ggi']

reg_combined = sm.OLS(y, X).fit()

print(summary_col([reg_only_fb,reg_only_bg,reg_combined],stars=True,float_format='%0.2f',model_names=['fb','bg','combined'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))},
                  regressor_order=fb_cols+lst_last_model_ind))

'''
'''


data = df_model[fb_cols+[f'{indicator}_ggi']].dropna()
X = data[fb_cols]
X = sm.add_constant(X)
y = data[f'{indicator}_ggi']

reg_only_fb_mob = sm.OLS(y, X).fit()

# model_only_bg

data = df_model[lst_last_model_ind+[f'{indicator}_ggi']].dropna()
X = data[lst_last_model_ind]
X = sm.add_constant(X)
y = data[f'{indicator}_ggi']

reg_only_bg_mob = sm.OLS(y, X).fit()



data = df_model[fb_cols+lst_last_model_ind+[f'{indicator}_ggi']].dropna()
X = data[fb_cols+lst_last_model_ind]
X = sm.add_constant(X)
y = data[f'{indicator}_ggi']

reg_combined_mob = sm.OLS(y, X).fit()


print(summary_col([reg_only_fb,reg_only_bg,reg_combined,reg_only_fb_mob,reg_only_bg_mob,reg_combined_mob],stars=True,float_format='%0.3f',model_names=['int_fb','int_bg','int_combined','mob_fb','mob_bg','mob_combined'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs))},
                  regressor_order= fb_cols+lst_last_model_ind))
'''


"""
# Created by valler at 10/06/2024
Feature: params of the project — all paths, constants, config, seeds and plot style.

Paths are resolved relative to the project root (CONVENTIONS.md §1), never hardcoded.
`data/` and `outputs/{results,models,graphs}` are gitignored symlinks into the shared
Dropbox tree; see data/README.md for what each one points at.
"""
from pathlib import Path

# ====================================================================================================
# paths (all relative to the project root)
# ====================================================================================================
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / 'data'
RAW = DATA / 'raw'              # -> Dropbox .../new_national_pipeline_files/files
PROCESSED = DATA / 'processed'  # -> Dropbox .../new_national_pipeline_files/data
EXTERNAL = DATA / 'external'    # -> Dropbox dgg_research (upstream fb pipeline)

# UN World Population Prospects. The single-age workbooks live under PROCESSED because they are
# acquired per refresh, not shipped with the pipeline; the panel the model stages read is the
# `un_1950_2023_processed.csv` in RAW. See doc/decisions.md D29.
UN_POP = PROCESSED / 'un_pop'
UN_POP_RAW = UN_POP / 'raw'

# Which WPP edition every stage weights with. This is the ONE place it is chosen — 13, 15 and 16
# all read `UN_POP_PROCESSED`, so switching edition is a one-line change and can never leave two
# stages on different vintages.
#
#   'shipped'  the file that has always been in RAW — WPP2022, 1950-2024 (its 2024 is a copy of
#              its 2023). Every published figure to date rests on this.
#   'wpp2024'  the rebuild from 01_01_un_population_wpp.py — WPP2024 estimates, 1950-2023.
#
# Switching re-bases every population-weighted regional figure (D16, D24, D27). Population enters
# only as a *weight*: country-level GGIs do not move at all. See D29 for the size of the revision
# and D35 for what it actually changed.
UN_POP_EDITION = 'shipped'

def _un_pop_panel(edition):
    if edition == 'shipped':
        return RAW / 'un_1950_2023_processed.csv'
    matches = sorted(UN_POP.glob(f'un_1950_2023_processed_{edition}_*.csv'))
    if not matches:
        raise FileNotFoundError(
            f'no un_1950_2023_processed_{edition}_<date>.csv in {UN_POP} — '
            'run src/01_01_un_population_wpp.py first')
    return matches[-1]

UN_POP_PROCESSED = _un_pop_panel(UN_POP_EDITION)

OUTPUTS = ROOT / 'outputs'
FIG = OUTPUTS / 'fig'
TABLES = OUTPUTS / 'tables'

# ----------------------------------------------------------------------------------------------------
# pipeline stages
# ----------------------------------------------------------------------------------------------------
# The numbering is the run order, and it is the same in three places: the notebook folders under
# `src/notebooks/`, the table folders under `outputs/tables/`, and the figure folders under
# `outputs/fig/`. An artefact's path therefore names the stage that produced it.
#
# The analysis code lives in the notebooks; the functions they call live in `src/` (CONVENTIONS §7).
STAGES = {
    'data_creation':     '01_data_creation',
    'model_fitting':     '02_model_fitting',
    'model_performance': '03_model_performance',
    'coherent_ggi':      '04_coherent_ggi',
    'trend_analysis':    '05_trend_analysis',
}


def table_dir(stage, create=True):
    """`outputs/tables/<stage>/` — where a stage's date-stamped CSVs go"""
    path = TABLES / STAGES[stage]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def fig_dir(stage, create=True):
    """`outputs/fig/<stage>/` — where a stage's figures go"""
    path = FIG / STAGES[stage]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
RESULTS = OUTPUTS / 'results'   # -> Dropbox .../new_national_pipeline_files/results
MODELS = OUTPUTS / 'models'     # -> Dropbox .../new_national_pipeline_files/models
GRAPHS = OUTPUTS / 'graphs'     # -> Dropbox .../new_national_pipeline_files/graphs

DOC = ROOT / 'doc'
LOGS = ROOT / 'logs'

# Legacy path names used by the notebooks imported from dgg_research's `origin/pipeline` branch
# (D20). They resolve through the data/external symlink, so §1 still holds — no absolute path.
# New code should use RAW / PROCESSED / RESULTS / MODELS above.
DROPBOX_ROOT = EXTERNAL / 'national/data_refresh/new_national_pipeline_files'
dropbox_path = DROPBOX_ROOT
dropbox_data_path = DROPBOX_ROOT / 'data'
dropbox_result_path = DROPBOX_ROOT / 'results/pred_by_year_and_month'

# upstream Facebook pipeline artefacts, reached through data/external
FB_COUNTS = EXTERNAL / 'national/pipeline/files/preprocessed_counts'
FB_PREPROCESSED = EXTERNAL / 'pipeline/preprocessed/national'

# Legacy inputs of 01_facebook_data.py. These no longer exist on Dropbox; the script was
# superseded by 01_01_facebook_data_monthly.py, which reads FB_COUNTS via utils.read_fb_data.
# See doc/decisions.md (D6) before repointing these at FB_COUNTS.
FB_AVERAGED = EXTERNAL / 'fb_data/averaged/national'
FB_PRE_PIPELINE = EXTERNAL / 'fb_data/pre_pipeline_result'

# Natural Earth 110m admin-0 boundaries, used for the country maps. Replaces
# geopandas' bundled `naturalearth_lowres`, removed in GeoPandas 1.0.
WORLD_SHP = EXTERNAL / 'shape_files/ne_110m_admin_0_countries'

# ====================================================================================================
# controls
# ====================================================================================================
latest_contrl = False  # whether to keep the latest survey for each country
missing_contrl = True
rolling_contrl = True
del_country_contrl = False
model_folder = '18_plus'  # folder to store the models

SEED = 42  # §9: every randomised step reads this

# Treat parity as the ceiling: a country where women's predicted adoption exceeds men's has no gap
# against women, so movement above parity is not a widening gap and should not be read as one.
#
# Implemented by capping the FEMALE LEVEL at the male level (female* = min(female, male)) rather
# than capping the GGI itself. The two give the same series — GGI* = min(GGI, 1) — but capping the
# level keeps GGI* a genuine ratio, so the decomposition identity
#     delta_log_GGI* = delta_log(female*) + [-delta_log(male)]
# still holds exactly. Capping the ratio directly would break it. See D25.
level_ceiling_ctrl = True

# The mirror of the ceiling, at the other end. Where the predicted FEMALE level sits exactly on the
# zero floor, the coherent GGI is 0/male = 0 by construction — it reports the clip, not a gender
# gap, and no observation can agree with it. Those rows say nothing about the measure's accuracy
# and everything about the level model's floor (D19), so the validation in 14_/analysis 09_01
# reports the sample twice: with them and without.
#
# In the current sample this is Mali 2018 and Chad 2019 (internet only; no mobile row is affected).
#
# READ THIS BEFORE QUOTING THE FILTERED NUMBERS: the exclusion is selected on the *coherent*
# measure's own failure mode, which the direct prediction does not share. Dropping these rows and
# then saying "coherent beats direct" is circular. The filtered sample answers only "how does the
# coherent GGI do where it is defined at all". See D34.
delete_coherent_zero_ctrl = False

# Countries with no Facebook data, excluded from the published series. `dgg_pipeline` and the
# `origin/pipeline` branch of dgg_research both use CHN (China); the 'CHI' below is Channel Islands
# and never matched anything. CHN is what the published predictions actually filter on. See D21.
sanction_countries = ['RUS', 'SDN', 'CHN']
sactioned_countries = sanction_countries  # legacy misspelling, kept so older scripts still resolve

# Commonwealth of Independent States. This is the "CIS" in the `combined_with_CIS` model name: the
# variant fitted *with* these countries in the training sample (D12).
# NOTE: the upstream list writes Tajikistan as 'TKJ', which is not an ISO3 code — the real code is
# 'TJK'. Corrected here; see D21 for what the typo cost.
CIS_countries = ['BEL', 'KAZ', 'ARM', 'KGZ', 'MDA', 'AZE', 'TJK', 'UZB', 'TKM']

# Excluded from analysis: Yemen's level predictions are degenerate (female internet pinned at the
# zero floor throughout), which makes every derived ratio meaningless. Both the upstream convergence
# analysis and trend.qmd drop it explicitly. See D21.
excluded_countries = ['YEM']


# list
background_vars = ['abr', 'bicc_gmi', 'coef_ineq', 'eys', 'eys_r', 'gdi', 'gdp_pcap', 'gggi_eas', 'gggi_ggi', 'gggi_hss', 'gggi_pes', 'gggi_pos', 'gii', 'gni_pc', 'gni_pc_r', 'hcpi_a', 'hdi', 'hdi_r', 'ihdi', 'ineq_edu', 'ineq_inc', 'ineq_le', 'le', 'le_r', 'lfpr_r', 'mmr', 'mys', 'mys_r', 'pov_hr', 'se_r', 'une_girlglsr', 'une_girlglst', 'une_girlgpr', 'une_girlgpt', 'vdem_gender', 'wbgi_pve', 'wdi_acel', 'wdi_fertility', 'wdi_gert', 'wdi_gertr', 'wdi_litrad', 'wdi_litradr', 'pr_r','wdi_unemp','wdi_unempr','wdi_internet']
fb_vars_18_plus = ['fb_18_999_r', 'fb_18_999_wom', 'fb_18_999_men']
fb_vars = ['fb_18_999_r', 'fb_18_999_wom', 'fb_18_999_men', 'fb_14_15_r', 'fb_14_15_wom', 'fb_14_15_men', 'fb_15_16_r', 'fb_15_16_wom', 'fb_15_16_men', 'fb_16_17_r', 'fb_16_17_wom', 'fb_16_17_men', 'fb_17_18_r', 'fb_17_18_wom', 'fb_17_18_men', 'fb_18_19_r', 'fb_18_19_wom', 'fb_18_19_men', 'fb_13_14_r', 'fb_13_14_wom', 'fb_13_14_men', 'fb_15_19_r', 'fb_15_19_wom', 'fb_15_19_men', 'fb_20_24_r', 'fb_20_24_wom', 'fb_20_24_men', 'fb_25_29_r', 'fb_25_29_wom', 'fb_25_29_men', 'fb_30_34_r', 'fb_30_34_wom', 'fb_30_34_men', 'fb_35_39_r', 'fb_35_39_wom', 'fb_35_39_men', 'fb_40_44_r', 'fb_40_44_wom', 'fb_40_44_men', 'fb_45_49_r', 'fb_45_49_wom', 'fb_45_49_men', 'fb_50_54_r', 'fb_50_54_wom', 'fb_50_54_men', 'fb_55_59_r', 'fb_55_59_wom', 'fb_55_59_men', 'fb_60_64_r', 'fb_60_64_wom', 'fb_60_64_men', 'fb_18_23_r', 'fb_18_23_wom', 'fb_18_23_men', 'fb_20_999_r', 'fb_20_999_wom', 'fb_20_999_men', 'fb_20_64_r', 'fb_20_64_wom', 'fb_20_64_men', 'fb_21_999_r', 'fb_21_999_wom', 'fb_21_999_men', 'fb_25_999_r', 'fb_25_999_wom', 'fb_25_999_men', 'fb_25_49_r', 'fb_25_49_wom', 'fb_25_49_men', 'fb_25_64_r', 'fb_25_64_wom', 'fb_25_64_men', 'fb_50_999_r', 'fb_50_999_wom', 'fb_50_999_men', 'fb_60_999_r', 'fb_60_999_wom', 'fb_60_999_men', 'fb_65_999_r', 'fb_65_999_wom', 'fb_65_999_men', 'fb_all_r', 'fb_android_device_users_r', 'fb_android_device_users_wom', 'fb_android_device_users_men', 'fb_ios_device_users_r', 'fb_ios_device_users_wom', 'fb_ios_device_users_men', 'fb_mobile_device_users_r', 'fb_mobile_device_users_wom', 'fb_mobile_device_users_men']
age_cols_dict = {'18_999': ['FB_age_18_plus_ratio', '18_inf_r'], '14_15': ['FB_age_14_15_ratio', '14_15_r'], '15_16': ['FB_age_15_16_ratio', '15_16_r'], '16_17': ['FB_age_16_17_ratio', '16_17_r'], '17_18': ['FB_age_17_18_ratio', '17_18_r'], '18_19': ['FB_age_18_19_ratio', '18_19_r'], '13_14': ['FB_age_13_14_ratio', '13_14_r'], '15_19': ['FB_age_15_19_ratio', '15_19_r'], '20_24': ['FB_age_20_24_ratio', '20_24_r'], '25_29': ['FB_age_25_29_ratio', '25_29_r'], '30_34': ['FB_age_30_34_ratio', '30_34_r'], '35_39': ['FB_age_35_39_ratio', '35_39_r'], '40_44': ['FB_age_40_44_ratio', '40_44_r'], '45_49': ['FB_age_45_49_ratio', '45_49_r'], '50_54': ['FB_age_50_54_ratio', '50_54_r'], '55_59': ['FB_age_55_59_ratio', '55_59_r'], '60_64': ['FB_age_60_64_ratio', '60_64_r'], '18_23': ['FB_age_18_23_ratio', '18_23_r'], '20_999': ['FB_age_20_plus_ratio', '20_inf_r'], '20_64': ['FB_age_20_64_ratio', '20_64_r'], '21_999': ['FB_age_21_plus_ratio', '21_inf_r'], '25_999': ['FB_age_25_plus_ratio', '25_inf_r'], '25_49': ['FB_age_25_49_ratio', '25_49_r'], '25_64': ['FB_age_25_64_ratio', '25_64_r'], '50_999': ['FB_age_50_plus_ratio', '50_inf_r'], '60_999': ['FB_age_60_plus_ratio', '60_inf_r'], '65_999': ['FB_age_65_plus_ratio', '65_inf_r']}

pop_cols = ['16_17_r', '20_64_r', '18_inf_r', '14_15_r', '18_23_r', '13_14_r', '45_49_r', '60_inf_r', '17_18_r', '65_inf_r', '50_54_r', '50_inf_r', '55_59_r', '20_inf_r', '40_44_r', '25_64_r', '60_64_r', '25_inf_r', '15_19_r', '15_16_r', '20_24_r', '30_34_r', '35_39_r', '25_29_r', '25_49_r', '21_inf_r', '18_19_r']

# all_countries = ['SEN', 'AGO', 'ARM', 'BEN', 'BDI', 'CMR', 'ETH', 'GAB', 'GIN', 'HTI', 'IND', 'IDN', 'JOR', 'LBR', 'MWI', 'MDV', 'MLI', 'MRT', 'NPL', 'NGA', 'PAK', 'PNG', 'SLE', 'ZAF', 'TZA', 'TLS', 'UGA', 'ZMB', 'ZWE', 'BFA', 'KHM', 'CIV', 'GMB', 'GHA', 'KEN', 'MDG', 'RWA', 'ALB', 'CAF', 'TCD', 'COM', 'CUB', 'COD', 'SWZ', 'FJI', 'GNB', 'GUY', 'KIR', 'LAO', 'LSO', 'MNG', 'WSM', 'STP', 'SUR', 'TGO', 'TON', 'TUN', 'TCA', 'TUV', 'VNM', 'DZA', 'AND', 'ARG', 'AUS', 'AUT', 'AZE', 'BHS', 'BHR', 'BGD', 'BLR', 'BEL', 'BLZ', 'BTN', 'BOL', 'BIH', 'BWA', 'BRA', 'VGB', 'BRN', 'BGR', 'CPV', 'CAN', 'CHL', 'CHN', 'COL', 'CRI', 'HRV', 'CUW', 'CYP', 'CZE', 'DNK', 'DJI', 'DOM', 'ECU', 'EGY', 'SLV', 'EST', 'FIN', 'FRA', 'GEO', 'DEU', 'GRC', 'GTM', 'HND', 'HKG', 'HUN', 'ISL', 'IRN', 'IRQ', 'IRL', 'ISR', 'ITA', 'JAM', 'JPN', 'KAZ', 'KOR', 'XKX', 'KWT', 'LVA', 'LBN', 'LTU', 'LUX', 'MAC', 'MYS', 'MLT', 'MUS', 'MEX', 'MNE', 'MSR', 'MAR', 'MOZ', 'MMR', 'NLD', 'NZL', 'NIC', 'NER', 'MKD', 'NOR', 'OMN', 'PAN', 'PRY', 'PER', 'POL', 'PRT', 'PRI', 'QAT', 'ROU', 'RUS', 'SAU', 'SRB', 'SGP', 'SVK', 'SVN', 'ESP', 'LKA', 'PSE', 'SDN', 'SWE', 'CHE', 'TWN', 'THA', 'TTO', 'TUR', 'UKR', 'ARE', 'GBR', 'USA', 'URY', 'UZB', 'VEN', 'GRD', 'PRK', 'PHL']

bg_cols = ['abr', 'bicc_gmi', 'coef_ineq', 'eys', 'eys_r', 'gdi', 'gdp_pcap', 'gggi_eas', 'gggi_ggi', 'gggi_hss', 'gggi_pes', 'gggi_pos', 'gii', 'gni_pc', 'gni_pc_r', 'hcpi_a', 'hdi', 'hdi_r', 'ihdi', 'ineq_edu', 'ineq_inc', 'ineq_le', 'le', 'le_r', 'lfpr_r', 'mmr', 'mys', 'mys_r', 'pov_hr', 'se_r',
           'une_girlglsr', 'une_girlglst', 'une_girlgpr', 'une_girlgpt', 'vdem_gender', 'wbgi_pve', 'wdi_acel', 'wdi_fertility', 'wdi_gert', 'wdi_gertr', 'wdi_litrad', 'wdi_litradr', 'pr_r', 'wdi_unemp', 'wdi_unempr', 'wdi_internet', 'educ_hdi_r']

bg_col_dict = {'abr': 'Adolescent birth rate', 'bicc_gmi': 'Global Militarization Index', 'coef_ineq': 'Coefficient of human inequality', 'conti': 'Continent', 'educ_hdi_r': 'Education inequality ratio', 'eys': 'Expected years of schooling', 'eys_r': 'Expected years of schooling ratio', 'gdi': 'Gender development index', 'gdp_pcap': 'Logged GDP per capita', 'gggi_eas': 'Global Gender Gap Educational Attainment Subindex', 'gggi_ggi': 'Overall Global Gender Gap Index', 'gggi_hss': 'Global Gender Gap Health and Survival Subindex', 'gggi_pes': 'Global Gender Gap Political Empowerment Subindex', 'gggi_pos': 'Global Gender Gap Economic Participation and Opportunity Subindex', 'gii': 'UN Gender inequality index', 'gni_pc': 'GNI per capita', 'gni_pc_r': 'GNI per capita ratio', 'hcpi_a': 'Headline CPI inflation', 'hdi': 'Human development index', 'hdi_r': 'Human development index ratio', 'ihdi': 'Inequality-adjusted Human Development Index', 'ineq_edu': 'Inequality-adjusted education index', 'ineq_inc': 'Inequality-adjusted income index', 'ineq_le': 'Inequality-adjusted life expectancy index', 'le': 'Life expectancy at birth', 'le_r': 'Life expectancy at birth ratio', 'lfpr_r': 'Female labor force participation rates (%)', 'mmr': 'Maternal mortality ratio', 'mys': 'Mean years of schooling', 'mys_r': 'Mean years of schooling ratio', 'pov_hr': 'Poverty headcount ratio at \n$2.15 a day (2017 PPP) (%)', 'pr_r': 'shares of parliamentary seats ratio', 'se_r': 'Population with at leat secondary eductions ratio', 'une_girlglsr': 'Gross intake ratio to the last grade of \nlower secondary general education ratio', 'une_girlglst': 'Gross intake ratio to the last grade of \nlower secondary general education (%)', 'une_girlgpr': 'Gross intake ratio to the \nlast grade of primary education ratio', 'une_girlgpt': 'Gross intake ratio to the \nlast grade of primary education (%)', 'vdem_gender': 'Women political empowerment index', 'wbgi_pve': 'Political Stability and \nAbsence of Violence/Terrorism', 'wdi_acel': 'Access to electricity (%)', 'wdi_fertility': 'Total fertility rate', 'wdi_gert': 'School enrollment, tertiary (% gross)', 'wdi_gertr': 'School enrollment at tertiary level ratio', 'wdi_internet': 'Internet usage in the last 3 months', 'wdi_litrad': 'Literacy rate', 'wdi_litradr': 'Literacy ratio', 'wdi_unemp': 'Unemployment rate', 'wdi_unempr': 'Unemployment ratio'}

model_datasets = {
    "online": ['combined_multiple_years_2021-06.csv', 'combined_multiple_years_2022-06.csv', 'combined_multiple_years_2023-06.csv', 'combined_multiple_years_2024-06.csv'],
    "offline": ['combined_multiple_years_2022-06.csv', 'combined_multiple_years_no_missing_2022-06.csv'],
    "combined": ['combined_multiple_years_2021-06.csv', 'combined_multiple_years_2022-06.csv', 'combined_multiple_years_2023-06.csv', 'combined_multiple_years_2024-06.csv', 'combined_multiple_years_no_missing_2021-06.csv', 'combined_multiple_years_no_missing_2022-06.csv',
                 'combined_multiple_years_no_missing_2023-06.csv', 'combined_multiple_years_no_missing_2024-06.csv']
}

best_model_specs = {'internet':{'offline':{"var_set":['se_r', 'wdi_fertility', 'pov_hr', 'gdi', 'gggi_ggi', 'ineq_inc', 'wdi_internet', 'wdi_acel', 'wdi_unempr', 'lfpr_r', 'abr', 'hcpi_a', 'bicc_gmi', 'le_r', 'une_girlgpt', 'vdem_gender', 'wbgi_pve', 'hdi_r', 'mys_r', 'une_girlglsr', 'gdp_pcap', 'hdi', 'eys', 'ihdi', 'gni_pc', 'educ_hdi_r', 'wdi_unemp', 'ineq_edu', 'mys', 'gni_pc_r', 'gggi_pos'],
                                      "missing":False,
                                      "year":2022},
                           'combined':{"var_set":['se_r', 'wdi_fertility', 'wdi_litrad', 'pov_hr', 'fb_30_34', 'fb_25_29', 'une_girlglsr', 'hcpi_a', 'wdi_internet', 'mys', 'wdi_litradr', 'wdi_acel', 'fb_ios_device_users_ratio', 'fb_25_49', 'fb_16_17', 'fb_15_16', 'une_girlgpt', 'bicc_gmi', 'wbgi_pve', 'une_girlgpr', 'vdem_gender', 'fb_40_44', 'fb_20_64', 'fb_18_999', 'wdi_gert'],
                                       "missing":True,
                                       "year":2024}},

               "mobile":  {'offline':{"var_set":['wdi_litrad', 'wbgi_pve', 'wdi_unemp', 'wdi_internet', 'abr', 'gdi', 'pov_hr', 'hcpi_a', 'gggi_pes', 'une_girlgpr', 'une_girlgpt', 'wdi_fertility', 'ineq_inc', 'gii', 'le_r', 'gni_pc', 'gdp_pcap', 'bicc_gmi', 'wdi_litradr', 'une_girlglst', 'lfpr_r', 'le', 'ineq_le', 'pr_r', 'mmr', 'hdi_r', 'mys_r', 'se_r', 'gggi_eas', 'mys'],
                                      "missing":False,
                                      "year": 2022},

                           'combined':{"var_set":['wdi_litrad', 'wbgi_pve', 'wdi_unemp', 'wdi_internet', 'fb_android_device_users_ratio', 'hcpi_a', 'gii', 'gggi_pes', 'fb_all', 'pov_hr', 'wdi_litradr', 'fb_60_64', 'une_girlgpt', 'wdi_fertility', 'fb_25_999', 'ineq_le', 'wdi_gertr', 'une_girlglst', 'fb_18_999', 'abr', 'une_girlgpr', 'le', 'vdem_gender', 'fb_65_999', 'fb_45_49', 'fb_20_64', 'bicc_gmi', 'lfpr_r', 'hdi_r', 'le_r', 'ineq_inc', 'gggi_eas', 'fb_25_29', 'fb_55_59', 'fb_50_54', 'fb_35_39', 'gdi'],
                                       "missing":False,
                                       "year": 2022}}}
# ====================================================================================================
# the final (production) model
# ====================================================================================================
# What dgg_pipeline/src/modelling/national_model.py actually loads to produce the published
# national estimates: an OLS on the ITU-deleted pooled panel. Verified by refitting — the
# coefficients reproduce the shipped pickles to ~1e-15. See doc/decisions.md D12.
FINAL_MODEL = {
    'model_folder': 'OLS',
    'model_type': 'combined_with_CIS',
    'dataset': 'combined_multiple_years_no_missing_keep_countries_fb_aligned_itu_deleted.csv',
    'spec': ['fb_18_999_men', 'fb_18_999_wom', 'fb_18_999_r', 'hdi', 'gdi', 'gdp_pcap', 'year'],
    'year_origin': 2015,   # the `year` regressor is {indicator}_year - year_origin
    'indicators': ['internet', 'mobile'],
    'outcome_vars': ['ggi', 'wom', 'men'],
    # LOCO is the headline validation. The ITU-deleted panel keys on iso3 (it carries no
    # `country` column); the shipped betas file is named `..._country_...` for historical reasons.
    'leave_column': 'iso3',
}

# filename pattern shared by the fitted pickles and by dgg_pipeline's loader
FINAL_MODEL_FILENAME = '{indicator}_{model_type}_{indicator}_{outcome_var}_full_model.pkl'

# ----------------------------------------------------------------------------------------------------
# the model variants that exist as fitted pickles, and how each one was actually built
# ----------------------------------------------------------------------------------------------------
# Established by matching every shipped pickle's coefficients against a refit, to ~1e-15 (D37).
# All variants read the SAME file — FINAL_MODEL['dataset'] — and differ in two things only:
# which rows they keep, and which regressors they use.
#
# `keep_itu` is the whole of the sample difference. The panel is named `..._itu_deleted` because an
# earlier ITU-only file was dropped from it, NOT because it has no ITU rows: 37 of its 108 internet
# rows and 24 of its 99 mobile rows are ITU.
#
# THE `_with_CIS` SUFFIX IS A MISNOMER and the single biggest source of confusion here. It does not
# select CIS countries. It means "keep the ITU rows", which incidentally brings four CIS countries
# (AZE, BEL, KAZ, UZB) into the sample — they appear in the panel through ITU rows and nowhere else.
# The filter is ITU; CIS membership is a side-effect of it.
_FB = ['fb_18_999_men', 'fb_18_999_wom', 'fb_18_999_r']
_OFFLINE = ['hdi', 'gdi', 'gdp_pcap', 'year']

MODEL_VARIANTS = {
    # variant                        spec                     keep ITU rows?   n (internet/mobile)
    'online':                        dict(spec=_FB,              keep_itu=False),   # 71 / 75
    'offline':                       dict(spec=_OFFLINE,         keep_itu=False),   # 71 / 75
    'combined':                      dict(spec=_FB + _OFFLINE,   keep_itu=False),   # 71 / 75
    'combined_top_shaps_no_fertility': dict(
        spec=['gggi_ggi', 'se_r', 'hdi', 'gdp_pcap', 'year'] + _FB, keep_itu=False),  # 71 / 75
    'online_with_CIS':               dict(spec=_FB,              keep_itu=True),    # 108 / 99
    'offline_with_CIS':              dict(spec=_OFFLINE,         keep_itu=True),    # 108 / 99
    'combined_with_CIS':             dict(spec=_FB + _OFFLINE,   keep_itu=True),    # 108 / 99  <- PRODUCTION
    # `_align` variants carry ONE Facebook term, aligned to the outcome being predicted:
    # `fb_18_999_wom` for the women's level, `fb_18_999_men` for the men's, `fb_18_999_r` for the
    # ratio. `align=True` means the literal `fb_18_999_r` in `spec` is substituted per outcome.
    # (`offline_with_CIS_align` has no Facebook term, so it is identical to `offline_with_CIS`.)
    'online_with_CIS_align':         dict(spec=['fb_18_999_r'],  keep_itu=True, align=True),
    'offline_with_CIS_align':        dict(spec=_OFFLINE,         keep_itu=True, align=True),
    'combined_with_CIS_align':       dict(spec=_OFFLINE + ['fb_18_999_r'], keep_itu=True, align=True),
}


def resolve_spec(variant, outcome_var):
    """
    the regressors a variant uses for one outcome

    Only the `_align` variants depend on the outcome: they swap the single Facebook term for the
    one matching the outcome, so the women's level is predicted from the women's audience share
    rather than from the gender ratio. Every other variant returns its spec unchanged.
    """
    cfg = MODEL_VARIANTS[variant]
    spec = list(cfg['spec'])
    if cfg.get('align') and outcome_var in ('wom', 'men'):
        spec = [f'fb_18_999_{outcome_var}' if c == 'fb_18_999_r' else c for c in spec]
    return spec

# The three bars of the published performance figure are these variants, on the production sample.
FIGURE_VARIANTS = ['online_with_CIS', 'offline_with_CIS', 'combined_with_CIS']

# ====================================================================================================
# coherent GGI (src/13_coherent_ggi.py)
# ====================================================================================================
# The internally coherent gender gap index, derived from the predicted female and male levels
# rather than predicted directly. See doc/methodology.md and doc/decisions.md D16.
COHERENT_GGI = {
    # dgg_pipeline's published series is the golden standard: read-only, never written to.
    'source': EXTERNAL / 'pipeline/result/national/aggregate_files.csv',
    'indicators': ['internet', 'mobile'],
    'parity_cap': 1.0,    # the primary measure is min(raw, 1) — values >1 are out of scope
    'near_zero': 1e-6,    # male levels below this are not divided by; the row is flagged instead
    'tolerance': 1e-9,    # numerical tolerance for the assertions
    'regions': RAW / 'iso3_regions.csv',   # World Bank regions, all 214 countries covered
    # The national models use fb_vars_18_plus only, so there is a single age group. Carried as an
    # explicit key so an age-disaggregated series can slot in without changing the grain.
    'age_group': '18_plus',
    'focus_years': [2015, 2020, 2025],
    # Survey ground truth for the validation panel: the harmonised DHS/MICS outcomes written by
    # 02_ground_truth_data_calculation.qmd. Glob, so the newest dated run is picked up (§3).
    'groundtruth_glob': str(EXTERNAL / 'national/adolescent_modelling/update_full_groundtruth_*.csv'),
    # The fitted panels, used only to split validation rows into in-sample and held-out.
    'training_panels': PROCESSED / 'combined_data/updated_ground_truth_and_fb',
    'training_file': 'combined_multiple_years_no_missing_keep_countries_fb_aligned_itu_deleted.csv',
}

countries_to_exclude_for_imputation = {'mobile': ['ABW', 'AND', 'ASM', 'ATG', 'BMU', 'CHI', 'CSK', 'CUW', 'CYM', 'DDR', 'DMA', 'FRO', 'GIB', 'GRL', 'GUM', 'HKG', 'IMN', 'KNA', 'MAC', 'MAF', 'MCO', 'MNP', 'MSR', 'MTQ', 'NCL', 'NIU', 'NRU', 'PRI', 'PRK', 'PSE', 'PYF', 'SAS', 'SCG', 'SHN', 'SUN', 'SXM', 'TCA', 'TWN', 'VDR', 'VGB', 'VIR', 'XKX', 'XTI', 'XXK', 'YMD', 'YUG'],
                                       'internet':['ABW', 'ASM', 'ATG', 'BMU', 'CHI', 'CSK', 'CUW', 'CYM', 'DDR', 'DMA', 'FRO', 'GIB', 'GRL', 'GUM', 'HKG', 'IMN', 'KNA', 'MAC', 'MAF', 'MCO', 'MNP', 'MSR', 'MTQ', 'NCL', 'NIU', 'PRI', 'PRK', 'PSE', 'PYF', 'SAS', 'SCG', 'SHN', 'SUN', 'SXM', 'TCA', 'TWN', 'VDR', 'VGB', 'VIR', 'XKX', 'XTI', 'XXK', 'YMD', 'YUG']}

####### codes to generate lists
"""
age_cols_dict = {}
for age_range in age_lst:
    
    fb_col = f'FB_age_{age_range[0]}_{age_range[1]}_ratio' if age_range[1]!=999 else f'FB_age_{age_range[0]}_plus_ratio'
    pop_col = f'{age_range[0]}_{age_range[1]}_r' if age_range[1]!=999 else f'{age_range[0]}_inf_r'
    age_cols_dict[f'{age_range[0]}_{age_range[1]}']=[fb_col,pop_col]

"""
# Project palette, rebuilt 2026-08-05 for colour-vision accessibility (D23).
# Constructed in OKLCH so every hue sits inside the readable lightness band with enough chroma not
# to read as grey, then ordered to maximise separation under simulated protanopia and deuteranopia.
# The previous palette failed both discriminability checks (purple vs blue: ΔE 8.2 normal-vision,
# 2.8 protan, against floors of 15 and 8); this one reaches 23.3 and 11.2.
colors = {
    'teal':    '#009F89',
    'orange':  '#E18528',
    'purple':  '#6C44A4',
    'rose':    '#DD7577',
    'sky':     '#43B2E1',
    'olive':   '#85861F',
    'blue':    '#2D72C4',
    'magenta': '#B53C7F',
}

# Names used by the older scripts, mapped onto the nearest new hue so they keep resolving.
colors.update({
    'light_blue': colors['sky'],
    'red': colors['rose'],
    'peach': colors['orange'],
    'soft_pink': colors['magenta'],
    'warm_yellow': colors['olive'],
})

# ====================================================================================================
# visualisation style (§6) — every figure reads from here, nothing hardcoded inline
# ====================================================================================================
# Categorical assignment order: the sequence that maximises the worst adjacent separation under
# simulated protanopia and deuteranopia. Assign hues in this order and never cycle past the eighth
# slot — a ninth series folds into "Other", small multiples, or a highlight-plus-context treatment.
plot_color_order = ['teal', 'orange', 'purple', 'rose', 'sky', 'olive', 'blue', 'magenta']

# Sub-Saharan Africa subregion split used by the trend figures, preserved from
# analysis/technical_report/trend.qmd. `region2` is the World Bank region, except that
# Sub-Saharan Africa is replaced by these subregions.
ssa_subregions = {
    'Eastern Africa': ['BDI', 'COM', 'DJI', 'ERI', 'ETH', 'KEN', 'MDG', 'MWI', 'MUS', 'MOZ',
                       'REU', 'RWA', 'SYC', 'SOM', 'SSD', 'TZA', 'UGA', 'ZMB', 'ZWE'],
    'Central Africa': ['AGO', 'CMR', 'CAF', 'TCD', 'COD', 'COG', 'GNQ', 'GAB', 'STP'],
    'Western Africa': ['BEN', 'BFA', 'CPV', 'CIV', 'GMB', 'GHA', 'GIN', 'GNB', 'LBR', 'MLI',
                       'MRT', 'NER', 'NGA', 'SEN', 'SLE', 'TGO'],
    'Southern Africa': ['BWA', 'LSO', 'NAM', 'SWZ', 'ZAF'],
}

STYLE = {
    "colors":   [colors[k] for k in plot_color_order],
    "figsize":  (12, 7),
    "title_fs": 16,
    "label_fs": 13,
    "tick_fs":  11,
    "dpi":      300,
    "save":     False,          # True -> write figures to savedir
    "savedir":  FIG,
}

# `utils.print_models_with_loco` orders regressors by fb then background columns.
# fb_cols was referenced but never defined; it is the fb variable list.
fb_cols = fb_vars

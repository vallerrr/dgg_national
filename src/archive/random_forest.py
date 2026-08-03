import utils
import params
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import os
import joblib

# Initialize an empty DataFrame to store the results
results_df = pd.DataFrame(columns=['model_name', 'method', 'outcome_var', 'model_type', 'best_adj_r2', 'best_r2', 'loco_r2', 'loyo_r2','loconto_r2','selected_var'])
country_specific_df = pd.DataFrame(columns=["name",'outcome_var','model_type','country','pred','true','train_size'])
year_specific_df = pd.DataFrame(columns=["name", 'outcome_var','model_type','year','pred','true','train_size'])
continent_specific_df = pd.DataFrame(columns=["name",'outcome_var','model_type','conti','pred','true','train_size'])


fb_cols = params.fb_vars
bg_cols = params.bg_cols
model_specs = {'online': fb_cols+['conti'], 'offline': bg_cols, 'combined': fb_cols + bg_cols}

# dataset = "combined_multiple_years_no_missing_fb_aligned.csv"
dataset = "combined_multiple_years_no_missing_keep_countries_fb_aligned.csv"
def extract_info_from_filename(filename):
    if "no_missing" in filename:
        missing_imputed = True
    else:
        missing_imputed = False

    year = filename.split('_')[-1].split('-')[0]
    return missing_imputed, year


def preprocess_data(data):
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    if len(non_numeric_cols) > 0:
        data = pd.get_dummies(data, columns=non_numeric_cols)
    return data


def save_model(model, file_path):
    joblib.dump(model, file_path)


def load_model(file_path):
    return joblib.load(file_path) if os.path.exists(file_path) else None


def evaluate_random_forest_leave_one_out(indicator, dataset, model_type, country_specific_df, year_specific_df, continent_specific_df, outcome_var):
    # Load the dataset
    data = pd.read_csv(data_path / dataset)
    model_spec = model_specs[model_type]

    # Drop rows where the target variable is missing
    data = data.dropna(subset=[f'{outcome_var}'])
    data.rename(columns={f'{indicator}_year':'year'}, inplace=True)

    # Preprocess data
    X = preprocess_data(data[model_spec])
    y = data[outcome_var]

    # Check if a model for this specific validation already exists
    model_filename = f'{indicator}_{model_type}_{outcome_var}_full_model.pkl'
    model_filepath = params.MODELS / 'rf' / model_filename

    # Try to load the model if it exists
    rf_loo = load_model(model_filepath)

    # If no model exists, train a new one
    if rf_loo is None:
        rf_loo = RandomForestRegressor(random_state=42)
        rf_loo.fit(X, y)
        save_model(rf_loo, model_filepath)

    y_pred_full = rf_loo.predict(X)
    r2_full_sample = r2_score(y, y_pred_full)


    # leave one out validation
    def leave_one_column_out(column):
        recorder = pd.DataFrame(columns=["name", 'outcome_var', 'model_type', column, 'pred', 'true', 'train_size'])
        for unique_value in data[column].unique():
            X_train_loo = X[data[column] != unique_value]
            y_train_loo = y[data[column] != unique_value]

            X_test_loo = X[data[column] == unique_value]
            y_test_loo = y[data[column] == unique_value]

            # Check if a model for this specific validation already exists
            model_filename = f'{indicator}_{model_type}_{outcome_var}_{column}_{unique_value}_model.pkl'
            model_filepath = params.MODELS / 'rf' / model_filename

            # Try to load the model if it exists
            rf_loo = load_model(model_filepath)

            # If no model exists, train a new one
            if rf_loo is None:
                rf_loo = RandomForestRegressor(random_state=42)
                rf_loo.fit(X_train_loo, y_train_loo)
                save_model(rf_loo, model_filepath)

            y_pred_loo = rf_loo.predict(X_test_loo)
            for i in range(len(y_pred_loo)):
                recorder.loc[len(recorder)] = [f'{indicator} {model_type}', outcome_var, model_type, unique_value, y_pred_loo[i], y_test_loo.values[i], len(y_train_loo)]

        r2 = r2_score(recorder['true'], recorder['pred'])
        return recorder, r2

    r2s = {}
    for leave_column in ['country', 'year', 'conti']:
        temp, r2s[leave_column] = leave_one_column_out(leave_column)
        if leave_column == 'country':
            country_specific_df = pd.concat([country_specific_df, temp], axis=0)
        elif leave_column == 'year':
            year_specific_df = pd.concat([year_specific_df, temp], axis=0)
        # else:
        #     continent_specific_df = pd.concat([continent_specific_df,temp], axis=0)

    # ----------------------------------------
    # Append the results to the main DataFrame
    results_df.loc[len(results_df)] = {
        "model_name": f'{indicator} {model_type}',
        'method': 'Random Forest',
        'outcome_var': outcome_var,
        "model_type": model_type,
        'best_r2': r2_full_sample,
        'loco_r2': r2s['country'],
        'loyo_r2': r2s['year'],
        # 'loconto_r2':r2s['conti'],
        'selected_var': ''
    }

    return year_specific_df, country_specific_df, continent_specific_df, results_df

# Loop through each dataset and evaluate the model using leave-one-country-out validation
for indicator in ['internet', 'mobile']:
    data_path = params.PROCESSED / f"combined_data/updated_ground_truth_and_fb/{indicator}"
    for model_type in ['online', 'offline', 'combined']:
        for outcome_var in [f'{indicator}_ggi', f'{indicator}_wom', f'{indicator}_men']:
            year_specific_df, country_specific_df, continent_specific_df, results_df = evaluate_random_forest_leave_one_out(indicator, dataset, model_type, country_specific_df, year_specific_df, continent_specific_df, outcome_var)

# Saving results
"""
country_specific_df.to_csv(params.RESULTS / 'logs/random_forest_loco_results.csv', index=False)
year_specific_df.to_csv(params.RESULTS / 'logs/random_forest_loyo_results.csv', index=False)
continent_specific_df.to_csv(params.RESULTS / 'logs/random_forest_loconto_results.csv', index=False)
results_df.to_csv(params.RESULTS / 'random_forest_results_summary.csv', index=False)
"""

# compare results_df and results_df_original(with the imputation)

results_df_original = pd.read_csv(params.RESULTS / 'random_forest_results_summary.csv')

results_df['method'] = ['Random Forest \n with imputed data']*len(results_df)



import matplotlib.pyplot as plt
colors = params.colors
df_res = pd.concat([results_df_original, results_df], axis=0)
indicator = 'internet'
# Grouping by outcome_var
grouped = df_res.loc[df_res['outcome_var'].str.contains(indicator)].groupby(['outcome_var', 'model_type'])

# Creating subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10) )
# set the portion of first column of subplots to be 0.4


axes = axes.flatten()
count = 0

for ((outcome_var, model_type), group), ax in zip(grouped, axes):
    #group['method'] = pd.Categorical(group['method'], categories=['on',], ordered=True)


    group = group.sort_values(by='method')  # Sorting by model_type within each outcome_var

    # Defining the position for each bar
    bar_width = 0.25  # Width of each bar
    x = range(len(group['method']))

    # Plotting the bars
    bars1 = ax.bar([pos - bar_width for pos in x], group['loco_r2'], width=bar_width, align='center', label='LOCO R2', color=colors['blue'])
    bars2 = ax.bar(x, group['loyo_r2'], width=bar_width, align='center', label='LOYO R2', color=colors['red'])
    #bars3 = ax.bar([pos + bar_width for pos in x], group['best_r2'], width=bar_width, align='center', label='Best R2', color=colors['peach'])

    # Adding the values
    for bars in [bars1, bars2]: #, bars3]:
        for bar in bars:
            yval = round(bar.get_height(), 2)
            ax.text(bar.get_x() + bar.get_width() / 2, yval, yval, ha='center', va='bottom', fontsize=10)

    # Add legend on the bottom plot
    if count == 7:
        ax.legend(bbox_to_anchor=(0.5, -0.4), loc='lower center', ncol=3)

    # Set title, labels, and grid
    ax.set_title(f"{model_type}")
    ax.set_xticks(x)

    #if count % 3 != 1:
    ax.set_xticklabels(group['method'])
    #else:
    #    ax.set_xticklabels([f'{x}' for x in zip(group['method'])])

    # Remove all spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Keep the y-axis spine only for the leftmost plots
    if count % 3 == 0:  # First column in each row
        ax.spines['left'].set_visible(True)
        ax.set_ylabel(f"{outcome_var.replace('_', ' ').replace(indicator,'')}\nR2")
    else:
        ax.set_yticks([])  # Remove y-axis ticks for other plots

    # ax.grid(axis='y', linestyle='--', alpha=0.6)
    # ax.yaxis.grid(True)

    count += 1

# Adjust layout and show the plot
fig.suptitle(f'{indicator}', fontsize=18)
plt.tight_layout()
# plt.savefig(params.GRAPHS / 'result_check/random_forest_results_compare.png')
plt.show()


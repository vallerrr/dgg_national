"""
# Created by valler at 26/08/2024
Feature: different validation results of the best models for each indicator

"""

import params
import pandas as pd
from seaborn import color_palette
import matplotlib.pyplot as plt

# ====================================================================================================
# 1. read results and plot
# ====================================================================================================
colors = params.colors

# df_res_linear = pd.read_csv(params.RESULTS / 'forward_selection_lasso_results.csv')
df_res_linear = pd.read_csv(params.RESULTS / 'baseline_var_select_aligned_comparison_result.csv')
df_res_linear.drop(df_res_linear.loc[(df_res_linear['model_type'] == 'offline') & (df_res_linear['fb_sets'] == 'baseline')].index, inplace=True)

df_res_rf = pd.read_csv(params.RESULTS / 'random_forest_results_summary.csv')
df_res_rf['model_name']=[f"{x}_{y.replace('combined','combine')}" for x,y in df_res_rf['model_name'].str.split(' ')]
df_res_rf['model_type'].replace({'combined':'combine'},inplace=True)

df_res = pd.concat([df_res_linear, df_res_rf], axis=0)
if 'fb_sets' in df_res.columns.to_list():
    df_res['fb_sets'].fillna('all',inplace=True)
    df_res['fb_sets'].replace({'not applicable':'all'}, inplace=True)
df_res.sort_values(by=['outcome_var', 'model_type','fb_sets'], inplace=True)


indicator = 'mobile'
# Grouping by outcome_var
grouped = df_res.loc[df_res['outcome_var'].str.contains(indicator)].groupby(['outcome_var', 'model_type'])

# Creating subplots
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(21, 15),gridspec_kw={'width_ratios': [0.4, 0.2, 0.4]} )
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
    ax.set_title(f" {model_type}")
    ax.set_xticks(x)

    if count % 3 != 1:
        ax.set_xticklabels([f'{x}\n({y})'for x,y in zip(group['method'],group['fb_sets'])], rotation=70)
    else:
        ax.set_xticklabels([f'{x}\n({y})' for x, y in zip(group['method'], group['fb_sets'])])

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

#plt.show()
plt.savefig(params.GRAPHS / f'{indicator}_loco_loyo_best_r2_comparison.pdf')

# =============================================================================
# 2. select best models
# =============================================================================

grouped = df_res.groupby(['outcome_var','model_type'],as_index=False)
temp_best = pd.DataFrame(columns =df_res.columns)
for outcome_var,group in grouped:
    group.reset_index(inplace=True,drop=True)
    temp_best.loc[len(temp_best),]= group.loc[group['loco_r2'].idxmax(),]

# visualise temp_best by outcome_var in one plot and model_type in different colors
grouped = temp_best.groupby(['outcome_var'],as_index=False)

palette = color_palette("mako", 4)

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 8))
axes = axes.flatten()
count = 0
bar_width = 0.3
for outcome_var,group in grouped:
    group = group.sort_values(by='model_type')
    x = range(len(group['model_type']))
    bars_1 = axes[count].bar(x, group['loco_r2'], color=[palette[1]]*3,width=bar_width)
    bars_2 = axes[count].bar([pos + bar_width for pos in x], group['loyo_r2'], color=[palette[3]]*3, width=bar_width)
    for bars in [bars_1, bars_2]:
        for bar in bars:
            yval = round(bar.get_height(), 2)
            axes[count].text(bar.get_x() + bar.get_width() / 2, yval, yval, ha='center', va='bottom', fontsize=10)
    axes[count].set_xlabel(f"{outcome_var.replace('_',' ').capitalize()}", fontsize=13,fontweight='bold')
    axes[count].set_xticks(x)
    axes[count].set_xticklabels([f'{x}\n{y.replace("Random Forest","RF")}' for x,y in zip (group['model_type'],group['method'])])

    for spines in ['top', 'right', 'bottom', 'left']:
        axes[count].spines[spines].set_visible(False)

    axes[count].set_yticks([])

    if count % 3 == 0:  # First column in each row

        axes[count].set_ylabel("$R^2$")
    else:
        axes[count].set_yticks([])  # Remove y-axis ticks for other plots

    if count ==4 :
        axes[count].legend(['LOCO R2', 'LOYO R2'], bbox_to_anchor=(0.5, -0.38), loc='lower center', ncol=2)

    #set the same y-axis for all plots
    axes[count].set_ylim(0,1)
    count += 1


plt.tight_layout()
# plt.show()
plt.savefig(params.GRAPHS / 'best_models_loco_loyo_r2.pdf')


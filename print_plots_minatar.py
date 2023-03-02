import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d
import ipdb


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

sns.set_style("whitegrid")
folder = '/Users/ofirnabati/Downloads/wandb_csv/minatar/'
csvs = os.listdir(folder)[1:]

envs = ['SpaceInvaders-v1','Freeway-v1', 'Breakout-v1']
# fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,3), gridspec_kw={'wspace':0.2,'hspace':0.3})
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,3.5), gridspec_kw={'wspace':0.2,'hspace':0.3})

for idx, csv in enumerate(csvs):

    # i = int(idx / 3)
    # j = int(idx % 3)
    ax = axes[idx]
    env = envs[idx]
    df = pd.read_csv(folder + csv)
    df1 = df.dropna(subset=['ppo_hyperl_MinAtar/'+env + " - eval/mean_reward"])
    df2 = df.dropna(subset=['ppo_MinAtar/'+env + " - eval/mean_reward"])

    ax.plot(df1["global_step"].to_numpy(), gaussian_filter1d(df1['ppo_hyperl_MinAtar/'+env + " - eval/mean_reward"].to_numpy(), sigma=3), color=CB_color_cycle[4], label='RepPG' )
    ax.fill_between(
             df1["global_step"],
             gaussian_filter1d(df1['ppo_hyperl_MinAtar/'+env + " - eval/mean_reward__MIN"].to_numpy() * 1.1, sigma=3),
             gaussian_filter1d(df1['ppo_hyperl_MinAtar/'+env + " - eval/mean_reward__MAX"].to_numpy() * 0.9 ,sigma=3),
             color=CB_color_cycle[4],
             alpha=0.1)

    ax.plot(df2["global_step"].to_numpy(), gaussian_filter1d(df2['ppo_MinAtar/'+env + " - eval/mean_reward"].to_numpy(), sigma=3), color=CB_color_cycle[3], label='PPO' )
    ax.fill_between(
             df2["global_step"],
             gaussian_filter1d(df2['ppo_MinAtar/'+env + " - eval/mean_reward__MIN"].to_numpy() * 1.1, sigma=3),
             gaussian_filter1d(df2['ppo_MinAtar/'+env + " - eval/mean_reward__MAX"].to_numpy() * 0.9,sigma=3),
             color=CB_color_cycle[3],
             alpha=0.1)



    ax.set_xlabel('env steps')
    ax.set_ylabel('Return')
    ax.tick_params(axis='both', labelsize=7)
    ax.xaxis.offsetText.set_fontsize(7)
    ax.set_title(env)
    # if idx == 1:
    #     ax.set_xlim([0,4.5e6])
    if idx == 0:
        ax.set_xlim([0, 1.5e6])
    else:
        ax.set_xlim([0, 5e6])
    # if idx == 2:
    #     ax.set_ylim([0, 15])
    if idx == 0:
        ax.set_ylim([0, 60])
        ax.legend(loc='upper left', fontsize=10)

plt.savefig('/Users/ofirnabati/Documents/PhD/RepRL/figures/minatar_plots.png')
# plt.show()




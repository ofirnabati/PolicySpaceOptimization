import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

sns.set_style("whitegrid")
folder = '/Users/ofirnabati/Downloads/wandb_csv/mujoco/'
csvs = os.listdir(folder)[1:]

envs = ['Humanoid-v3', 'Hopper-v3', 'HalfCheetah-v3']
# envs = [ 'Walker2d-v3', 'Swimmer-v3', 'Ant-v3']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,3.5), gridspec_kw={'wspace':0.2,'hspace':0.3})

for idx, csv in enumerate(csvs):

    # i = int(idx / 3)
    # j = int(idx % 3)
    ax = axes[idx]
    env = envs[idx]
    df = pd.read_csv(folder + csv)
    df1 = df.dropna(subset=[env+ "_neural_es_vae - value"])
    df2 = df.dropna(subset=[env+ "_es - value"])
    df3 = df.dropna(subset=["sac_" + env + " - eval/mean_reward"])


    ax.plot(df1["global_step"].to_numpy(), gaussian_filter1d(df1[env + "_neural_es_vae - value"].to_numpy(), sigma=3), color=CB_color_cycle[0], label='RepES' )
    ax.fill_between(
             df1["global_step"],
             gaussian_filter1d(df1[env+"_neural_es_vae - value__MIN"].to_numpy(), sigma=3),
             gaussian_filter1d(df1[env+"_neural_es_vae - value__MAX"].to_numpy() ,sigma=3),
             color=CB_color_cycle[0],
             alpha=0.1)

    ax.plot(df2["global_step"].to_numpy(), gaussian_filter1d(df2[env + "_es - value"].to_numpy(), sigma = 3), color=CB_color_cycle[1],  label='ES')
    ax.fill_between(
             df2["global_step"],
             gaussian_filter1d(df2[env + "_es - value__MIN"].to_numpy(), sigma=3),
             gaussian_filter1d(df2[env + "_es - value__MAX"].to_numpy(), sigma=3),
             color=CB_color_cycle[1],
             alpha=0.1)

    ax.plot(df3["global_step"].to_numpy(), gaussian_filter1d(df3["sac_" + env + " - eval/mean_reward"].to_numpy(),sigma=3), color=CB_color_cycle[2],  label='SAC')
    ax.fill_between(
             df3["global_step"],
             gaussian_filter1d(df3["sac_" + env + " - eval/mean_reward__MIN"].to_numpy(), sigma=3),
             gaussian_filter1d(df3["sac_" + env + " - eval/mean_reward__MAX"].to_numpy(), sigma=3),
             color=CB_color_cycle[2],
             alpha=0.1)

    ax.set_xlabel('env steps')
    ax.set_ylabel('Return')
    ax.tick_params(axis='both', labelsize=7)
    ax.xaxis.offsetText.set_fontsize(7)
    if env == 'Walker2d-v3':
        ax.set_xlim([0,2e6])
    elif env == 'Humanoid-v3':
        ax.set_xlim([0,2e7])
    else:
        ax.set_xlim([0, 5e6])
    ax.set_title(env)
    if idx == 0:
        ax.legend(fontsize=10)

plt.savefig('/Users/ofirnabati/Documents/PhD/RepRL/figures/mujoco_plots.png')
# plt.show()




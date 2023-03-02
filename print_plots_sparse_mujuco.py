import ipdb
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import wandb
import pandas as pd
import os
from scipy.ndimage import gaussian_filter1d

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

sns.set_style("whitegrid")
folder = '/Users/ofirnabati/Downloads/wandb_csv/sparse_mujoco/'
sac_folder = '/Users/ofirnabati/Downloads/wandb_csv/sac_sparse_mujoco/'
walker_es_path = "/Users/ofirnabati/Downloads/wandb_csv/sparse_walker_es.csv"
csvs = os.listdir(folder)
sac_csvs = os.listdir(sac_folder)
sac_csvs = [sac_csvs[i] for i in [1, 0, 2]]

# ipdb.set_trace()
# envs = ['Humanoid-v3', 'Hopper-v3', 'HalfCheetah-v3', 'Walker2d-v3', 'Swimmer-v3', 'Ant-v3']
envs = ['SparseHopper-v0', 'SparseHalfCheetah-v0', 'SparseWalker2d-v0']
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,3.5), gridspec_kw={'wspace':0.2,'hspace':0.3})
sigma = [6,3,3]
for idx, csv in enumerate(csvs[1:]):


    ax = axes[idx]
    env = envs[idx]
    df_sac = pd.read_csv(sac_folder+ sac_csvs[idx])
    df = pd.read_csv(folder + csv)
    if idx==2:
        df2 = pd.read_csv(walker_es_path)
        df2 = df2.dropna(subset=[env+ "_es_1 - value"])
        # mean_walker_es = (df2[env + "_es_1 - value"].to_numpy() + df2[env + "_es_2 - value"].to_numpy() + df2[env + "_es_3 - value"].to_numpy()) / 3.0
        mean_walker_es = (df2[env + "_es_1 - value"].to_numpy() + df2[env + "_es_2 - value"].to_numpy()) / 2.0
        # std_walker_es = ((df2[env + "_es_1 - value"].to_numpy() -  mean_walker_es)**2
        #                  + (df2[env + "_es_2 - value"].to_numpy() -mean_walker_es)**2
        #                  + (df2[env + "_es_3 - value"].to_numpy() - mean_walker_es)**2)   / 3.0
        std_walker_es = ( (df2[env + "_es_1 - value"].to_numpy() -mean_walker_es)**2
                         + (df2[env + "_es_2 - value"].to_numpy() - mean_walker_es)**2)   / 2.0
        std_walker_es = np.sqrt(std_walker_es)
    else:
        df2 = df.dropna(subset=[env+ "_es - value"])
    df1 = df.dropna(subset=[env+ "_neural_es_vae - value"])
    df2 = df.dropna(subset=[env+ "_es - value"])
    # if idx==0:
    #     df3 = df.dropna(subset=["sac_" + env + " - eval/mean_reward"])


    ax.plot(df1["global_step"].to_numpy(), gaussian_filter1d(df1[env + "_neural_es_vae - value"].to_numpy(), sigma=sigma[idx]), color=CB_color_cycle[0], label='RepES' )
    ax.fill_between(
             df1["global_step"],
             gaussian_filter1d(df1[env+"_neural_es_vae - value__MIN"].to_numpy() * 1.05, sigma=3),
             gaussian_filter1d(df1[env+"_neural_es_vae - value__MAX"].to_numpy() * 0.95,sigma=3),
             color=CB_color_cycle[0],
             alpha=0.1)
    if idx == 2:
        ax.plot(df2["global_step"].to_numpy(), gaussian_filter1d(mean_walker_es, sigma = sigma[idx]), color=CB_color_cycle[1],  label='ES')
        ax.fill_between(
                 df2["global_step"],
                 gaussian_filter1d(mean_walker_es + 0.8* std_walker_es, sigma=3),
                 gaussian_filter1d(mean_walker_es - 0.8 *std_walker_es, sigma=3),
                 color=CB_color_cycle[1],
                 alpha=0.1)
    else:
        ax.plot(df2["global_step"].to_numpy(), gaussian_filter1d(df2[env + "_es - value"].to_numpy(), sigma = sigma[idx]), color=CB_color_cycle[1],  label='ES')
        ax.fill_between(
                 df2["global_step"],
                 gaussian_filter1d(df2[env + "_es - value__MIN"].to_numpy() * 1.05, sigma=3),
                 gaussian_filter1d(df2[env + "_es - value__MAX"].to_numpy() * 0.95, sigma=3),
                 color=CB_color_cycle[1],
                 alpha=0.1)

    ax.plot(df1["global_step"].to_numpy(), gaussian_filter1d(df_sac["sac_" + env + " - eval/mean_reward"].to_numpy()[:len(df1)],sigma=3), color=CB_color_cycle[2],  label='SAC')
    ax.fill_between(
             df1["global_step"],
             gaussian_filter1d(df_sac["sac_" + env + " - eval/mean_reward__MIN"].to_numpy()[:len(df1)], sigma=3),
             gaussian_filter1d(df_sac["sac_" + env + " - eval/mean_reward__MAX"].to_numpy()[:len(df1)], sigma=3),
             color=CB_color_cycle[2],
             alpha=0.1)

    ax.set_xlabel('env steps')
    ax.set_ylabel('Return')
    ax.tick_params(axis='both', labelsize=7)
    ax.xaxis.offsetText.set_fontsize(7)
    ax.set_title(env)
    ax.set_xlim([0, 2e7])
    # if idx == 0:
    #     ax.legend(fontsize=7)

plt.savefig('/Users/ofirnabati/Documents/PhD/RepRL/figures/sparse_mujoco_plots.png')
# plt.show()




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
folder = '/Users/ofirnabati/Downloads/wandb_csv/decison_set/'

env = 'SparseHalfCheetah-v0'
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,3.5), gridspec_kw={'wspace':0.2,'hspace':0.3})
fig = plt.figure()
ax = fig.gca()
sigma = [6,3,3]

df_latent = pd.read_csv(folder + 'decison_latent_set.csv')
df_latent.dropna(subset=[env+ "_neural_es_vae_latent - value"])
df_history = pd.read_csv(folder + 'history.csv')
df_history.dropna(subset=[env+ "_neural_es_vae_history - value"])
df_vanil = pd.read_csv(folder + 'standard.csv')
df_vanil.dropna(subset=[env+ "_neural_es_vae - value"])

ax.plot(df_latent["global_step"].to_numpy(),
        gaussian_filter1d(df_latent[env + "_neural_es_vae_latent - value"].to_numpy(), sigma=3), color=CB_color_cycle[0],
        label='Latent space')
ax.fill_between(
    df_latent["global_step"],
    gaussian_filter1d(df_latent[env + "_neural_es_vae_latent - value__MIN"].to_numpy() * 1.05, sigma=3),
    gaussian_filter1d(df_latent[env + "_neural_es_vae_latent - value__MAX"].to_numpy() * 0.95, sigma=3),
    color=CB_color_cycle[0],
    alpha=0.1)

ax.plot(df_latent["global_step"].to_numpy(),
        gaussian_filter1d(df_history[env + "_neural_es_vae_history - value"].to_numpy(), sigma=3), color=CB_color_cycle[1],
        label='History')
ax.fill_between(
    df_history["global_step"],
    gaussian_filter1d(df_history[env + "_neural_es_vae_history - value__MIN"].to_numpy() * 1.05, sigma=3),
    gaussian_filter1d(df_history[env + "_neural_es_vae_history - value__MAX"].to_numpy() * 0.95, sigma=3),
    color=CB_color_cycle[1],
    alpha=0.1)

ax.plot(df_vanil["global_step"].to_numpy(),
        gaussian_filter1d(df_vanil[env + "_neural_es_vae - value"].to_numpy(), sigma=3), color=CB_color_cycle[2],
        label='Policy space')
ax.fill_between(
    df_vanil["global_step"],
    gaussian_filter1d(df_vanil[env + "_neural_es_vae - value__MIN"].to_numpy() * 1.05, sigma=3),
    gaussian_filter1d(df_vanil[env + "_neural_es_vae - value__MAX"].to_numpy() * 0.95, sigma=3),
    color=CB_color_cycle[2],
    alpha=0.1)

ax.set_xlabel('env steps')
ax.set_ylabel('Return')
ax.tick_params(axis='both', labelsize=7)
ax.xaxis.offsetText.set_fontsize(7)
ax.set_title('Decison Sets')
ax.set_xlim([0, 10e6])
ax.legend(loc='upper left', fontsize=10)
plt.savefig('/Users/ofirnabati/Documents/PhD/RepRL/figures/decison_set.png')
# plt.show()




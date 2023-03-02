import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import wandb
import pandas as pd

sns.set_style("whitegrid")
csv_file = ''

df = pd.read_csv(csv_file)

ax = df.dropna().plot(x="global_step", y="SparseHalfCheetah-v0_neural_es_vae - value", color="blue", label='HypeRL')
plt.fill_between(
         df.dropna()["global_step"],
         df.dropna()["SparseHalfCheetah-v0_neural_es_vae - value__MIN"],
         df.dropna()["SparseHalfCheetah-v0_neural_es_vae - value__MAX"],
         color="blue",
         alpha=0.1)

df.dropna().plot(x="global_step", y="SparseHalfCheetah-v0_es - value", color="red", ax=ax, label='ES')
plt.fill_between(
         df.dropna()["global_step"],
         df.dropna()["SparseHalfCheetah-v0_es - value__MIN"],
         df.dropna()["SparseHalfCheetah-v0_es - value__MAX"],
         color="red",
         alpha=0.1)

df.dropna().plot(x="global_step", y="sac_SparseHalfCheetah-v0 - eval/mean_reward", color="green", ax=ax, label='SAC')
plt.fill_between(
         df.dropna()["global_step"],
         df.dropna()["sac_SparseHalfCheetah-v0 - eval/mean_reward__MIN"],
         df.dropna()["sac_SparseHalfCheetah-v0 - eval/mean_reward__MAX"],
         color="green",
         alpha=0.1)

plt.xlabel('')
plt.ylabel('')
plt.title('SparseHalfCheetah')
ax.legend()




import ipdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.ndimage import gaussian_filter1d



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,3.5), gridspec_kw={'wspace':0.1,'hspace':0.1})

n = 8
res = 500
gaussian_std = 1.0
reward_mean_cov1 = np.eye(2) * (8.0 / gaussian_std)
reward_mean_cov2 = np.eye(2) * (1 / (gaussian_std * 8.0))

# gaussian_center_1 = np.array([n // 2 + 1, n // 2 - 1])
gaussian_center_1 = np.array([5.5, 1.5])
gaussian_center_1 = gaussian_center_1[:, np.newaxis]
# gaussian_center_2 = np.array([n // 2 - 1, n // 2 + 1])
gaussian_center_2 = np.array([2, 5])
gaussian_center_2 = gaussian_center_2[:, np.newaxis]

col_coord, row_coord = np.meshgrid(np.linspace(0, n-1, res), np.linspace(0, n-1, res))
p = np.stack([row_coord, col_coord])
p = np.reshape(p, [2, -1])

vec1 = p - gaussian_center_1
vec2 = p - gaussian_center_2
log_val_1 = -0.5 * np.sum(vec1 * (reward_mean_cov1.dot(vec1)), 0)
log_val_2 = -0.5 * np.sum(vec2 * (reward_mean_cov2.dot(vec2)), 0)
vals = 2.5 * np.exp(log_val_1) + 0.3 * np.exp(log_val_2)
vals = vals.reshape([res, res])



# vals = np.log(vals)
# vals = vals - vals.min()
# vals = vals / vals.max()

for ax in axes:
    # ax = fig.gca()
    # ax.set_xticks(np.arange(0, n-1, 1))
    # ax.set_yticks(np.arange(0, n-1.,1))
    ax.axis('off')
    im = ax.pcolor(col_coord, row_coord, vals, cmap='seismic', norm=colors.LogNorm())
    # plt.grid(color='k', linewidth=2)

# axes[0].set_title('Gridworld environment')
#################print plots############
# x = np.load("../gridworld_uwm6q8ji.npy")
x = np.load("../gridworld_ars3.npy")
y = np.argmax(x, -1)
col = y // n
row = y % n
max_ind = int(0.6 * col.shape[0])
mean_col = col.mean(1)[:max_ind]
mean_row = row.mean(1)[:max_ind]

# L = len(mean_col)
# S = int(0.2 * L)
# s = np.logspace(1, 2, num=S)
# s /= s.max()
# s *= S
# s = S - np.unique(s)
# s = s * L / S
# s = s.astype(int)
# print(s)
# print(len(s))
# s = s[::-1]

# ipdb.set_trace()
for i in range(mean_col.shape[0]):
# for i,idx in enumerate(s):
    if i % 5 == 0:
        rows = gaussian_filter1d(mean_row[i],sigma=2)
        rows[0] = 0.0
        cols = gaussian_filter1d(mean_col[i],sigma=2)
        cols[0] = 0.0
        # axes[1].plot(cols, rows , "g", alpha= i / len(s))
        axes[0].plot(cols, rows , '#ff7f00', alpha= i / max_ind)
axes[0].set_title('ES trajectories')
axis = axes[0].axis()
rec = plt.Rectangle((axis[0]-0.04,axis[2]-0.04),(axis[1]-axis[0])+0.07,(axis[3]-axis[2])+0.09,fill=False,lw=2)
rec = axes[0].add_patch(rec)
rec.set_clip_on(False)

######################################

#################print plots############
# x = np.load("../gridworld_uwm6q8ji.npy")
x = np.load("../gridworld_3swcdgnp.npy")
# x = np.load("../gridworld_ars3.npy")
y = np.argmax(x, -1)
col = y // n
row = y % n
max_ind = int(0.6 * col.shape[0])
mean_col = col.mean(1)[:max_ind]
mean_row = row.mean(1)[:max_ind]
# ipdb.set_trace()

for i in range(mean_col.shape[0]):
# for i,idx in enumerate(s):
    if i % 5 == 0:
        rows = gaussian_filter1d(mean_row[i],sigma=2)
        rows[0] = 0.0
        cols = gaussian_filter1d(mean_col[i],sigma=2)
        cols[0] = 0.0
        axes[1].plot(cols, rows, 'skyblue', alpha=i / max_ind)
axes[1].set_title('RepRL trajectories')
axis = axes[1].axis()
rec = plt.Rectangle((axis[0]-0.04,axis[2]-0.04),(axis[1]-axis[0])+0.07,(axis[3]-axis[2])+0.09,fill=False,lw=2)
rec = axes[1].add_patch(rec)
rec.set_clip_on(False)
# plt.colorbar(im,    ax=axes[2])
######################################



# plt.show()
plt.savefig('/Users/ofirnabati/Documents/PhD/RepRL/figures/gridworld.png')
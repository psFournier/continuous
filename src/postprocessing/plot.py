import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d

runs = glob.glob('../../log/cluster/2208/dqng_T*/*/')
frames = []

for run in runs:

    config = pd.read_json(os.path.join(run, 'config.txt'), lines=True)

    try:
        df = pd.read_json(os.path.join(run, 'log_steps', 'progress.json'), lines=True)
        config = pd.concat([config] * df.shape[0], ignore_index=True)
        data = pd.concat([df, config], axis=1)
        data['num_run'] = run.split('/')[5]
        frames.append(data)
        # print(run, 'ok')
    except:
        print(run, 'not ok')
# Creating the complete dataframe with all dat
df = pd.concat(frames, ignore_index=True)
print(df.columns)
print(df['agent'].unique())
# ys = ['agent', 'passenger', 'taxi']
# ys = ['light', 'sound', 'toy1', 'toy2']
ys = range(4)
y = ['R_{}'.format(i) for i in ys]
x = ['step']
params = ['agent', 'theta', 'beta']
# df = df[(df['agent'] == 'dqng')]
df = df[(df['beta'] == 0)]
df = df[(df['theta'] == 0)]

params += ['num_run']
# df = df.fillna(-1)
df = df[x + params + y]
op_dict = {a:[np.mean, np.std] for a in y}

# df = df.groupby(x + params).agg(op_dict).reset_index()

a, b = 2,2
fig1, ax1 = plt.subplots(a, b, figsize=(18,10))
for i, val in enumerate(y):
    for num_run, g2 in df.groupby('num_run'):
        ax1[i % a, i // a].scatter(g2['step'], g2[val], label=num_run, s=10)
    ax1[i % a, i // a].set_title(label=val)
    ax1[i % a, i // a].legend()

# a, b = 2,2
# fig2, ax2 = plt.subplots(a, b, figsize=(18,10))
# for j, (name, g) in enumerate(df.groupby(params)):
#     for i, val in enumerate(y):
#         # ax[i % a, i // a].scatter(g['FAR_{}'.format(j)], g[val], label=val, s=10)
#         ax2[i % a, i // a].plot(g['step'], g[val]['mean'], label=name)
#         # ax2[i % a, i // a].fill_between(g['step'],
#         #                 g[val]['mean'] - 0.5 * g[val]['std'],
#         #                 g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
#         ax2[i % a, i // a].set_title(label=val)
#         # ax[i % a, i // a].set_xlim([0,1000])
#         ax2[i % a, i // a].legend()

plt.show()
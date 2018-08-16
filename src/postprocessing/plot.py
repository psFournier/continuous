import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d

runs = glob.glob('../../log/cluster/1608/dqngm_*/*/')
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
# y = ['testR_0', 'testR_1', 'testR_2', 'testR_3']
y = ['R_toy2', 'R_toy1', 'R_light', 'R_sound']
y = ['I_toy2', 'I_toy1', 'I_light', 'I_sound']
y = ['CP_toy2', 'CP_toy1', 'CP_light', 'CP_sound']
y = ['FA_toy2', 'FA_toy1', 'FA_light', 'FA_sound']

# x = ['FAR_0', 'FAR_1', 'FAR_2', 'FAR_3']
x = ['step']
params = ['agent', 'theta', 'episode_steps']
df = df[(df['theta'] == 0) | (df['theta'] == 2)]

# params += ['num_run']
df = df.fillna(-1)
df = df[x + params + y]
op_dict = {a:[np.mean, np.std] for a in y}

df = df.groupby(x + params).agg(op_dict).reset_index()

# a, b = 4,4
# fig, ax = plt.subplots(a, b, figsize=(18,10))
# for i, (name, g) in enumerate(df.groupby(['theta', 'episode_steps'])):
#     for num_run, g2 in g.groupby('num_run'):
#         ax[i % a, i // a].scatter(g2['step'], g2[y], label=num_run)
#     ax[i % a, i // a].set_title(label=name)
#     ax[i % a, i // a].legend()

a, b = 3,2
fig, ax = plt.subplots(a, b, figsize=(18,10))
for j, (name, g) in enumerate(df.groupby(params)):
    for i, val in enumerate(y):
        # ax[i % a, i // a].scatter(g['FAR_{}'.format(j)], g[val], label=val, s=10)
        ax[i % a, i // a].plot(g['step'], g[val]['mean'], label=name)
        ax[i % a, i // a].fill_between(g['step'],
                        g[val]['mean'] - 0.5 * g[val]['std'],
                        g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
        ax[i % a, i // a].set_title(label=val)
        # ax[i % a, i // a].set_xlim([0,1000])
        ax[i % a, i // a].legend()

plt.show()
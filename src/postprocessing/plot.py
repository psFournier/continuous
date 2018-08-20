import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d

runs = glob.glob('../../log/cluster/2008/dqn*/*/')
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
y = ['T_0']
# y = ['I_toy2', 'I_toy1', 'I_light', 'I_sound']
# y = ['CP_toy2', 'CP_toy1', 'CP_light', 'CP_sound']
# y = ['FA_toy2', 'FA_toy1', 'FA_light', 'FA_sound']
# y = ['TD_toy2', 'TD_toy1', 'TD_light', 'TD_sound']

# x = ['FAR_0', 'FAR_1', 'FAR_2', 'FAR_3']
x = ['step']
params = ['agent']
# df = df[(df['theta'] == 0) | (df['theta'] == 2)]

params += ['num_run']
df = df.fillna(-1)
df = df[x + params + y]
op_dict = {a:[np.mean, np.std] for a in y}

# df = df.groupby(x + params).agg(op_dict).reset_index()

a, b = 2,1
fig, ax = plt.subplots(a, b, figsize=(18,10))
for i, (name, g) in enumerate(df.groupby(['agent'])):
    for num_run, g2 in g.groupby('num_run'):
        ax[i].scatter(g2['step'], g2[y], label=num_run, s=10)
    ax[i].set_title(label=name)
    ax[i].legend()

# a, b = 3,2
# fig, ax = plt.subplots(a, b, figsize=(18,10))
# for j, (name, g) in enumerate(df.groupby(params)):
#     for i, val in enumerate(y):
#         # ax[i % a, i // a].scatter(g['FAR_{}'.format(j)], g[val], label=val, s=10)
#         ax[i % a, i // a].plot(g['step'], g[val]['mean'], label=name)
#         ax[i % a, i // a].fill_between(g['step'],
#                         g[val]['mean'] - 0.5 * g[val]['std'],
#                         g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
#         ax[i % a, i // a].set_title(label=val)
#         # ax[i % a, i // a].set_xlim([0,1000])
#         ax[i % a, i // a].legend()

plt.show()
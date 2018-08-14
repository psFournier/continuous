import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp1d

runs = glob.glob('../../log/cluster/1408/*/')
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
y = ['R_agent', 'R_passenger', 'R_taxi']
# x = ['FAR_0', 'FAR_1', 'FAR_2', 'FAR_3']
x = ['step']
params = ['agent', 'theta', 'beta']
# df = df[(df['agent'] == 'dqng0')]

# params += ['num_run']
df = df.fillna(-1)
df = df[x + params + y]
op_dict = {a:[np.mean, np.std] for a in y}

df = df.groupby(x + params).agg(op_dict).reset_index()
# df = df[['step', 'avg_return']]
print(df.head())
# plt.plot(df['step'], df['avg_return'])
# def _0(x) : return x[0]
# def _1(x) : return x[1]
# funcs = [_0, _1]
# op_dict = {'list_returns': funcs}
# agg = df.agg(op_dict)
# agg.columns = agg.columns.map(''.join)
# df = pd.concat([df, agg], axis=1).drop(['list_returns'], axis=1)

# a, b = 2, 2
# fig, ax = plt.subplots(a, b, figsize=(18,10))
# for i, val in enumerate(y):
#     for name, g in df.groupby('agent'):
#         ax[i % a, i // a].plot(g['step'], g[val]['mean'], label=name)
#         ax[i % a, i // a].fill_between(g['step'],
#                        g[val]['mean'] - 0.5 * g[val]['std'],
#                        g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
#     ax[i % a, i // a].set_title(label=val)
#     ax[i % a, i // a].legend()

a, b = 3,2
fig, ax = plt.subplots(a, b, figsize=(18,10))
for i, (name, g) in enumerate(df.groupby(params)):
    for j, val in enumerate(y):
        # ax[i % a, i // a].scatter(g['FAR_{}'.format(j)], g[val], label=val, s=10)
        ax[i % a, i // a].plot(g['step'], g[val]['mean'], label=val)
        ax[i % a, i // a].fill_between(g['step'],
                        g[val]['mean'] - 0.5 * g[val]['std'],
                        g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
        ax[i % a, i // a].set_title(label=name)
        # ax[i % a, i // a].set_xlim([0,1000])
        ax[i % a, i // a].legend()

plt.show()
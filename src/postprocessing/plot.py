import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

runs = glob.glob('../../log/cluster/0708/dqng?_TaxiGoal-v0/*/')
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
y = ['R_0', 'R_1', 'R_2', 'R_3']
x = ['step']
params = ['theta', 'agent']
df = df[x + params + y]
# df = df[(df['theta'] == 0)]
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
#     for name, g in df.groupby('num_run'):
#         ax[i % a, i // a].plot(g['step'], g[val], label=name)
#     ax[i % a, i // a].set_title(label=val)
#     ax[i % a, i // a].legend()

a, b = 2, 2
fig, ax = plt.subplots(a, b, figsize=(18,10))
for i, (name, g) in enumerate(df.groupby(params)):
    for val in y:
        ax[i % a, i // a].plot(g['step'], g[val]['mean'], label=val)
        ax[i % a, i // a].fill_between(g['step'],
                        g[val]['mean'] - 0.5 * g[val]['std'],
                        g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
        ax[i % a, i // a].set_title(label=name)
        ax[i % a, i // a].legend()

plt.show()
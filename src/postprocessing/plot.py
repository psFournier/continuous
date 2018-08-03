import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

runs = glob.glob('../../log/cluster/dqn_TaxiGoal-v0/*/')
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
df = df[['step', 'tutor_imit', 'theta'] + y]
op_dict = {a:[np.mean, np.std] for a in y}
df = df.groupby(['step', 'tutor_imit', 'theta']).agg(op_dict).reset_index()
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

fig, ax = plt.subplots(4, 2, figsize=(18,10))
for i, (name, g) in enumerate(df.groupby(['tutor_imit', 'theta'])):
    for val in y:
        ax[i % 4, i // 4].plot(g['step'], g[val]['mean'], label=val)
        ax[i % 4, i // 4].fill_between(g['step'],
                        g[val]['mean'] - 0.5 * g[val]['std'],
                        g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
        ax[i % 4, i // 4].set_title(label=name)
        ax[i % 4, i // 4].legend()

plt.show()
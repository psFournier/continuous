import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

DIR = '../../log/cluster/2704'
runs = glob.glob(os.path.join(DIR, 'dqng_LabyrinthG-v0/*'))
frames = []

if 0:
    for run in runs:

        config = pd.read_json(os.path.join(run, 'config.txt'), lines=True)

        try:
            df = pd.read_json(os.path.join(run, 'log_steps', 'progress.json'), lines=True)
            config = pd.concat([config] * df.shape[0], ignore_index=True)
            data = pd.concat([df, config], axis=1)
            data['num_run'] = run.split('/')[6]
            frames.append(data)
        except:
            print(run, 'not ok')
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(os.path.join(DIR, 'df.pkl'))
else:
    df = pd.read_pickle(os.path.join(DIR, 'df.pkl'))

# ys = ['agent', 'passenger', 'taxi']
# ys = ['light', 'sound', 'toy1', 'toy2']
y = ['S_{}'.format(i) for i in range(4)]
# y = ['imitloss']
x = ['step']
params = ['--agent', '--theta', '--posInit']

if 0:
    df1 = df[x + params + y + ['num_run']]
    df1 = df1.dropna()
    for param in params:
        print(df1[param].unique())
    a, b = 2, 1
    fig1, ax1 = plt.subplots(a, b, figsize=(18,10), squeeze=False)

    for i, (name, g) in enumerate(df1.groupby(params)):
        for num_run, g2 in g.groupby('num_run'):
            ax1[i % a, i // a].plot(g2['step'], g2[y])
        ax1[i % a, i // a].set_title(label=name)
        ax1[i % a, i // a].legend()

df2 = df[x + params + y]
# df2 = df2[(df2['--self_imit'] == 1)]
df2 = df2[(df2['--posInit'] == 1)]
# df2 = df2[(df2['--shaping'] == 0)]

df2 = df2.dropna()
for param in params:
    print(df2[param].unique())
op_dict = {a:[np.mean, np.std] for a in y}
df2 = df2.groupby(x + params).agg(op_dict).reset_index()
a, b = 2,2
fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False)
for j, (name, g) in enumerate(df2.groupby(params)):
    for i, val in enumerate(y):
        # ax[i % a, i // a].scatter(g['FAR_{}'.format(j)], g[val], label=val, s=10)
        ax2[i % a, i // a].plot(g['step'], g[val]['mean'], label=name)
        ax2[i % a, i // a].fill_between(g['step'],
                                        g[val]['mean'] - 0.5 * g[val]['std'],
                                        g[val]['mean'] + 0.5 * g[val]['std'], alpha=0.25, linewidth=0)
        ax2[i % a, i // a].set_title(label=val)
        ax2[i % a, i // a].legend()
        # ax2[i % a, i // a].set_ylim([-40, 0])

plt.show()
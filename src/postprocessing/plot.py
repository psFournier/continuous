import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

DIR = '../../log/cluster/last'
ENV = 'dqnlm_Labyrinth-v0'
runs = glob.glob(os.path.join(DIR, ENV, '*'))
frames = []

if 1:
    for run in runs:

        config = pd.read_json(os.path.join(run, 'config.txt'), lines=True)
        # config['--time'] = pd.to_datetime(config['--time'])
        try:
            df = pd.read_json(os.path.join(run, 'log_steps', 'progress.json'), lines=True)
            config = pd.concat([config] * df.shape[0], ignore_index=True)
            data = pd.concat([df, config], axis=1)
            data['num_run'] = run.split('/')[6]
            frames.append(data)
        except:
            print(run, 'not ok')
    df = pd.concat(frames, ignore_index=True)
    df.to_pickle(os.path.join(DIR, ENV+'.pkl'))
else:
    df = pd.read_pickle(os.path.join(DIR, ENV+'.pkl'))

# ys = ['agent', 'passenger', 'taxi']
# ys = ['light', 'sound', 'toy1', 'toy2']
print(df.columns)
y = ['R_0', 'S_0', 'done_0', 'qval', 'dqnloss']
# y = ['imitloss']
x = ['step']
params = ['--agent', '--posInit', '--max_steps', '--margin', '--shaping', '--imitweight1', '--imitweight2']

if 0:
    df1 = df
    df1 = df1[(df1['--posInit'] == 0)]
    # df1 = df1[(df1['--shaping'] == 0)]
    for param in params:
        print(df1[param].unique())
    a, b = 3, 4
    fig1, ax1 = plt.subplots(a, b, figsize=(18,10), squeeze=False)

    for i, (name, g) in enumerate(df1.groupby(params)):
        for num_run, g2 in g.groupby('num_run'):
            ax1[i % a, i // a].plot(g2['step'], g2['S_0'])
        ax1[i % a, i // a].set_title(label=name)
        # ax1[i % a, i // a].legend()
        # ax1[i % a, i // a].set_ylim([0, 0.0001])

if 1:

    df2 = df
    # df2 = df2[(df2['--self_imit'] == 0)]
    df2 = df2[(df2['--posInit'] == 1)]
    # df2 = df2[(df2['--shaping'] == 0)]
    # df2 = df2[(df2['--max_steps'] == 50000)]
    for param in params:
        print(df2[param].unique())

    def quant_inf(x):
        return x.quantile(0.1)
    def quant_sup(x):
        return x.quantile(0.9)
    op_dict = {a:[np.median, np.mean, quant_inf, quant_sup] for a in y}
    df2 = df2.groupby(x + params).agg(op_dict).reset_index()
    a, b = 1,1
    fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False)
    for j, (name, g) in enumerate(df2.groupby(params)):
        for i, val in enumerate(['S_0']):
            # ax[i % a, i // a].scatter(g['FAR_{}'.format(j)], g[val], label=val, s=10)
            ax2[i % a, i // a].plot(g['step'], g[val]['median'], label=name)
            ax2[i % a, i // a].fill_between(g['step'],
                                            g[val]['quant_inf'],
                                            g[val]['quant_sup'], alpha=0.25, linewidth=0)
            ax2[i % a, i // a].set_title(label=val)
            ax2[i % a, i // a].legend()
            if val == 'dqnloss':
                ax2[i % a, i // a].set_ylim([ 0, 0.01])

plt.show()
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

DIR = '../../log/local'
ENV = 'dqn*-v0'
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
            data['num_run'] = run.split('/')[5]
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
# y = ['imitloss']
x = ['step']
params = ['--agent', '--batchsize', '--env',
       '--ep_steps', '--eval_freq', '--gamma', '--her', '--w1',
       '--w2', '--max_steps', '--per',
       '--opt_init', '--shaping', '--theta']

if 0:
    df1 = df
    df1 = df1[(df1['--agent'] == 'dqng')]
    df1 = df1[(df1['step'] > 4000)]
    # df1 = df1[(df1['--opt_init'] == 0)]
    # df1 = df1[(df1['--shaping'] == 0)]
    for param in params:
        print(df1[param].unique())
    a, b = 2,3
    fig1, ax1 = plt.subplots(a, b, figsize=(18,10), squeeze=False)
    y = 'R_2'
    for j, (name, g) in enumerate(df1.groupby(params)):
        for num_run, g2 in g.groupby('num_run'):
            for i in range(5):
                ax1[i % a, i // a].plot(g2['step'], g2['R_'+str(i)], label='R_'+str(i))
                ax1[i % a, i // a].plot(g2['step'], g2['CP_'+str(i)], label='CP_'+str(i))
                ax1[i % a, i // a].plot(g2['step'], g2['R_'+str(i)].ewm(5).mean(), label='R_'+str(i)+"_smooth")
                ax1[i % a, i // a].plot(g2['step'], g2['R_'+str(i)].ewm(5).mean().diff(10), label='CP_'+str(i)+"_smooth")
                # ax1[i % a, i // a].set_title(label=name)
                ax1[i % a, i // a].set_xlim([0, 200000])
                ax1[i % a, i // a].legend()
            # ax1[i % a, i // a].plot(g2['step'], g2['attempt_2'], label='attempt_2')
            # ax1[i % a, i // a].plot(g2['step'], g2['done_2'], label='done_2')
            # ax1[i % a, i // a].plot(g2['step'], g2[y].ewm(10).mean().diff(20), label=y+"'")
            break

        break

        # ax1[i % a, i // a].legend()
        # ax1[i % a, i // a].set_ylim([0, 0.0001])

if 1:

    df2 = df
    df2 = df2[(df2['--agent'] == 'dqngm')]
    # df2 = df2[(df2['--env'] == 'PlayroomG-v0')]
    # df2 = df2[(df2['--opt_init'] == 0)]
    # df2 = df2[(df2['--shaping'] == 0)]
    # df2 = df2[(df2['--theta'] == 1)]
    # df2 = df2[(df2['--theta'] == 0) | (df2['--theta'] == 1)]
    y = ['I_'+i for i in ['agent', 'light', 'sound', 'toy1', 'toy2']]
    # y = ['T_'+i for i in ['agent', 'passenger', 'taxi']]
    # y = ['R_'+str(i) for i in range(5)]
    # y = ['T']
    # y = ['R_0']
    def quant_inf(x):
        return x.quantile(0.2)
    def quant_sup(x):
        return x.quantile(0.8)
    op_dict = {a:[np.median, np.mean, quant_inf, quant_sup] for a in y}
    df2 = df2.groupby(x + params).agg(op_dict).reset_index()

    paramsStudied = []
    for param in params:
        l = df2[param].unique()
        print(param, l)
        if len(l) > 1:
            paramsStudied.append(param)

    print(params)
    a, b = 3,2
    fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False)
    for j, (name, g) in enumerate(df2.groupby(params)):
        for i, val in enumerate(y):
            # ax2[i % a, i // a].scatter(g['attempt_{}'.format(i)], g[val], label=val, s=1)
            # ax2[i % a, i // a].plot(range(1500), range(1500), 'r-')
            ax2[i % a, i // a].plot(g['step'], g[val]['median'], label=name)
            # ax2[i % a, i // a].fill_between(g['step'],
            #                                 g[val]['quant_inf'],
            #                                 g[val]['quant_sup'], alpha=0.25, linewidth=0)
            ax2[i % a, i // a].set_title(label=val)
            ax2[i % a, i // a].legend()
            # ax2[i % a, i // a].set_ylim([0, 10])

plt.show()
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

DIR = '../../log/cluster/last'
ENV = 'dqn*-v0'
runs = glob.glob(os.path.join(DIR, ENV, '*'))
frames = []

if 0:
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
# y = ['imitloss']
x = ['step']
params = ['--agent',
          '--batchsize',
          '--env',
          '--eval_freq',
          '--gamma',
          '--her',
          '--w0',
          '--w1',
          '--w2',
          '--per',
          '--imit',
          '--opt_init',
          '--shaping',
          '--theta',
          '--clipping',
          '--explo']

if 0:
    df1 = df
    # df1 = df1[(df1['--agent'] == 'dqng')]
    # df1 = df1[(df1['step'] > 4000)]
    # df1 = df1[(df1['--opt_init'] == 0)]
    df1 = df1[(df1['--theta'] == 3)]
    for param in params:
        print(df1[param].unique())
    a, b = 3,4
    fig1, ax1 = plt.subplots(a, b, figsize=(18,10), squeeze=False)
    y = 'R_2'
    for j, (name, g) in enumerate(df1.groupby(params)):
        for i, (num_run, g2) in enumerate(g.groupby('num_run')):
            ax1[i % a, i // a].scatter(g2['step_toy2'], g2['R_toy2'], s=1, c='b')
            print(g2['R_toy2'])
            # ax1[i % a, i // a].plot(g2['step'], g2['CP_toy2'])
            # ax1[i % a, i // a].plot(g2['step'], g2['R_toy2'])
            # ax1[i % a, i // a].plot(g2['step'], g2['attempt_toy2'])
            # ax1[i % a, i // a].plot(g2['step'], g2['R'+str(name)], label='R_'+str(i))
            # ax1[i % a, i // a].plot(g2['step'], g2['CP'+str(name)], label='CP_'+str(i))
            # ax1[i % a, i // a].plot(g2['step'], g2['R'+str(name)].ewm(5).mean(), label='R_'+str(i)+"_smooth")
            # ax1[i % a, i // a].plot(g2['step'], g2['R'+str(name)].ewm(5).mean().diff(10), label='CP_'+str(i)+"_smooth")
            # ax1[i % a, i // a].set_title(label=name)
            # ax1[i % a, i // a].set_xlim([0, 400000])
            ax1[i % a, i // a].legend()
            # ax1[i % a, i // a].plot(g2['step'], g2['attempt_2'], label='attempt_2')
            # ax1[i % a, i // a].plot(g2['step'], g2['done_2'], label='done_2')
            # ax1[i % a, i // a].plot(g2['step'], g2[y].ewm(10).mean().diff(20), label=y+"'")
            break
        break

    fig1.suptitle('theta: 3')

        # ax1[i % a, i // a].legend()
        # ax1[i % a, i // a].set_ylim([0, 0.0001])

if 1:

    df2 = df
    df2 = df2[(df2['--agent'] == 'dqngm')]
    df2 = df2[(df2['--env'] == 'PlayroomGM-v0')]
    # df2 = df2[(df2['--w1'] == 0)]
    # df2 = df2[(df2['--opt_init'] == 1)]
    # df2 = df2[(df2['--network'] == 2)]
    # df2 = df2[(df2['--clipping'] == 1)]
    # df2 = df2[(df2['--explo'] == 1)]
    # df2 = df2[(df2['--theta'] == 4)]
    # df2 = df2[(df2['--theta']     == 0) | (df2['--theta'] == 2)]
    y = ['R_xy']+['R'+s+str(i) for s in ['_light'] for i in range(1, 5)]
    # y = ['R' + i for i in ['_agent', '_light', '_key1', '_chest1', '_chest2', '_chest3']]
    # y = ['imit_loss', 'lambda_2_loss', 'loss', 'advantage_loss']
    # x = ['step' + i for i in ['_agent', '_light', '_sound', '_toy1', '_toy2']]
    # y = ['R'+i for i in ['_agent', '_passenger', '_taxi']]
    # x = ['step'+i for i in ['_agent', '_passenger', '_taxi']]
    # y = ['R_'+str(i) for i in range(5)]
    # y = ['T']
    # y = ['R_0']

    paramsStudied = []
    for param in params:
        l = df2[param].unique()
        print(param, l)
        if len(l) > 1:
            paramsStudied.append(param)
    print(df2['num_run'].unique())

    def quant_inf(x):
        return x.quantile(0.2)
    def quant_sup(x):
        return x.quantile(0.8)
    op_dict = {a:[np.median, np.mean, quant_inf, quant_sup] for a in y}
    df2 = df2.groupby(x + params).agg(op_dict).reset_index()



    print(paramsStudied)
    a, b = 2,3
    fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False)
    colors = ['b', 'r']

    for j, (name, g) in enumerate(df2.groupby(paramsStudied)):
        for i, val in enumerate(y):
            # ax2[i % a, i // a].scatter(g[x[i]], g[val], s=1)
            # ax2[i % a, i // a].plot(range(1500), range(1500), 'g-')
            if isinstance(name, tuple):
                label = ','.join(['{}:{}'.format(paramsStudied[k][2:], name [k]) for k in range(len(paramsStudied))])
            else:
                label = '{}:{}'.format(paramsStudied[0][2:], name)
            ax2[i % a, i // a].plot(g['step'], g[val]['median'], label=label)
            # ax2[i % a, i // a].plot(g['step'], g[val]['median'].ewm(5).mean().diff(10),
            #                         label='CP_' + str(i) + "_smooth")
            ax2[i % a, i // a].fill_between(g['step'],
                                            g[val]['quant_inf'],
                                            g[val]['quant_sup'], alpha=0.25, linewidth=0)
            ax2[i % a, i // a].set_title(label=val)
            ax2[i % a, i // a].legend()
            # ax2[i % a, i // a].set_ylim([0, 10])
    fig2.suptitle('network:3')

plt.show()
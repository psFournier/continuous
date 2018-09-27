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
          '--w0',
          '--w1',
          '--w2',
          '--per',
          '--imit',
          '--opt_init',
          '--shaping',
          '--theta']

if 0:
    df1 = df
    # df1 = df1[(df1['--agent'] == 'dqng')]
    # df1 = df1[(df1['step'] > 4000)]
    # df1 = df1[(df1['--opt_init'] == 0)]
    df1 = df1[(df1['--w1'] == 0) & (df1['--w2'] == 0)]
    df1 = df1[(df1['--w0'] == 1)]
    y = ['R_xy'] + ['R' + s + str(i) for s in ['_light'] for i in range(1, 5)]
    y = ['model_2_loss', 'model_3_loss', 'model_3_advantage_loss', 'model_3_imit_loss', 'model_3_lambda_2_loss']
    params += ['num_run']
    for param in params:
        print(df1[param].unique())
    a, b = 2,3
    fig1, ax1 = plt.subplots(a, b, figsize=(18,10), squeeze=False)
    for j, (num_run, g) in enumerate(df1.groupby('num_run')):
        for i, val in enumerate(y):
            ax1[i % a, i // a].plot(g['step'], g[val], label=num_run)
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
            # ax1[i % a, i // a].set_ylim([0, 1])

    fig1.suptitle('theta: 3')

        # ax1[i % a, i // a].legend()
        # ax1[i % a, i // a].set_ylim([0, 0.0001])

if 1:

    df2 = df
    df2 = df2[(df2['--agent'] == 'dqngm')]
    df2 = df2[(df2['--env'] == 'PlayroomGM-v0')]
    df2 = df2[(df2['--imit'] == 1)]
    # df2 = df2[(df2['--w1'] == 0) | (df2['--w1'] == 0.5) | (df2['--w1'] == 2)]
    df2 = df2[(df2['--w1'] == 0)]
    df2 = df2[(df2['--w0'] == 0)]
    df2 = df2[(df2['--opt_init'] == 0)]
    # df2 = df2[(df2['--network'] == 2)]
    # df2 = df2[(df2['--clipping'] == 1)]
    # df2 = df2[(df2['--explo'] == 1)]
    # df2 = df2[(df2['--theta'] == 0)]
    # df2 = df2[(df2['--theta'] == 4)]
    y = ['R']
    y = ['I'+s for s in ['_xy', '_light','_key1', '_key2', '_chest1', '_chest2', '_chest3', '_chest4']]
    # y = ['good_exp', 'loss_dqn', 'loss_dqn2', 'qval']
    # y = ['loss_imit']
    # y = ['model_2_loss', 'model_3_loss', 'model_3_advantage_loss', 'model_3_imit_loss', 'model_3_lambda_2_loss']
    # y = ['R' + i for i in ['_agent', '_light', '_key1', '_chest1', '_chest2', '_chest3']]
    # y = ['loss', 'advantage_loss']
    # x = ['step'+s+str(i) for s in ['_light', '_key','_chest'] for i in range(1, 5)]
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
    a, b = 3,3
    fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False, sharey=True, sharex=True)
    colors = ['b', 'r']
    p = 'num_run'
    p= paramsStudied
    for j, (name, g) in enumerate(df2.groupby(p)):
        for i, valy in enumerate(y):
            # ax2[i % a, i // a].plot(range(1500), range(1500), 'g-')
            if isinstance(name, tuple):
                label = ','.join(['{}:{}'.format(paramsStudied[k][2:], name [k]) for k in range(len(paramsStudied))])
            else:
                label = '{}:{}'.format(paramsStudied[0][2:], name)
            ax2[i % a, i // a].plot(g['step'], g[valy]['mean'], label=label)
            # ax2[i % a, i // a].plot(g['step'], abs(g[valy].diff(1).ewm(com=5).mean()))
            # ax2[i % a, i // a].scatter(g[x[i], g[valy], s=1, c=colors[j], label=label)
            # ax2[i % a, i // a].plot(g['step'], g[val]['median'].ewm(5).mean().diff(10),
            #                         label='CP_' + str(i) + "_smooth")
            ax2[i % a, i // a].fill_between(g['step'],
                                            g[valy]['quant_inf'],
                                            g[valy]['quant_sup'], alpha=0.25, linewidth=0)
            ax2[i % a, i // a].set_title(label=valy)
            ax2[i % a, i // a].legend()
            # ax2[i % a, i // a].set_ylim([0, 10])

plt.show()
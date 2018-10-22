import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

DIR = '../../log/cluster/last'
ENV = '*-v0'
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
          # '--w0',
          # '--w1',
          # '--w2',
          '--her',
          '--wimit',
          '--opt_init',
          '--shaping',
          '--theta',
          '--inv_grad',
          '--margin']

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
    # df2 = df2[(df2['--agent'] == 'ddpgg')]
    # df2 = df2[(df2['--env'] == 'Playroom2GM0-v0')]
    # df2 = df2[(df2['--imit'] == 2)]
    # # df2 = df2[(df2['--w1'] == 0) | (df2['--w1'] == 0.5) | (df2['--w1'] == 2)]
    df2 = df2[(df2['--wimit'] == 1)]
    # df2 = df2[(df2['--opt_init'] == -20)]
    # # df2 = df2[(df2['--network'] == 2)]
    # # df2 = df2[(df2['--clipping'] == 1)]
    # # df2 = df2[(df2['--explo'] == 1)]
    df2 = df2[(df2['--margin'] == 0.5)]
    df2 = df2[(df2['--theta'] == 1)]
    # y = ['R']
    # y = ['agentR']
    # y = ['agentR_'+s for s in ['[0.02]','[0.04]','[0.06]','[0.08]','[0.1]']]
    # y = ['agentR'+s for s in ['_light','_key1', '_key2', '_key3', '_key4', '_chest1', '_chest2', '_chest3', '_chest4']]
    y = ['agentC'+s for s in ['_pos', '_light','_key1', '_chest1']]

    # y = ['R_key1', 'R_key2', 'R_key3', 'R_key4', 'R_light1',
    #    'R_light2', 'R_light3', 'R_light4', 'R_xy']

    # y = ['loss_dqn', 'loss_imit', 'qval', 'val']
    # y = ['good_exp', 'loss_dqn2', 'qval2', 'val2']
    # y = ['loss_imit']
    # y = ['model_2_loss', 'model_3_loss', 'model_3_advantage_loss', 'model_3_imit_loss', 'model_3_lambda_2_loss']
    # y = ['R' + i for i in ['_agent', '_light', '_key1', '_chest1', '_chest2', '_chest3']]
    # y = ['loss', 'advantage_loss']
    # x = ['step'+s for s in ['_light','_key1', '_key2', '_key3', '_key4', '_chest1', '_chest2', '_chest3', '_chest4']]
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
    avg = 0
    if avg:
        df2 = df2.groupby(x + params).agg(op_dict).reset_index()

    print(paramsStudied)
    a, b = 2,2
    fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False, sharey=True, sharex=True)
    colors = ['b', 'r']
    p = 'num_run'
    if avg:
        p= paramsStudied
    for j, (name, g) in enumerate(df2.groupby(p)):
        if avg:
            if isinstance(name, tuple):
                label = ','.join(['{}:{}'.format(paramsStudied[k][2:], name[k]) for k in range(len(paramsStudied))])
            else:
                label = '{}:{}'.format(paramsStudied[0][2:], name)
        # ax2[0,0].plot(g['step'], g.iloc[:, g.columns.get_level_values(1) == 'mean'].mean(axis=1), label=label)
        # ax2[0,0].legend()
        for i, valy in enumerate(y):
            # ax2[i % a, i // a].plot(range(1500), range(1500), 'g-')

            # ax2[i % a, i // a].plot(g['step'], g[valy]['mean'], label=label)
            # ax2[i % a, i // a].plot(g['step'], g[valy]['mean'].ewm(com=5).mean(), label=label)
            if avg:
                ax2[i % a, i // a].plot(g['step'], g[valy]['median'], label=label)
            else:
                ax2[i % a, i // a].plot(g['step'], g[valy], label=None)
            # ax2[i % a, i // a].scatter(g[x[i]], g[valy], s=1, c=['red', 'blue', 'green'][j])
            # ax2[i % a, i // a].plot(g['step'], abs(g[valy].rolling(window=20).mean().diff(10)))
            # ax2[i % a, i // a].plot(g['step'], g[val]['median'].ewm(5).mean().diff(10),
            #                         label='CP_' + str(i) + "_smooth")
            # ax2[i % a, i // a].fill_between(g['step'],
            #                                 g[valy]['quant_inf'],
            #                                 g[valy]['quant_sup'], alpha=0.25, linewidth=0)
            ax2[i % a, i // a].set_title(label=valy)
            ax2[i % a, i // a].legend()
            # ax2[i % a, i // a].set_ylim([0, 100])
        # break

plt.show()
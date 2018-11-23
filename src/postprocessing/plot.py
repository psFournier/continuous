import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.signal import lfilter
import matplotlib.ticker as ticker
DIR = '../../log/cluster/last/'
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
          '--ep_tasks',
          '--rnd_demo',
          '--wimit',
          '--ep_steps',
          '--inv_grad',
          '--margin',
          '--demo',
          '--eps1',
          '--eps2',
          '--eps3',
          '--network',
          ]


df2 = df
# df2 = df2[(df2['--agent'] == 'ddpgg')]
# df2 = df2[(df2['--env'] == 'Playroom3GM-v0')]
# df2 = df2[(df2['--imit'] == 2)]
# df2 = df2[(df2['--tutorTask'] == 'hard')]
df2 = df2[(df2['--wimit'] == 1)]
# df2 = df2[(df2['--opt_init'] == -20)]
df2 = df2[(df2['--demo'] == 1)]
df2 = df2[(df2['--network'] == 0)]
# df2 = df2[(df2['--ep_tasks'] == 2)]
# df2 = df2[(df2['--ep_tasks'] == 1)]
df2 = df2[(df2['--margin'] == 0.3)]
# df2 = df2[(df2['--eps1'] == 0)]
# df2 = df2[(df2['--eps2'] == 0)]
# df2 = df2[(df2['--eps3'] == 1)]



# y = ['R']
# y = ['agentR']
# y = ['agentR_'+s for s in ['[0.02]','[0.04]','[0.06]','[0.08]','[0.1]']]
# y = ['agentR'+s for s in ['_light','_key1', '_key2', '_key3', '_key4', '_chest1', '_chest2', '_chest3', '_chest4']]
y = ['qval{}'.format(str(s)) for s in [i for i in [2, 3, 4, 5]]]
# y = ['loss_dqn{}'.format(str(s)) for s in [i for i in [2, 3, 4]]]

# x = ['attempts'+s for s in ['_light','_key1', '_chest1']]

# y = ['R_key1', 'R_key2', 'R_key3', 'R_key4', 'R_light1',
#    'R_light2', 'R_light3', 'R_light4', 'R_xy']

# y = ['loss_dqn', 'qval', 'loss_imit']
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
# y2 = ['CP'+s for s in ['_light','_key1', '_chest1']]
# y3 = ['trainstep'+s for s in ['_light','_key1', '_chest1']]


paramsStudied = []
for param in params:
    l = df2[param].unique()
    print(param, l)
    if len(l) > 1:
        paramsStudied.append(param)
print(df2['num_run'].unique())

op_dict = {a:[np.median, np.mean, np.std] for a in y}
avg = 0
if avg:
    df2 = df2.groupby(x + params).agg(op_dict).reset_index()


print(paramsStudied)
a, b = 2, 2
fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False, sharey=False, sharex=True)
colors = ['b', 'r']
p = 'num_run'
if avg:
    p= paramsStudied
# fig, ax = plt.subplots(1, 1, figsize=(18, 10), squeeze=False, sharey=True, sharex=True)

for j, (name, g) in enumerate(df2.groupby(p)):
    if avg:
        if isinstance(name, tuple):
            label = ','.join(['{}:{}'.format(paramsStudied[k][2:], name[k]) for k in range(len(paramsStudied))])
        else:
            label = '{}:{}'.format(paramsStudied[0][2:], name)
    # ax2[0,0].plot(g['step'], g.iloc[:, g.columns.get_level_values(1) == 'mean'].mean(axis=1), label=label)
    # ax2[0,0].legend()

    # g['alltasks'] = g[['trainsteps_[{}]'.format(s) for s in [2, 3, 4]]].apply(np.sum, axis=1)
    # ax[0, 0].plot(g['step'], g['alltasks'].cumsum())

    for i, valy in enumerate(y):
        # ax2[i % a, i // a].plot(range(1500), range(1500), 'g-')

        # ax2[i % a, i // a].plot(g['step'], g[valy]['mean'], label=label)
        # ax2[i % a, i // a].plot(g['step'], g[valy]['mean'].ewm(com=5).mean(), label=label)
        if avg:
            ax2[i % a, i // a].plot(g['step'][(g['step'] - 12000)%10000==0], 1/(g[valy]['mean'][(g['step'] - 12000)%10000==0]/(500/6)), label=label)

        else:
            # n = 50  # the larger n is, the smoother curve will be
            # yy = lfilter([1.0 / n] * n, 1, g[valy])
            ax2[i % a, i // a].plot(g['step'][(g['step'] - 12000)%10000==0], g[valy][(g['step'] - 12000)%10000==0]/(2000/6), label=None)
            # ax2[i % a, i // a].plot(g['step'], g[valy2], label=None)
            # ax2[i % a, i // a].plot(g['step'], g[valy3], label=None)

            # ax2[i % a, i // a].scatter(g[x[i]], g[valy], s=1)
        # ax2[i % a, i // a].plot(g['step'], abs(g[valy].rolling(window=20).mean().diff(10)))
        # ax2[i % a, i // a].plot(g['step'], g[val]['median'].ewm(5).mean().diff(10),
        #                         label='CP_' + str(i) + "_smooth")
        # ax2[i % a, i // a].fill_between(g['step'],
        #                                 g[valy]['quant_inf'],
        #                                 g[valy]['quant_sup'], alpha=0.25, linewidth=0)
        # ax2[i % a, i // a].fill_between(g['step'],
        #                                 g[valy]['mean'] - 0.5*g[valy]['std'],
        #                                 g[valy]['mean'] + 0.5*g[valy]['std'], alpha=0.25, linewidth=0)
        ax2[i % a, i // a].set_title(label=valy)
        if i == 0: ax2[i % a, i // a].legend()
        ax2[i % a, i // a].set_xlim([0, 300001])

        ax2[i % a, i // a].xaxis.set_major_locator(ticker.MultipleLocator(50000))
        ax2[i % a, i // a].xaxis.set_minor_locator(ticker.MultipleLocator(10000))
        ax2[i % a, i // a].grid(True, which='minor')
        # ax2[i % a, i // a].set_ylim([0, 2000])
    # break
    # ax[0,0].legend()

plt.show()
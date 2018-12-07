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
          '--prop_demo',
          '--freq_demo',
          '--deter',
          '--filter',
          '--lrimit',
          '--rndv',
          '--features',
          '--tutoronly'
          ]


df2 = df
# df2 = df2[(df2['--agent'] == 'ddpgg')]
# df2 = df2[(df2['--env'] == 'Playroom3GM-v0')]
# df2 = df2[(df2['--imit'] == 2)]
# df2 = df2[(df2['--tutorTask'] == 'hard')]
# df2 = df2[(df2['--wimit'] == 1)]
# df2 = df2[(df2['--filter'] == 2)]
# df2 = df2[(df2['--margin'] == 10)]

# df2 = df2[(df2['--demo'] == 0)]
# df2 = df2[(df2['--rndv'] == 1)]
# df2 = df2[(df2['--deter'] == 0)]
# df2 = df2[(df2['--freq_demo'] == 20000)]
# df2 = df2[(df2['--eps1'] == 0)]
# df2 = df2[(df2['--eps2'] == 0)]
# df2 = df2[(df2['--eps3'] == 1)]
# df2 = df2[(df2['--tutoronly'] == -1)]
df2 = df2[(df2['--demo'] == 4)]

y = ['C{}'.format(str(s)) for s in [i for i in [2, 3, 4]]]


for i in [2, 3, 4, 5, 6, 7]:
    df2['qval'+str(i)] = df2['qval'+str(i)] * df2['envstep'+str(i)]/2000

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

op_dict = {a:[np.median, np.mean, np.std, quant_inf, quant_sup] for a in y}
avg = 1
if avg:
    df2 = df2.groupby(x + params).agg(op_dict).reset_index()


print(paramsStudied)
a, b = 2,2
# a, b = 2, 3

fig2, ax2 = plt.subplots(a, b, figsize=(18,10), squeeze=False, sharey=True, sharex=True)
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
        # m = 6/(20000*0.015)
        # m = (6/2000)
        m = 1
        if avg:
            # ax2[i % a, i // a].plot(g['step'][g[valy]['mean']!=0], g[valy]['mean'][g[valy]['mean']!=0] * m, label=label)
            ax2[i % a, i // a].plot(g['step'], g[valy]['median'] * m, label=label)
        else:
            # n = 50  # the larger n is, the smoother curve will be
            # yy = lfilter([1.0 / n] * n, 1, g[valy])
            ax2[i % a, i // a].plot(g['step'], g[valy] * m, label=None)
            # ax2[i % a, i // a].plot(g['step'], g[valy2], label=None)
            # ax2[i % a, i // a].plot(g['step'], g[valy3], label=None)

            # ax2[i % a, i // a].scatter(g[x[i]], g[valy], s=1)
        # ax2[i % a, i // a].plot(g['step'], abs(g[valy].rolling(window=20).mean().diff(10)))
        # ax2[i % a, i // a].plot(g['step'], g[val]['median'].ewm(5).mean().diff(10),
        #                         label='CP_' + str(i) + "_smooth")
        ax2[i % a, i // a].fill_between(g['step'],
                                        g[valy]['quant_inf'],
                                        g[valy]['quant_sup'], alpha=0.25, linewidth=0)
        # ax2[i % a, i // a].fill_between(g['step'],
        #                                 g[valy]['mean'] - 0.5*g[valy]['std'],
        #                                 g[valy]['mean'] + 0.5*g[valy]['std'], alpha=0.25, linewidth=0)
        ax2[i % a, i // a].set_title(label=valy)
        if i == 0: ax2[i % a, i // a].legend()
        ax2[i % a, i // a].set_xlim([0, 300000])

        ax2[i % a, i // a].xaxis.set_major_locator(ticker.MultipleLocator(50000))
        ax2[i % a, i // a].xaxis.set_minor_locator(ticker.MultipleLocator(10000))
        ax2[i % a, i // a].grid(True, which='minor')
        # ax2[i % a, i // a].set_ylim([0, 1000])
    # break
    # ax[0,0].legend()

plt.show()
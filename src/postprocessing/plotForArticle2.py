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
df2 = df2[(df2['--wimit'] == 1)]
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
# df2 = df2[(df2['--demo'] != -1)]
# for i in [2, 3, 4, 5, 6, 7]:
#     df2['qval'+str(i)] = df2['qval'+str(i)]

y = ['qval4']
x = ['step']

def quant_inf(x):
    return x.quantile(0.2)
def quant_sup(x):
    return x.quantile(0.8)

op_dict = {a:[np.median, np.mean, np.std, quant_inf, quant_sup] for a in y}
df2 = df2.groupby(x + params).agg(op_dict).reset_index()

for param in params:
    l = df2[param].unique()
    print(param, l)

a, b = 2,2
fig, ax = plt.subplots(a, b, figsize=(18,9), squeeze=False, sharey=True, sharex=True)
fig.text(0.08, 0.5, 'Competence of the agent on affecting object 1', va='center', rotation='vertical', fontsize='large')

for i, (n1, g1) in enumerate(df2.groupby('--demo')):
    for j, (n2, g2) in enumerate(g1.groupby('--eps3')):
        ax[i % a, i // a].plot(g2['step'], g2['qval4']['median'], label=n2)
        # ax[i % a, i // a].fill_between(g2['step'],
        #                                g2['C4']['quant_inf'],
        #                                g2['C4']['quant_sup'], alpha=0.25, linewidth=0)
        # ax2[i % a, i // a].fill_between(g['step'],
        #                                 g[valy]['mean'] - 0.5*g[valy]['std'],
        #                                 g[valy]['mean'] + 0.5*g[valy]['std'], alpha=0.25, linewidth=0)
        ax[i % a, i // a].set_title(label=n1)
        if i == 0: ax[i % a, i // a].legend(loc='lower right', bbox_to_anchor=(0.97, 0.035), prop={'size':10})
        ax[i % a, i // a].set_xlim([0, 300000])

        ax[i % a, i // a].xaxis.set_major_locator(ticker.MultipleLocator(50000))
        ax[i % a, i // a].xaxis.set_minor_locator(ticker.MultipleLocator(10000))
        ax[i % a, i // a].grid(True, which='minor')
        # ax2[i % a, i // a].set_ylim([0, 1000])

plt.show()
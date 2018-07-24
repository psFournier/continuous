import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

runs = glob.glob('../../log/local/dqnfd2_TaxiTutor-v0_1/*/')
frames = []

for run in runs:

    config = pd.read_json(os.path.join(run, 'config.txt'), lines=True)

    try:
        df = pd.read_json(os.path.join(run, 'log_steps', 'progress.json'), lines=True)
        config = pd.concat([config] * df.shape[0], ignore_index=True)
        data = pd.concat([df, config], axis=1)
        data['num_run'] = run.split('/')[5]
        frames.append(data)
    except:
        print(run, 'not ok')

# Creating the complete dataframe with all dat
df = pd.concat(frames, ignore_index=True)
df = df[['step', 'CP_0', 'CP_1', 'freq_0', 'freq_1', 'comp_0', 'comp_1']]
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

fig, axes = plt.subplots(3, 1, figsize=(18,10))
# ax.plot(df['step'], df['list_returns_0'].rolling(10).mean())
axes[0].plot(df['step'], df['comp_0'].rolling(10).mean(), label='pick_up')
axes[0].plot(df['step'], df['comp_1'].rolling(10).mean(), label='drop-off')
axes[0].legend()

axes[1].plot(df['step'], df['CP_0'].rolling(10).mean(), label='pick_up')
axes[1].plot(df['step'], df['CP_1'].rolling(10).mean(), label='drop-off')
axes[1].legend()

axes[2].plot(df['step'], df['freq_0'].rolling(10).mean(), label='pick_up')
axes[2].plot(df['step'], df['freq_1'].rolling(10).mean(), label='drop-off')
axes[2].legend()


plt.show()
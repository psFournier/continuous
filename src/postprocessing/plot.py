import glob
import pandas as pd
import matplotlib.pyplot as plt
import os

runs = glob.glob('../../log/local/td3_Ant-v2_0_0.4_128_no_no_0_0_10/20180621001323_058909/')
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
df = df[['step', 'avg_return']]
print(df.head())

# def _0(x) : return x[0]
# def _1(x) : return x[1]
# funcs = [_0, _1]
# op_dict = {'list_returns': funcs}
# agg = df.agg(op_dict)
# agg.columns = agg.columns.map(''.join)
# df = pd.concat([df, agg], axis=1).drop(['list_returns'], axis=1)

fig, ax = plt.subplots(figsize=(18,10))
# ax.plot(df['step'], df['list_returns_0'].rolling(10).mean())
ax.plot(df['step'], df['avg_return'].rolling(10).mean())

plt.show()
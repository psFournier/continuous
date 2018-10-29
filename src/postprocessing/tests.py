import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from scipy.signal import lfilter, savgol_filter

def sigm(x, l):
    return 1 / (1 + np.exp(-l*x))

t = np.linspace(0, 500000, 10000)
cos = 0.04*np.cos(2*np.pi*3.85*t/500000)
sin = np.sin(2*np.pi*0.75*(t/500000)*(1-(t/500000)) + 2.1)
sin = 0
x1 = sigm(t - 2.5e5, 2e-5) + cos + sin
x2 = sigm(t - 2e5, 2.5e-5)  + cos + sin
x3 = sigm(t - 1.4e5, 4.5e-5) + cos + sin
x4 = sigm(t - 5.5e5, 1e-5) + cos + sin
x5 = sigm(t - 1e5, 4e-5) + cos + sin
noise = np.random.randn(len(t)) * np.random.randn(len(t)) * 0.05
# samples1 = np.clip(np.random.normal(sigm(t - 2.5e5, 2e-5), scale=0.05), 0, 1)
# samples5 = np.clip(np.random.normal(sigm(t - 2e5, 2.5e-5), scale=0.05), 0, 1)
#
# samples2 = np.clip(np.random.normal(sigm(t - 1.4e5, 4.5e-5), scale=0.05), 0, 1)
# samples3 = np.clip(np.random.normal(sigm(t - 5.5e5, 1e-5), scale=0.05), 0, 1)
# samples4 = np.clip(np.random.normal(sigm(t - 1e5, 4e-5), scale=0.05), 0, 1)
samples = np.stack([np.clip(x+noise, 0, 1) for x in  [x1, x2, x3, x4, x5]], axis=1)
columns = ['1', '2', '3', '4', '5']
samples = pd.DataFrame(samples, index=t, columns=columns)
res = ['d'+i for i in columns]
samples[res] = samples[columns].ewm(span=10).mean()
samples[res] = samples[res].apply(lambda col: savgol_filter(col, 99, 10), axis=0)
# samples[['d'+i for i in columns]] = .rolling(window=100).mean().diff(100)

for c in columns:
    samples['norm_'+c] = samples[['d'+i for i in columns]].\
        apply(lambda row: (row['d'+c] - np.min(row)) / (np.max(row) - np.min(row)), axis=1)

fig2, ax2 = plt.subplots(2, 1, figsize=(18,10), squeeze=False, sharey=False, sharex=True)
for col in columns:
    ax2[0, 0].plot(samples[col])
    ax2[1, 0].plot(samples['d'+col])
    ax2[1, 0].set_ylim([0, 1.1])
plt.show()
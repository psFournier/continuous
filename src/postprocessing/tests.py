import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

def sigm(x, l):
    return 1 / (1 + np.exp(-l*x))

t = np.array(range(0, 500000, 200))

scale1 = np.linspace(0.01, 0.2, len(t)//2)
scale2 = np.linspace(0.2, 0.01, len(t) - len(t)//2)
scale = np.hstack([scale1, scale2])
samples1 = np.clip(np.random.normal(sigm(t - 2.5e5, 2e-5), scale=0.05), 0, 1)
samples5 = np.clip(np.random.normal(sigm(t - 2e5, 2.5e-5), scale=0.05), 0, 1)

samples2 = np.clip(np.random.normal(sigm(t - 1.4e5, 4.5e-5), scale=0.05), 0, 1)
samples3 = np.clip(np.random.normal(sigm(t - 8e5, 1e-5), scale=0.05), 0, 1)
samples4 = np.clip(np.random.normal(sigm(t - 1e5, 4e-5), scale=0.05), 0, 1)
samples = np.stack([samples1, samples2, samples3, samples4, samples5], axis=1)
columns = ['1', '2', '3', '4', '5']
samples = pd.DataFrame(samples, index=t, columns=columns)
derivated = samples.rolling(window=100).mean().diff(100)

def sum(row):
    return np.sum([np.exp(row[col] * 20) for col in columns])
derivated['sum'] = derivated.apply(sum, axis=1)

for c in columns:
    derivated['soft'+c] = derivated.apply(lambda row: np.exp(row[c] * 20) / row['sum'], axis=1)

fig2, ax2 = plt.subplots(2, 1, figsize=(18,10), squeeze=False, sharey=False, sharex=True)
for col in columns:
    ax2[0, 0].plot(samples[col])
    ax2[1, 0].plot(derivated['soft'+col])
    ax2[1, 0].set_ylim([0, 1])
plt.show()
from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 20, maxlen=200):
        self.window = window
        self.r = deque(maxlen=maxlen)
        self.t = deque(maxlen=maxlen)
        self.CP = [0]
        self.R = [-200]
        self.T = [0]

    def append(self, val):
        self.r.append(val)
        self.t.append(val > -200)
        self.update()

    def update(self):
        Rs = list(self.r)[-(min(self.size, self.window)):]
        self.R.append(np.mean(Rs))
        newCP = self.R[-1] - self.R[-(min(self.size, 10))]
        self.CP.append(0.8 * self.CP[-1] + 0.2 * newCP)
        self.T.append(np.mean([1 if r > -200 else 0 for r in Rs]))

    @property
    def size(self):
        return len(self.r)

    @property
    def full(self):
        return self.size >= 2 * self.window


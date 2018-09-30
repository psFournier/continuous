from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 20, maxlen=200):
        self.window = window
        self.r = deque(maxlen=maxlen)
        self.CP = [0]
        self.R = [0]

    def append(self, val):
        self.r.append(val)
        self.update()

    def update(self):
        Rs = list(self.r)[-(min(self.size, self.window)):]
        self.R.append(np.mean(Rs))
        newCP = self.R[-1] - self.R[-(min(self.size, 10))]
        self.CP.append(0.8 * self.CP[-1] + 0.2 * newCP)

    @property
    def size(self):
        return len(self.r)

    @property
    def full(self):
        return self.size >= 2 * self.window


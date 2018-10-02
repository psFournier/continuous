from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 200, maxlen=201):
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
        self.CP.append(newCP)

    @property
    def size(self):
        return len(self.r)

    @property
    def full(self):
        return self.size >= 2 * self.window


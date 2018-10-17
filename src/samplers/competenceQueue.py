from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 100, maxlen=200):
        self.window = window
        self.mcr = deque(maxlen=maxlen)
        self.t = deque(maxlen=maxlen)
        self.CP = [0]
        self.MCR = [0]
        self.T = [0]

    def append(self, mcr, t):
        self.mcr.append(mcr)
        self.t.append(t)
        self.update()

    def update(self):
        if self.size > 10:
            window = min(self.size, self.window)
            MCRs = list(self.mcr)[-window:]
            Ts = list(self.t)[-window:]
            self.MCR.append(np.mean(MCRs))
            self.T.append(np.mean(Ts))
            newCP = self.MCR[-1] - self.MCR[-min(len(self.MCR), self.window)]
            # newCP = self.T[-1] - self.T[-min(len(self.T), self.window)]
            self.CP.append(newCP)

    @property
    def size(self):
        return len(self.mcr)

    @property
    def full(self):
        return self.size >= 2 * self.window


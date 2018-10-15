from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 200, maxlen=201):
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
        MCRs = list(self.mcr)[-(min(self.size, self.window)):]
        Ts = list(self.t)[-(min(self.size, self.window)):]
        self.MCR.append(np.mean(MCRs))
        self.T.append(np.mean(Ts))
        # newCP = self.MCR[-1] - self.MCR[-(min(self.size, 10))]
        newCP = self.T[-1] - self.T[-(min(self.size, 10))]
        self.CP.append(newCP)

    @property
    def size(self):
        return len(self.mcr)

    @property
    def full(self):
        return self.size >= 2 * self.window


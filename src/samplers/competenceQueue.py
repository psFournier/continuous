from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 20, maxlen=200):
        self.window = window
        self.points = {key: deque(maxlen=maxlen) for key in ['R', 'S', 'T']}
        self.CP = []
        self.R = []
        self.S = []
        self.T = []

    def append(self, point):
        for key, val in point.items():
            self.points[key].append(val)

    def update(self):
        if self.size >= 4:
            Rs = list(self.points['R'])
            Ss = list(self.points['S'])
            Ts = list(self.points['T'])
            mid = min(self.size // 2, self.window)
            self.R.append(np.mean(Rs[-mid:]))
            self.S.append(np.mean(Ss[-mid:]))
            self.T.append(np.mean(Ts[-mid:]))
        if len(self.R) >= 8:
            self.CP.append(self.R[-1] - self.R[-8])

    @property
    def mincp(self):
        return 1/self.window

    @property
    def size(self):
        return len(self.points['R'])

    @property
    def full(self):
        return self.size >= 2 * self.window


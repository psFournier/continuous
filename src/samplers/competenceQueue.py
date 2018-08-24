from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 20, maxlen=200):
        self.window = window
        self.points = {key: deque(maxlen=maxlen) for key in ['R', 'S', 'step']}
        self.CP = 0.001
        self.R = 0.001
        self.S = 0.001

    def append(self, point):
        for key, val in point.items():
            self.points[key].append(val)
        Rs = list(self.points['R'])
        Ss = list(self.points['S'])
        steps = list(self.points['step'])
        if self.size >= 4:
            mid = min(self.size // 2, self.window)
            R1 = np.mean(Rs[-mid:])
            # s1 = np.mean(steps[-mid:])
            R2 = np.mean(Rs[-2*mid:-mid])
            # s2 = np.mean(steps[-2*mid:-mid])
            self.CP = (R2 - R1)
            self.R = R1
            self.S = np.mean(Ss[-mid:])

    @property
    def size(self):
        return len(self.points['R'])

    @property
    def full(self):
        return self.size >= 2 * self.window


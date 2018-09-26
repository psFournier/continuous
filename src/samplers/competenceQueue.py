from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 20, maxlen=200):
        self.window = window
        self.points = deque(maxlen=maxlen)
        self.CP = []
        self.R = [-200]

    def append(self, val):
        self.points.append(val)
        self.update()

    def update(self):
        Rs = list(self.points)
        self.R.append(np.mean(Rs[-(min(self.size, self.window)):]))
        self.CP.append(self.R[-1] - self.R[-(min(self.size, 10))])

    @property
    def size(self):
        return len(self.points)

    @property
    def full(self):
        return self.size >= 2 * self.window


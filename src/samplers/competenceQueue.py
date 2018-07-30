from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 20, maxlen=200):
        self.window = window
        self.points = deque(maxlen=maxlen)
        self.CP = 0.001
        self.R_mean = 0.001
        self.T_mean = 0

    def update_CP(self):
        if self.size > 2:
            window = min(self.size // 2, self.window)
            Rs = [point[1] for point in self.points]
            Ts = [point[2] for point in self.points]
            R1 = list(itertools.islice(Rs, self.size - window, self.size))
            comp1 = np.sum(R1) / window
            R2 = list(itertools.islice(Rs, 0, self.size - window))
            comp2 = np.sum(R2) / (self.size - window)
            self.CP = comp1 - comp2
            self.R_mean = np.sum(Rs) / self.size
            self.T_mean = np.sum(Ts) / self.size

    def append(self, point):
        self.points.append(point)
        self.update_CP()

    @property
    def size(self):
        return len(self.points)

    @property
    def full(self):
        return self.size >= 2 * self.window


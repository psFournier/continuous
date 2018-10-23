from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 100, maxlen=200):
        self.window = window
        self.C = deque(maxlen=maxlen)
        self.C_avg = 0
        self.CP = 0

    def append(self, C):
        self.C.append(C)
        self.update()

    def update(self):
        size = len(self.C)
        if size > 10:
            window = min(size, self.window)
            Cs = list(self.C)
            C1 = Cs[-window//2:]
            C2 = Cs[-window:-window//2]
            self.C_avg = np.mean(C1)
            self.CP = self.C_avg - np.mean(C2)


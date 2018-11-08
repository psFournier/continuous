from collections import deque
from gym.spaces import Box
import itertools
import numpy as np
from scipy.signal import lfilter


class CompetenceQueue():
    def __init__(self, window = 100, maxlen=250):
        self.window = window
        self.C = deque(maxlen=maxlen)
        self.C_avg = deque(maxlen=10)
        self.CP = 0

    def append(self, C):
        self.C.append(C)
        # self.update()

    def update(self):
        size = len(self.C)
        if size > 2:
            window = min(size, self.window)
            self.C_avg.append(np.mean(list(self.C)[-window:]))
            self.CP = self.C_avg[-1] - self.C_avg[0]

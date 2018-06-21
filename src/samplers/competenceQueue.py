from collections import deque
from gym.spaces import Box
import itertools
import numpy as np

class CompetenceQueue():
    def __init__(self, window = 10, maxlen=20):
        self.window = window
        self.points = deque(maxlen=maxlen)
        self.CP = 0.001
        self.competence = 0.001

    def update_CP(self):
        if self.size > 2:
            window = min(self.size // 2, self.window)
            q = [point[2] for point in self.points]
            q1 = list(itertools.islice(q, self.size - window, self.size))
            q2 = list(itertools.islice(q, self.size - 2 * window, self.size - window))
            self.CP = max(np.abs(np.sum(q1) - np.sum(q2)) / (2 * window), 0.001)
            self.competence = np.sum(q1) / window

    def append(self, point):
        self.points.append(point)
        self.update_CP()

    @property
    def size(self):
        return len(self.points)

    @property
    def full(self):
        return self.size >= 2 * self.window


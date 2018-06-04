from gym.spaces import Box
import numpy as np
from ddpg.competenceQueue import CompetenceQueue

class Region(Box):

    def __init__(self, low = np.array([-np.inf]), high=np.array([np.inf])):
        super(Region, self).__init__(low, high)
        self.queue = CompetenceQueue()
        self.dim_split = None
        self.val_split = None
        self.freq = 0
        # self.max_CP = 0
        # self.min_CP = 0
        self.sum_CP = 0

    def sample(self):
        return np.random.uniform(low=self.low, high=self.high)

    def contains(self, x):
        shape_ok = (x.shape == self.low.shape)
        low_ok = (x >= self.low).all()
        high_ok = (x <= self.high).all()
        return shape_ok and low_ok and high_ok

    def split(self, dim, split_val):
        low_right = np.copy(self.low)
        low_right[dim] = split_val
        high_left = np.copy(self.high)
        high_left[dim] = split_val
        left = Region(self.low, high_left)
        right = Region(low_right, self.high)

        left.queue.CP = self.queue.CP
        right.queue.CP = self.queue.CP
        # left.queue.competence = self.queue.competence
        # right.queue.competence = self.queue.competence
        for point in self.queue.points:
            if left.contains(point[0]):
                left.queue.points.append(point)
            else:
                right.queue.points.append(point)
        left.queue.update_CP()
        right.queue.update_CP()

        return left, right

    def add(self, point):
        self.queue.append(point)


    @property
    def is_leaf(self):
        return (self.dim_split is None)
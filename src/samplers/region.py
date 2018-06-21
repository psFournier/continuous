from gym.spaces import Box
import numpy as np
from samplers.competenceQueue import CompetenceQueue

class Region(Box):

    def __init__(self, low = np.array([-np.inf]), high=np.array([np.inf]), window=10, maxlen=20, dtype='float32'):
        super(Region, self).__init__(low=low, high=high, dtype=dtype)
        self.queue = CompetenceQueue(window=window, maxlen=maxlen)
        self.dim_split = None
        self.val_split = None
        self.line = None
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
        eval = self.eval_split(left, right)
        return left, right, eval

    def add(self, point):
        self.queue.append(point)

    def eval_split(self, left, right):
        return left.size * right.size * np.sqrt((right.CP-left.CP)**2)

    def best_split(self, dims, n_split, split_min):
        best = 0
        best_left, best_right = Region(), Region()
        for dim in dims:
            sub_regions = np.linspace(self.low[dim], self.high[dim], n_split+2)
            for num_split, split_val in enumerate(sub_regions[1:-1]):
                left, right, eval = self.split(dim, split_val)
                if eval > best and eval > split_min:
                    best_left = left
                    best_right = right
                    self.val_split = split_val
                    self.dim_split = dim
                    best = eval
        return best_left, best_right

    def compute_line(self):
        if not self.is_leaf:
            if self.dim_split == 0:
                line1_xs = 2 * [self.val_split]
                line1_ys = [self.low[1], self.high[1]]
            else:
                line1_ys = 2 * [self.val_split]
                line1_xs = [self.low[0], self.high[0]]
            self.line = [line1_xs, line1_ys]

    @property
    def is_leaf(self):
        return (self.dim_split is None and self.is_init)

    @property
    def is_init(self):
        return (not np.isinf(self.high[0]))

    @property
    def CP(self):
        return self.queue.CP

    @property
    def size(self):
        return self.queue.size

    @property
    def area(self):
        return (self.high[0] - self.low[0]) * (self.high[1] - self.low[1])
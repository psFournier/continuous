from samplers.region import Region

import numpy as np

class TreeSampler():
    def __init__(self, space, R=128, auto=True, theta=1):
        self.n_split = 10
        self.split_min = 0
        self.nRegions = R
        self.auto = auto
        self.dims = range(space.low.shape[0])
        self.lines = []
        self.theta = theta

        capacity = 1
        while capacity < self.nRegions:
            capacity *= 2
        self.capacity = capacity

        self.region_array = [Region() for _ in range(2 * self.capacity)]
        self.n_leaves = 0
        self.n_points = 0

        self.initialize(space)
        self.update_CP_tree()

    def initialize(self, space):
        self.region_array[1] = Region(space.low, space.high, maxlen=int(1e5))
        self.n_leaves += 1
        assert self.nRegions & (self.nRegions - 1) == 0  # n must be a power of 2
        if not self.auto:
            self._divide(1, self.nRegions, 0)

    def _divide(self, idx , n, dim_idx):
        if n > 1:
            dim = self.dims[dim_idx]
            region = self.region_array[idx]
            low = region.low[dim]
            high = region.high[dim]
            val_split = (high+low)/2
            self.region_array[2 * idx], self.region_array[2 * idx + 1], _ = region.split(dim, val_split)
            region.dim_split = dim
            region.val_split = val_split
            self.n_leaves += 1
            region.compute_line()
            self.lines.append(region.line)
            next_dim_idx = (dim_idx+1)%(len(self.dims))
            self._divide(2 * idx, n/2, next_dim_idx)
            self._divide(2 * idx + 1, n/2, next_dim_idx)

    def append(self, point):
        regions_idx = self.find_regions(point[0])
        for idx in regions_idx:
            region = self.region_array[idx]
            region.add(point)
            self.n_points += 1
            to_split = self.auto and region.queue.full and idx < self.capacity and region.is_leaf
            if to_split:
                self.region_array[2 * idx], self.region_array[2 * idx + 1] = region.best_split(self.dims, self.n_split, self.split_min)
                if not region.is_leaf:
                    self.n_leaves += 1
                    region.compute_line()
                    self.lines.append(region.line)
        self.update_CP_tree()


    def find_regions(self, sample):
        regions = self._find_regions(sample, 1)
        return regions

    def _find_regions(self, sample, idx):
        regions = [idx]
        region = self.region_array[idx]
        if not region.is_leaf:
            left = self.region_array[2 * idx]
            if left.contains(sample):
                regions_left = self._find_regions(sample, 2 * idx)
                regions += regions_left
            else:
                regions_right = self._find_regions(sample, 2 * idx + 1)
                regions += regions_right
        return regions

    def find_prop_region(self, sum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= sum
        """
        assert 0 <= sum <= self.sum_CP + 1e-5
        idx = 1
        while not self.region_array[idx].is_leaf:
            s = self.region_array[2 * idx].sum_CP
            if s > sum:
                idx = 2 * idx
            else:
                sum -= s
                idx = 2 * idx + 1
        return self.region_array[idx]

    def sample(self, rnd_prop):
        if np.random.random() > rnd_prop:
            sum = self.sum_CP
            mass = np.random.random() * sum
            region = self.find_prop_region(mass)
        else:
            leaf = np.random.choice(self.list_leaves)
            region = self.region_array[leaf]
        region.freq += 1
        sample = region.sample().flatten()
        return sample

    def update_CP_tree(self):
        self._update_CP_tree(1)

    def _update_CP_tree(self, idx):
        region = self.region_array[idx]
        if region.is_leaf:
            region.sum_CP = region.queue.CP ** self.theta
        else:
            self._update_CP_tree(2 * idx)
            self._update_CP_tree(2 * idx + 1)
            left = self.region_array[2 * idx]
            right = self.region_array[2 * idx + 1]
            region.sum_CP = np.sum([left.sum_CP, right.sum_CP])

    def stats(self):
        stats = {}
        # stats['list_goal'] = [goal[self.buffer.envs.internal] for goal in self.goal_set]
        stats['list_CP'] = self.list_CP
        stats['list_comp'] = self.list_comp
        stats['list_freq'] = self.region_freq
        # stats['max_CP'] = self.max_CP
        # stats['min_CP'] = self.min_CP
        return stats

    @property
    def list_leaves(self):
        return [i for i,r in enumerate(self.region_array) if r.is_leaf]

    @property
    def list_CP(self):
        return [float("{0:.3f}".format(region.queue.CP)) for region in self.region_array]

    @property
    def max_CP_leaf(self):
        return self.list_leaves[np.argmax([self.list_CP[i] for i in self.list_leaves]).squeeze()]

    @property
    def list_comp(self):
        return [float("{0:.3f}".format(region.queue.competence)) for region in self.region_array]

    @property
    def region_freq(self):
        return [region.freq for region in self.region_array]

    @property
    def root(self):
        return self.region_array[1]

    @property
    def sum_CP(self):
        return self.root.sum_CP

    @property
    def points(self):
        return list(self.root.queue.points)


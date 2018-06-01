from ddpg.region import Region

import numpy as np


# import matplotlib.pyplot as plt
# import matplotlib.lines as lines
# import matplotlib.patches as patches
# from matplotlib.collections import PatchCollection
# Blues = plt.get_cmap('Blues')

class RegionTree():
    def __init__(self, space, nRegions, beta, render):
        self.n_split = 10
        self.split_min = 0.0001
        self.nRegions = nRegions
        self.beta = beta
        self.dims = range(space.low.shape[0])

        capacity = 1
        while capacity < self.nRegions:
            capacity *= 2
        self.capacity = capacity

        self.region_array = [Region() for _ in range(2 * self.capacity)]
        self.n_leaves = 0
        self.initialize(space)
        self.update_CP_tree()

        self.render = render
        # self.ax = None
        # self.figure = None
        # self.lines = []
        # self.patches = []
        # self.points = []
        # self.history = []


    def initialize(self, space):
        self.region_array[1] = Region(space.low, space.high)
        self.n_leaves += 1
        assert self.nRegions & (self.nRegions - 1) == 0  # n must be a power of 2
        self._divide(1, self.nRegions, 0)

    def _divide(self, idx , n, dim_idx):
        if n > 1:
            dim = self.dims[dim_idx]
            region = self.region_array[idx]
            low = region.low[dim]
            high = region.high[dim]
            val_split = (high+low)/2
            self.region_array[2 * idx], self.region_array[2 * idx + 1] = region.split(dim, val_split)
            region.dim_split = dim
            region.val_split = val_split
            self.n_leaves += 1
            next_dim_idx = (dim_idx+1)%(len(self.dims))
            self._divide(2 * idx, n/2, next_dim_idx)
            self._divide(2 * idx + 1, n/2, next_dim_idx)

    def append(self, point):
        target = point[0]
        success = point[1]
        regions_idx = self.find_regions(target)
        for idx in regions_idx:
            self.region_array[idx].add((target, success))
        self.update_CP_tree()
        # self.update_display()

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
            leaf = np.random.randint(self.n_leaves)
            region = self.region_array[self.capacity + leaf]
        region.freq += 1
        return region

    def update_CP_tree(self):
        self._update_CP_tree(1)

    def _update_CP_tree(self, idx):
        region = self.region_array[idx]
        if region.is_leaf:
            # region.max_CP = region.queue.CP
            # region.min_CP = region.queue.CP
            region.sum_CP = region.queue.CP
        else:
            self._update_CP_tree(2 * idx)
            self._update_CP_tree(2 * idx + 1)
            left = self.region_array[2 * idx]
            right = self.region_array[2 * idx + 1]
            # region.max_CP = np.max([left.max_CP, right.max_CP])
            # region.min_CP = np.min([left.min_CP, right.min_CP])
            region.sum_CP = np.sum([left.sum_CP, right.sum_CP])

    def stats(self):
        stats = {}
        # stats['list_goal'] = [goal[self.buffer.env.internal] for goal in self.goal_set]
        stats['list_CP'] = self.list_CP
        stats['list_comp'] = self.list_comp
        stats['list_freq'] = self.region_freq
        # stats['max_CP'] = self.max_CP
        # stats['min_CP'] = self.min_CP
        return stats

    # def update_display(self):
    #     if self.render:
    #         self.compute_image()
    #         if self.figure is None:
    #             self.init_display()
    #         self.plot_image()
    #
    # def compute_image(self):
    #     self.lines.clear()
    #     self.patches.clear()
    #     self.points.clear()
    #     self._compute_image(1)
    #
    # def _compute_image(self, idx):
    #     region = self.region_array[idx]
    #     low1 = region.low[1]
    #     high1 = region.high[1]
    #
    #     if region.is_leaf:
    #         angle = (region.low[0], low1)
    #         width = region.high[0] - region.low[0]
    #         height = high1 - low1
    #         self.patches.append({'angle': angle,
    #                              'width': width,
    #                              'height': height,
    #                              # 'max_cp': self.max_CP,
    #                              # 'min_cp': self.min_CP,
    #                              'cp': region.queue.CP,
    #                              'freq': region.freq
    #                              })
    #     else:
    #         if region.dim_split == 0:
    #             line1_xs = 2 * [region.val_split]
    #             line1_ys = [low1, high1]
    #             self.lines.append({'xdata': line1_xs,
    #                                'ydata': line1_ys})
    #         elif region.dim_split == 1:
    #             line1_ys = 2 * [region.val_split]
    #             line1_xs = [region.low[0], region.high[0]]
    #             self.lines.append({'xdata': line1_xs,
    #                            'ydata': line1_ys})
    #
    #         self._compute_image(2 * idx)
    #         self._compute_image(2 * idx + 1)
    #
    # def init_display(self):
    #     self.figure = plt.figure()
    #     self.ax = plt.axes()
    #     self.ax.set_xlim(self.root.low[0], self.root.high[0])
    #     self.ax.set_ylim(self.root.low[1], self.root.high[1])
    #     plt.ion()
    #     plt.show()
    #
    # def plot_image(self):
    #     self.ax.lines.clear()
    #     self.ax.collections.clear()
    #     colors = []
    #     patch_list = []
    #     for line_dict in self.lines:
    #         self.ax.add_line(lines.Line2D(xdata=line_dict['xdata'],
    #                                       ydata=line_dict['ydata'],
    #                                       linewidth=2,
    #                                       color='blue'))
    #     for patch_dict in self.patches:
    #         colors.append(patch_dict['cp'])
    #         patch_list.append(patches.Rectangle(xy=patch_dict['angle'],
    #                                             width=patch_dict['width'],
    #                                             height=patch_dict['height'],
    #                                             fill=True,
    #                                             edgecolor=None,
    #                                             alpha=0.8))
    #     p = PatchCollection(patch_list)
    #     p.set_array(np.array(colors))
    #     self.ax.add_collection(p)
    #     self.cb = self.figure.colorbar(p, ax=self.ax)
    #     plt.draw()
    #     plt.pause(0.001)
    #     self.cb.remove()

    @property
    def list_CP(self):
        return [float("{0:.3f}".format(region.queue.CP)) for region in self.region_array]

    @property
    def list_comp(self):
        return [float("{0:.3f}".format(region.queue.competence)) for region in self.region_array]

    @property
    def region_freq(self):
        return [region.freq for region in self.region_array]

    @property
    def root(self):
        return self.region_array[1]
    #
    # @property
    # def max_CP(self):
    #     return self.root.max_CP
    #
    # @property
    # def min_CP(self):
    #     return self.root.min_CP

    @property
    def sum_CP(self):
        return self.root.sum_CP


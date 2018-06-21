from samplers.treeSampler import TreeSampler
import matplotlib.lines as lines
import matplotlib.patches as patches


import matplotlib.pyplot as plt
Blues = plt.get_cmap('Blues')

class TreePlotter(TreeSampler):
    def __init__(self, space, R=128, auto=True, theta=1):
        self.init_display()
        super(TreePlotter, self).__init__(space, R, auto, theta)
        self.ax.set_xlim(self.root.low[0], self.root.high[0])
        self.ax.set_ylim(self.root.low[1], self.root.high[1])
        self.colors = {0: 'black', 1: 'red'}

    def _divide(self, idx, n, dim_idx):
        if n > 1:
            dim = self.dims[dim_idx]
            region = self.region_array[idx]
            low = region.low[dim]
            high = region.high[dim]
            val_split = (high + low) / 2
            self.region_array[2 * idx], self.region_array[2 * idx + 1], _ = region.split(dim, val_split)
            region.dim_split = dim
            region.val_split = val_split
            self.n_leaves += 1
            region.compute_line()
            self.lines.append(region.line)
            self.ax.add_line(lines.Line2D(xdata=region.line[0],
                                          ydata=region.line[1],
                                          linewidth=2,
                                          color='blue'))
            next_dim_idx = (dim_idx + 1) % (len(self.dims))
            self._divide(2 * idx, n / 2, next_dim_idx)
            self._divide(2 * idx + 1, n / 2, next_dim_idx)
            self.update_display()

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
                    self.ax.add_line(lines.Line2D(xdata=region.line[0],
                                          ydata=region.line[1],
                                          linewidth=1,
                                          color='blue'))
        self.update_CP_tree()
        self.update_display()

    def update_display(self):
        if self.n_points % 10 == 0:

            self.ax.scatter([point[0][0] for point in self.points],
                            [point[0][1] for point in self.points],
                            color=[self.colors[point[2]] for point in self.points],
                            s=0.3)

            region = self.region_array[self.max_CP_leaf]
            self.ax.patches.clear()
            self.ax.add_patch(patches.Rectangle(xy=(region.low[0], region.low[1]),
                                                width=region.high[0] - region.low[0],
                                                height=region.high[1] - region.low[1],
                                                fill=True,
                                                edgecolor=None,
                                                alpha=0.2,
                                                color='blue'))

            plt.draw()
            plt.pause(0.001)

    def init_display(self):
        self.figure = plt.figure()
        self.ax = plt.axes()
        # self.scatter = self.ax.scatter([],[])
        plt.ion()
        plt.show()
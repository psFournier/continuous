from regions.regionTree import RegionTree
import numpy as np
import matplotlib.lines as lines


import matplotlib.pyplot as plt
Blues = plt.get_cmap('Blues')

class RegionTreePlot(RegionTree):
    def __init__(self, space, nRegions, auto, beta, window):
        super(RegionTreePlot, self).__init__(space, nRegions, auto, beta, window)
        self.init_display()

    def append(self, point):
        target = point[0]
        success = point[1]
        regions_idx = self.find_regions(target)
        for idx in regions_idx:
            region = self.region_array[idx]
            region.add((target, success))
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
                                          linewidth=2,
                                          color='blue')  )
        self.update_CP_tree()
        self.update_display()

    def update_display(self):
        if self.n_points % 10 == 0:
            plt.draw()
            plt.pause(0.001)

    def init_display(self):
        self.figure = plt.figure()
        self.ax = plt.axes()
        self.ax.set_xlim(self.root.low[0], self.root.high[0])
        self.ax.set_ylim(self.root.low[1], self.root.high[1])
        plt.ion()
        plt.show()
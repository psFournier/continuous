from samplers.competenceQueue import CompetenceQueue
import math
import numpy as np

class ListSampler():
    def __init__(self, space, theta):
        self.theta = theta
        self.tasks = space
        self.task_queues = [CompetenceQueue() for _ in self.tasks]
        self.task_freqs = [0] * len(self.tasks)

    def sample(self):
        CPs = [math.pow(queue.CP, self.theta) for queue in self.task_queues]
        sum = np.sum(CPs)
        mass = np.random.random() * sum
        idx = 0
        s = CPs[0]
        while mass > s:
            idx += 1
            s += CPs[idx]
        self.task_freqs[idx] += 1
        return self.tasks[idx]

    def append(self, point):
        self.task_queues[self.tasks.index(point[0])].append(point)

    def stats(self):
        stats = {}

        for i, queue in enumerate(self.task_queues):
            stats['CP_{}'.format(i)] = float("{0:.3f}".format(queue.CP))
            stats['comp_{}'.format(i)] = float("{0:.3f}".format(queue.competence))
            stats['freq_{}'.format(i)] = float("{0:.3f}".format(self.task_freqs[i]))

        return stats

    @property
    def max_CP(self):
        return max([q.CP for q in self.task_queues])
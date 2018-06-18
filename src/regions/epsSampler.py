from regions.competenceQueue import CompetenceQueue
import math
import numpy as np

class EpsSampler():
    def __init__(self, space, theta):
        self.theta = theta
        self.epsilons = space
        self.epsilon_queues = [CompetenceQueue() for _ in self.epsilons]
        self.epsilon_freq = [0] * len(self.epsilons)

    def sample(self):
        CPs = [math.pow(queue.CP, self.theta) for queue in self.epsilon_queues]
        sum = np.sum(CPs)
        mass = np.random.random() * sum
        idx = 0
        s = CPs[0]
        while mass > s:
            idx += 1
            s += CPs[idx]
        self.epsilon_freq[idx] += 1
        return self.epsilons[idx]

    def append(self, point):
        self.epsilon_queues[self.epsilons.index(point[0])].append(point)

    def stats(self):
        stats = {}
        stats['list_CP'] = self.list_CP
        stats['list_comp'] = self.list_comp
        stats['list_freq'] = self.epsilon_freq
        return stats

    @property
    def list_CP(self):
        return [float("{0:.3f}".format(self.epsilon_queues[idx].CP)) for idx in range(len(self.epsilons))]

    @property
    def list_comp(self):
        return [float("{0:.3f}".format(self.epsilon_queues[idx].competence)) for idx in range(len(self.epsilons))]

    @property
    def list_freq(self):
        return self.epsilon_freq
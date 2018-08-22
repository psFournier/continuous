from gym import Wrapper
import numpy as np
import math
from samplers.competenceQueue import CompetenceQueue

# Wrappers override step, reset functions, as well as the defintion of action, observation and goal spaces.

class CPBased(Wrapper):
    def __init__(self, env, args):
        super(CPBased, self).__init__(env)
        self.goals = []
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.theta = float(args['theta'])
        self.steps = []
        self.interests = []
        self.goal = None

    def init(self):
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.steps = [0 for _ in self.goals]
        self.interests = [0 for _ in self.goals]

    def get_idx(self):
        CPs = [abs(q.CP) for q in self.queues]
        maxCP = max(CPs)
        minCP = min(CPs)

        Rs = [q.R for q in self.queues]
        maxR = max(Rs)
        minR = min(Rs)

        try:
            if maxCP > 1:
                self.interests = [(cp - minCP) / (maxCP - minCP) for cp in CPs]
            else:
                self.interests = [1 - (r - minR) / (maxR - minR) for r in Rs]
        except:
            self.interests = [1 for _ in self.queues]

        weighted_interests = [math.pow(I, self.theta) + 0.1 for I in self.interests]
        sum = np.sum(weighted_interests)
        mass = np.random.random() * sum
        idx = 0
        s = weighted_interests[0]
        while mass > s:
            idx += 1
            s += weighted_interests[idx]
        return idx

    def get_stats(self):
        stats = {}
        for i, goal in enumerate(self.goals):
            stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].R))
            stats['CP_{}'.format(goal)] = float("{0:.3f}".format(abs(self.queues[i].CP)))
            stats['I_{}'.format(goal)] = float("{0:.3f}".format(self.interests[i]))
            stats['S_{}'.format(goal)] = float("{0:.3f}".format(self.steps[i]))
        return stats

    # @property
    # def min_avg_length_ep(self):
    #     return np.min([q.L_mean for q in self.queues if q.L_mean != 0])
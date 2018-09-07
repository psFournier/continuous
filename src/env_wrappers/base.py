from gym import Wrapper
import numpy as np
import math
from samplers.competenceQueue import CompetenceQueue
from utils.linearSchedule import LinearSchedule
from abc import ABCMeta, abstractmethod
# Wrappers override step, reset functions, as well as the defintion of action, observation and goal spaces.

class CPBased(Wrapper):
    def __init__(self, env, args):
        super(CPBased, self).__init__(env)
        self.theta = float(args['--theta'])
        self.gamma = float(args['--gamma'])
        self.shaping = args['--shaping'] != '0'
        self.opt_init = int(args['--opt_init'])
        self.goals = []
        self.goal = None

    def init(self):
        self.queue = CompetenceQueue()
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.steps = [0 for _ in self.goals]
        self.interests = [0 for _ in self.goals]
        self.dones = [0 for _ in self.goals]
        self.attempts = [0 for _ in self.goals]
        self.mincp = min([q.mincp for q in self.queues])

    # def explor_temp(self, t):
    #     T = self.queues[self.goal].T
    #     return self.explorations[self.goal].value(t, T)

    def explor_eps(self):
        step = self.steps[self.goal]
        return 1 + min(float(step) / 1e4, 1) * (0.1 - 1)

    def processEp(self, R, S, T):
        self.queues[self.goal].append({'R': R, 'S': S, 'T': T})
        self.queue.append({'R': R, 'S': S, 'T': T})

    def step(self, exp):
        self.steps[self.goal] += 1
        exp['state1'] = self.env.step(exp['action'])
        exp['goal'] = self.goal
        exp = self.eval_exp(exp)
        if exp['terminal']:
            self.dones[self.goal] += 1
        return exp

    def is_term(self, exp):
        return False

    def eval_exp(self, exp):
        term = self.is_term(exp)
        if term:
            r = 1
        else:
            r = 0
        r = self.shape(r, term)
        exp['reward'] = r
        exp['terminal'] = term
        return exp

    def shape(self, r, term):
        if self.opt_init == 1:
            r += (self.gamma - 1)
            if term:
                r -= self.gamma
        elif self.opt_init == 2:
            r += (self.gamma - 1)
        elif self.opt_init == 3:
            r -= 1
        return r

    def unshape(self, r, term):
        if self.opt_init == 1:
            r -= (self.gamma - 1)
            if term:
                r += self.gamma
        elif self.opt_init == 2:
            r -= (self.gamma - 1)
        elif self.opt_init == 3:
            r += 1
        return r

    def get_idx(self):
        CPs = [abs(q.CP) for q in self.queues]
        Rs = [q.R for q in self.queues]
        maxR = max(Rs)
        minR = min(Rs)
        maxCP = max(CPs)
        minCP = min(CPs)

        if maxCP > self.mincp:
            self.interests = [(cp - minCP) / (maxCP - minCP + 0.001) for cp in CPs]
        else:
            self.interests = [1 - (r - minR) / (maxR - minR + 0.001) for r in Rs]

        # self.interests = [CP * (1 - (R - minR) / (maxR - minR + 0.001)) for CP, R in zip(CPs, Rs)]

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
            stats['S_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].S))
            stats['T_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].T))
            stats['done_{}'.format(goal)] = float("{0:.3f}".format(self.dones[i]))
            stats['step_{}'.format(goal)] = float("{0:.3f}".format(self.steps[i]))
            stats['attempt_{}'.format(goal)] = float("{0:.3f}".format(self.attempts[i]))
        stats['R'] = float("{0:.3f}".format(self.queue.R))
        stats['CP'] = float("{0:.3f}".format(abs(self.queue.CP)))
        stats['S'] = float("{0:.3f}".format(self.queue.S))
        stats['T'] = float("{0:.3f}".format(self.queue.T))
        return stats



    # @property
    # def min_avg_length_ep(self):
    #     return np.min([q.L_mean for q in self.queues if q.L_mean != 0])
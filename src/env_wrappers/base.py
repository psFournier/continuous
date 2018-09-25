from gym import Wrapper
import numpy as np
import math
from samplers.competenceQueue import CompetenceQueue
from utils.linearSchedule import LinearSchedule
from abc import ABCMeta, abstractmethod
# Wrappers override step, reset functions, as well as the defintion of action, observation and goal spaces.

class Base(Wrapper):
    def __init__(self, env, args):
        super(Base, self).__init__(env)
        self.args = args
        self.minR = -1
        self.maxR = 0
        self.init()

    def init(self):
        pass

    def step(self, exp):
        exp['state1'], exp['reward'], exp['terminal'], _ = self.env.step(exp['action'])
        return exp

    def processEp(self, R, S, T):
        pass

    def get_stats(self):
        stats = {}
        return stats

    @property
    def state_dim(self):
        return self.env.observation_space.shape

    @property
    def action_dim(self):
        return self.env.action_space.shape

class CPBased(Wrapper):
    def __init__(self, env, args):
        super(CPBased, self).__init__(env)
        self.theta = float(args['--theta'])
        self.gamma = float(args['--gamma'])
        self.shaping = args['--shaping'] != '0'
        self.opt_init = float(args['--opt_init'])
        self.goals = []
        self.goal = None
        self.minR = self.shape(-1, False)
        self.maxR = self.shape(0, True)

    def init(self):
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.steps = [0 for _ in self.goals]
        self.replays = [0 for _ in self.goals]

    def processEp(self, R, S, T):
        self.queues[self.goal].append({'R': R, 'S': S, 'T': T})

    def step(self, exp):
        self.steps[self.goal] += 1
        exp['state1'] = self.env.step(exp['action'])
        exp['goal'] = self.goal
        exp = self.eval_exp(exp)
        return exp

    def is_term(self, exp):
        return False

    def eval_exp(self, exp):
        term = self.is_term(exp)
        if term:
            r = self.maxR
        else:
            r = self.minR
        exp['reward'] = r
        exp['terminal'] = term
        return exp

    def shape(self, r, term):
        b = (self.gamma - 1) * self.opt_init
        r += b
        if term:
            c = -self.gamma * self.opt_init
            r += c
        return r

    def unshape(self, r, term):
        b = (self.gamma - 1) * self.opt_init
        r -= b
        if term:
            c = -self.gamma * self.opt_init
            r -= c
        return r

    def get_idx(self):

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
            stats['step_{}'.format(goal)] = float("{0:.3f}".format(self.steps[i]))
            stats['replay_{}'.format(goal)] = float("{0:.3f}".format(self.replays[i]))
            stats['I_{}'.format(goal)] = float("{0:.3f}".format(self.interests[i]))

        return stats

    @property
    def interests(self):

        CPs = [abs(q.CP[-1]) if q.CP else 0 for q in self.queues]
        Ts = [q.T[-1] if q.T else 0 for q in self.queues]
        # interests = [cp * (1 - t) for cp, t in zip(CPs, Ts)]
        maxT = max(Ts)
        minT = min(Ts)
        maxCP = max(CPs)
        minCP = min(CPs)

        if maxCP - minCP > 5:
            interests = [(cp - minCP) / (maxCP - minCP) for cp in CPs]
        else:
            interests = [1 - t for t in Ts]

        return interests
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
        self.gamma = float(args['--gamma'])
        self.opt_init = float(args['--opt_init'])
        self.goals = [None]
        self.minR = 0
        self.maxR = 1
        self.init()

    def init(self):
        self.queue = CompetenceQueue()

    def reset(self, goal=None):
        state = self.env.reset()
        return state

    def step(self, exp):
        exp['s1'], exp['r'], exp['t'], _ = self.env.step(exp['a'])
        exp['t'] = False
        exp['r'] = self.shape(exp['r'], exp['t'])
        return exp

    def processEp(self, R, S, T):
        pass

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

    def get_stats(self):

        stats = {}
        stats['agentR'] = float("{0:.3f}".format(self.queue.R[-1]))
        stats['agentT'] = float("{0:.3f}".format(self.queue.T[-1]))
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
        self.minR = self.shape(0, False)
        self.maxR = self.shape(1, False)

    def init(self):
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.steps = [0 for _ in self.goals]
        self.replays = [0 for _ in self.goals]

    def step(self, exp):
        self.steps[self.goal] += 1
        exp['s1'] = self.env.step(exp['a'])
        exp['g'] = self.goal
        exp = self.eval_exp(exp)
        return exp

    def is_term(self, exp):
        return False

    def eval_exp(self, exp):
        if self.is_term(exp):
            exp['r'] = self.maxR
        else:
            exp['r'] = self.minR
        exp['t'] = False
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

        weighted_interests = [math.pow(I, self.theta) for I in self.interests]
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
            stats['CP_{}'.format(goal)] = float("{0:.3f}".format(self.CPs[i]))
            stats['agentR_{}'.format(goal)] = float("{0:.3f}".format(self.Rs[i]))
        return stats

    @property
    def interests(self):

        maxCP = max(self.CPs)
        minCP = min(self.CPs)
        maxR = max(self.Rs)
        minR = min(self.Rs)
        if maxCP - minCP > 5:
            interests = [(cp - minCP) / (maxCP - minCP) for cp in self.CPs]
        else:
            interests = [1 - (r - minR) / (maxR - minR + 0.0001) for r in self.Rs]

        return interests

    @property
    def CPs(self):
        return [abs(q.CP[-1]) if q.CP else 0 for q in self.queues]

    @property
    def Rs(self):
        return [q.R[-1] for q in self.queues]


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
        self.gamma = float(args['--gamma'])
        self.opt_init = float(args['--opt_init'])
        self.minR = self.shape(0, False)
        self.maxR = self.shape(1, False)
        self.init()

    def init(self):
        self.queue = CompetenceQueue()

    def step(self, exp):
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def is_term(self, exp):
        pass

    def end_episode(self, trajectory):
        R = np.sum([self.unshape(exp['r'], exp['t']) for exp in trajectory])
        self.queue.append(R)
        augmented_ep = []
        for i, expe in enumerate(reversed(trajectory)):
            augmented_ep.append(expe.copy())
        return augmented_ep

    def eval_exp(self, exp):
        if self.is_term(exp):
            exp['r'] = self.maxR
        else:
            exp['r'] = self.minR
        exp['t'] = False
        return exp

    def reset(self, goal=None):
        state = self.env.reset()
        return state

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
        return stats

    @property
    def state_dim(self):
        return self.env.observation_space.shape

    @property
    def action_dim(self):
        return self.env.action_space.shape

class RndBased(Base):
    def __init__(self, env, args, low, high):
        super(RndBased, self).__init__(env, args)
        self.goal = None
        self.low, self.high = low, high

    def step(self, exp):
        exp['s1'] = self.env.step(exp['a'])[0]
        exp['g'] = self.goal
        exp = self.eval_exp(exp)
        return exp

    def reset(self, idx=None, goal=None):
        if goal is None:
            self.goal = np.random.uniform(self.low, self.high)
        else:
            self.goal = goal
        state = self.env.reset()
        return state

class CPBased(Base):
    def __init__(self, env, args, goals):
        self.goals = goals
        super(CPBased, self).__init__(env, args)
        self.theta = float(args['--theta'])
        self.goal = None
        self.idx = None

    def init(self):
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.steps = [0 for _ in self.goals]
        self.replays = [0 for _ in self.goals]

    def step(self, exp):
        self.steps[self.idx] += 1
        exp['s1'] = self.env.step(exp['a'])[0]
        exp['g'] = self.goal
        exp = self.eval_exp(exp)
        return exp

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

    def reset(self, idx=None, goal=None):

        if idx is None:
            if goal is None:
                self.idx = self.get_idx()
                self.goal = np.array(self.goals[self.idx])
            else:
                self.idx = 0
                self.goal = goal
        else:
            self.idx = idx
            self.goal = np.array(self.goals[self.idx])


        state = self.env.reset()
        return state

    def get_stats(self):
        stats = {}
        for i, goal in enumerate(self.goals):
            stats['step_{}'.format(goal)] = float("{0:.3f}".format(self.steps[i]))
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


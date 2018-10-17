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
        self.init()

    def init(self):
        self.queue = CompetenceQueue()

    def step(self, exp):
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def end_episode(self, trajectory):
        R = 0
        for exp in reversed(trajectory):
            R = R * self.gamma + exp['r']
        self.queue.append(R)

    def eval_exp(self, exp):
        pass

    def reset(self):
        state = self.env.reset()
        return state

    # def shape(self, r, term):
    #     b = (self.gamma - 1) * self.opt_init
    #     r += b
    #     if term:
    #         c = -self.gamma * self.opt_init
    #         r += c
    #     return r
    #
    # def unshape(self, r, term):
    #     b = (self.gamma - 1) * self.opt_init
    #     r -= b
    #     if term:
    #         c = -self.gamma * self.opt_init
    #         r -= c
    #     return r

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

    def reset(self):
        self.goal = np.random.uniform(self.low, self.high)
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

    def sample_task(self):
        sum = np.sum(self.interests)
        mass = np.random.random() * sum
        idx = 0
        s = self.interests[0]
        while mass > s:
            idx += 1
            s += self.interests[idx]
        return idx

    def end_episode(self, trajectory):
        R = 0
        for exp in reversed(trajectory):
            R = R * self.gamma + exp['r']
        self.queues[self.idx].append(R)

    def reset(self):
        self.idx = self.sample_task()
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
            stats['agentT_{}'.format(goal)] = float("{0:.3f}".format(self.Ts[i]))
        return stats

    @property
    def interests(self):
        minCP = min(self.CPs)
        maxCP = max(self.CPs)
        CPs = [math.pow((cp - minCP) / (maxCP - minCP + 0.0001), self.theta) for cp in self.CPs]
        sumCP = np.sum(CPs)
        Ntasks = len(self.CPs)
        espilon = 0.4
        interests = [espilon / Ntasks + (1 - espilon) * cp / (sumCP + 0.0001) for cp in CPs]
        return interests

    @property
    def CPs(self):
        return [abs(q.CP[-1]) if q.CP else 0 for q in self.queues]

    @property
    def Rs(self):
        return [q.MCR[-1] for q in self.queues]

    @property
    def Ts(self):
        return [q.T[-1] for q in self.queues]


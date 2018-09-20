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
        self.queue = CompetenceQueue()

    def step(self, exp):
        exp['state1'], exp['reward'], exp['terminal'], _ = self.env.step(exp['action'])
        return exp

    def processEp(self, R, S, T):
        self.queue.append({'R': R, 'S': S, 'T': T})

    def get_stats(self):
        stats = {}
        self.queue.update()
        if self.queue.R:
            stats['R'] = float("{0:.3f}".format(self.queue.R[-1]))
            stats['S'] = float("{0:.3f}".format(self.queue.S[-1]))
            stats['T'] = float("{0:.3f}".format(self.queue.T[-1]))
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
        self.opt_init = int(args['--opt_init'])
        self.goals = []
        self.goal = None
        self.minR = self.shape(-1, False)
        self.maxR = self.shape(0, True)

    def init(self):
        self.queue = CompetenceQueue()
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.steps = [0 for _ in self.goals]
        self.interests = [0 for _ in self.goals]
        self.dones = [0 for _ in self.goals]
        self.attempts = [0 for _ in self.goals]
        self.mincp = min([q.mincp for q in self.queues])

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
            r = self.maxR
        else:
            r = self.minR
        # r = self.shape(r, term)
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
        if all(q.CP for q in self.queues):
            CPs = [abs(q.CP[-1]) for q in self.queues]
            Rs = [q.R[-1] for q in self.queues]
            maxR = max(Rs)
            minR = min(Rs)
            maxCP = max(CPs)
            minCP = min(CPs)

            if maxCP - minCP > 5:
                self.interests = [(cp - minCP) / (maxCP - minCP) for cp in CPs]
            else:
                self.interests = [1 - (r - minR) / (maxR - minR + 0.001) for r in Rs]

            weighted_interests = [math.pow(I, self.theta) + 0.1 for I in self.interests]
            sum = np.sum(weighted_interests)
            mass = np.random.random() * sum
            idx = 0
            s = weighted_interests[0]
            while mass > s:
                idx += 1
                s += weighted_interests[idx]
        else:
            idx = np.random.choice(len(self.queues))
        return idx

    def get_stats(self):
        stats = {}

        for i, goal in enumerate(self.goals):
            self.queues[i].update()
            if self.queues[i].R:
                stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].R[-1]))
                stats['S_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].S[-1]))
                stats['T_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].T[-1]))
            if self.queues[i].CP:
                stats['CP_{}'.format(goal)] = float("{0:.3f}".format(abs(self.queues[i].CP[-1])))

            stats['done_{}'.format(goal)] = float("{0:.3f}".format(self.dones[i]))
            stats['step_{}'.format(goal)] = float("{0:.3f}".format(self.steps[i]))
            stats['attempt_{}'.format(goal)] = float("{0:.3f}".format(self.attempts[i]))
            stats['I_{}'.format(goal)] = float("{0:.3f}".format(self.interests[i]))

        self.queue.update()
        if self.queue.R:
            stats['R'] = float("{0:.3f}".format(self.queue.R[-1]))
            stats['S'] = float("{0:.3f}".format(self.queue.S[-1]))
            stats['T'] = float("{0:.3f}".format(self.queue.T[-1]))

        return stats


    # @property
    # def min_avg_length_ep(self):
    #     return np.min([q.L_mean for q in self.queues if q.L_mean != 0])
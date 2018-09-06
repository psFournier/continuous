from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class PlayroomG(Wrapper):
    def __init__(self, env, args):
        super(PlayroomG, self).__init__(env)
        self.state_low = self.env.state_low
        self.state_high = self.env.state_high
        self.init_state = self.env.state_init
        self.opt_init = int(args['--opt_init'])
        self.gamma = float(args['--gamma'])
        self.queue = CompetenceQueue()
        self.steps = [0]

    def processEp(self, episode):
        T = int(episode[-1]['terminal'])
        R = np.sum([exp['reward'] for exp in episode])
        S = len(episode)
        self.queue.append({'R': R, 'S': S, 'T': T})

    # def explor_temp(self, t):
    #     return 10 + max(float(t)/10000, 1)*(0.5 - 10)

    def explor_eps(self):
        step = self.steps[0]
        return 1 + min(float(step) / 5e4, 1) * (0.1 - 1)

    def step(self, exp):
        self.steps[0] += 1
        exp['state1'] = self.env.step(exp['action'])
        exp['goal'] = self.goal
        exp = self.eval_exp(exp)
        return exp

    def eval_exp(self, exp):
        term = self.is_term(exp)
        if term:
            r = 1
        else:
            r = 0
        r = self.transform_r(r, term)
        exp['reward'] = r
        exp['terminal'] = term
        return exp

    def transform_r(self, r, term):
        if self.opt_init == 1:
            r += self.gamma - 1
            if term:
                r -= self.gamma
        elif self.opt_init == 2:
            r += self.gamma - 1
        elif self.opt_init == 3:
            r -= 1
        return r

    def is_term(self, exp):
        return ((exp['state1'] == self.goal).all() and (exp['state0'] != self.goal).any())

    def reset(self):
        self.goal = np.zeros(shape=self.goal_dim)
        while True:
            for idx in range(self.goal_dim[0]):
                self.goal[idx] = np.random.randint(self.state_low[idx], self.state_high[idx] + 1)
            if (self.goal != self.init_state).any():
                break
        state = self.env.reset()
        return state

    def get_stats(self):
        stats = {}
        stats['R'] = float("{0:.3f}".format(self.queue.R))
        stats['S'] = float("{0:.3f}".format(self.queue.S))
        stats['T'] = float("{0:.3f}".format(self.queue.T))
        return stats

    @property
    def state_dim(self):
        return 18,

    @property
    def goal_dim(self):
        return 18,

    @property
    def action_dim(self):
        return 11
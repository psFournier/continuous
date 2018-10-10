from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import RndBased

class PlayroomGM2(RndBased):
    def __init__(self, env, args):
        super(PlayroomGM2, self).__init__(env, args, None, None)
        self.mask = None
        self.init()
        self.obj_feat = [[i] for i in range(2, 11)]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.init)

    def step(self, exp):
        exp['g'] = self.goal
        exp['m'] = self.mask
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def eval_exp(self, exp):
        exp['r'] = np.sum([mi if exp['g'][i] == exp['s1'][i] else 0 for i, mi in enumerate(exp['m'])])
        exp['t'] = False
        exp['r'] = self.shape(exp['r'], exp['t'])
        return exp

    def reset(self, idx=None, goal=None):

        self.mask = np.random.randint(0,2,size=self.state_dim)
        self.goal = self.init_state.copy()
        for i, mi in enumerate(self.mask):
            if mi:
                self.goal[i] = np.random.randint(self.state_low[i], self.state_high[i] + 1)
        self.mask = self.mask / np.sum(self.mask)

        state = self.env.reset()
        return state

    def obj2mask(self, idx):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[idx]] = 1
        return res

    @property
    def state_dim(self):
        return 11,

    @property
    def goal_dim(self):
        return 11,

    @property
    def action_dim(self):
        return 7
from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from .base import CPBased

class Taxi2GM(CPBased):
    def __init__(self, env, args):
        super(Taxi2GM, self).__init__(env, args)
        self.goals = ['agent', 'passenger', 'taxi']
        self.init()
        self.obj_feat = [[0, 1], [2, 3], [4]]
        self.state_low = [0, 0, 0, 0, 0]
        self.state_high = [self.env.nR - 1, self.env.nC - 1, self.env.nR - 1, self.env.nC - 1, 1]
        self.init_state = [2, 2, 0, 0, 0]

        self.test_goals = [(np.array([0, 0, 0, 0, 1]), 2),
                           (np.array([0, 0, 0, 4, 0]), 1),
                           (np.array([0, 0, 4, 0, 0]), 1),
                           (np.array([0, 0, 4, 3, 0]), 1)]

    def eval_exp(self, exp):
        if self.posInit:
            r = -1
        else:
            r = 0
        term = False

        goal_feat = self.obj_feat[exp['goal']]
        goal_vals = exp['goalVals'][goal_feat]
        s1_proj = exp['state1'][goal_feat]
        s0_proj = exp['state0'][goal_feat]
        if ((s1_proj == goal_vals).all() and (s0_proj != goal_vals).any()):
            r += 1
            term = True
        return r, term

    def reset(self):
        self.goal = self.get_idx()
        features = self.obj_feat[self.goal]
        self.goalVals = self.init_state.copy()
        self.mask = self.obj2mask(self.goal)
        while True:
            for idx in features:
                self.goalVals[idx] = np.random.randint(self.state_low[idx], self.state_high[idx] + 1)
            if self.goalVals != self.init_state:
                self.goalVals = np.array(self.goalVals)
                break
        state = self.env.reset()
        return state

    def obj2mask(self, obj):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[obj]] = 1
        return res

    @property
    def state_dim(self):
        return 5,

    @property
    def goal_dim(self):
        return 5,

    @property
    def action_dim(self):
        return 6
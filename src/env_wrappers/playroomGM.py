from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class PlayroomGM(CPBased):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env, args)
        self.goals = [obj.name for obj in self.env.objects]
        self.goalVals = None
        self.mask = None
        self.init()
        self.obj_feat = [[4 + 4 * j] for j in range(len(self.goals))]
        self.state_low = self.env.state_low
        self.state_high = self.env.state_high
        self.init_state = self.env.state_init

    def is_term(self, exp):
        goal_feat = self.obj_feat[exp['goal']]
        goal_vals = exp['goalVals'][goal_feat]
        s1_proj = exp['state1'][goal_feat]
        s0_proj = exp['state0'][goal_feat]
        return ((s1_proj == goal_vals).all() and (s0_proj != goal_vals).any())

    def reset(self):
        self.goal = self.get_idx()
        features = self.obj_feat[self.goal]
        self.goalVals = np.zeros(shape=self.state_dim)
        self.mask = np.zeros(shape=self.state_dim)
        for idx in features:
            self.mask[idx] = 1
            while True:
                s = np.random.randint(self.state_low[idx], self.state_high[idx] + 1)
                if s != self.init_state[idx]: break
            self.goalVals[idx] = s
        state = self.env.reset()
        return state

    def obj2mask(self, obj):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[obj]] = 1
        return res

    @property
    def state_dim(self):
        return 18,

    @property
    def goal_dim(self):
        return 18,

    @property
    def action_dim(self):
        return 11
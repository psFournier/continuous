from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class PlayroomGM(CPBased):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env, args)
        self.goals = ['agent'] + [obj.name for obj in self.env.objects]
        self.goalVals = None
        self.init()
        self.obj_feat = [[0, 1]] + [[4*j + i + 2 for i in range(4)] for j in range(len(self.env.objects))]
        self.state_low = self.env.state_low
        self.state_high = self.env.state_high
        self.init_state = self.env.state_init

    def step(self, exp):
        self.steps[self.goal] += 1
        exp['goal'] = self.goal
        exp['goalVals'] = self.goalVals
        exp['state1'] = self.env.step(exp['action'])
        exp = self.eval_exp(exp)
        return exp

    def is_term(self, exp):
        goal_feat = self.obj_feat[exp['goal']]
        goal_vals = exp['goalVals'][goal_feat]
        s1_proj = exp['state1'][goal_feat]
        s0_proj = exp['state0'][goal_feat]
        return ((s1_proj == goal_vals).all() and (s0_proj != goal_vals).any())

    def reset(self):
        self.goal = self.get_idx()
        features = self.obj_feat[self.goal]
        self.goalVals = np.array(self.init_state)
        self.mask = np.zeros(shape=self.state_dim)
        while True:
            for idx in features:
                self.mask[idx] = 1
                self.goalVals[idx] = np.random.randint(self.state_low[idx], self.state_high[idx] + 1)
            if (self.goalVals != self.init_state).any():
                break
        state = self.env.reset()
        return state

    @property
    def state_dim(self):
        return 18,

    @property
    def goal_dim(self):
        return 18,

    @property
    def action_dim(self):
        return 11
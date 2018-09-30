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
        self.object = None
        self.mask = None
        self.init()
        self.obj_feat = [[i] for i in range(2, 11)]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.init)

    def step(self, exp):
        self.steps[self.object] += 1
        exp['g'] = self.goal
        exp['m'] = self.mask
        exp['s1'] = self.env.step(exp['a'])
        exp = self.eval_exp(exp)
        return exp

    def is_term(self, exp):
        indices = np.where(exp['m'])
        goal = exp['g'][indices]
        s1_proj = exp['s1'][indices]
        return (s1_proj == goal).all()

    def reset(self, goal=None):

        if goal is None:
            self.object = self.get_idx()
        else:
            self.object = goal

        features = self.obj_feat[self.object]
        self.goal = self.init_state.copy()
        self.mask = self.obj2mask(self.object)

        while True:
            for idx in features:
                self.goal[idx] = np.random.randint(self.state_low[idx], self.state_high[idx] + 1)
            if (self.goal != self.init_state).any():
                break

        state = self.env.reset()
        return state

    def obj2mask(self, obj):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[obj]] = 1
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
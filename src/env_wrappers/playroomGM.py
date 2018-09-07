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
        self.object = None
        self.mask = None
        self.init()
        self.obj_feat = [[0, 1]] + [[4*j + i + 2 for i in range(4)] for j in range(len(self.env.objects))]
        self.state_low = self.env.state_low
        self.state_high = self.env.state_high
        self.init_state = self.env.state_init

    def step(self, exp):
        self.steps[self.object] += 1
        exp['goal'] = self.goal
        exp['mask'] = self.mask
        exp['state1'] = self.env.step(exp['action'])
        exp = self.eval_exp(exp)
        if exp['terminal']:
            self.dones[self.object] += 1
        return exp

    def explor_eps(self):
        step = self.steps[self.object]
        return 1 + min(float(step) / 1e4, 1) * (0.1 - 1)

    def processEp(self, R, S, T):
        self.queues[self.object].append({'R': R, 'S': S, 'T': T})

    def is_term(self, exp):
        indices = np.where(exp['mask'])
        goal = exp['goal'][indices]
        s1_proj = exp['state1'][indices]
        s0_proj = exp['state0'][indices]
        return ((s1_proj == goal).all() and (s0_proj != goal).any())

    def reset(self):
        self.object = self.get_idx()
        self.attempts[self.object] += 1
        features = self.obj_feat[self.object]
        self.goal = np.array(self.init_state)
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
        return 18,

    @property
    def goal_dim(self):
        return 18,

    @property
    def action_dim(self):
        return 11
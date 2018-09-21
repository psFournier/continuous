from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class PlayroomGM(CPBased):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env, args)
        self.goals = ['xy'] + [obj.name for obj in self.env.objects]
        self.object = None
        self.mask = None
        self.init()
        self.obj_feat = [[0, 1]] + [[i] for i in range(2, 14)]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.init)

    def step(self, exp):
        self.steps[self.object] += 1
        exp['goal'] = self.goal
        exp['mask'] = self.mask
        exp['state1'] = self.env.step(exp['action'])
        exp = self.eval_exp(exp)
        # if exp['terminal']:
            # self.dones[self.object] += 1
        return exp

    def processEp(self, R, S, T):
        self.queues[self.object].append({'R': R, 'S': S, 'T': T})
        self.queue.append({'R': R, 'S': S, 'T': T})

    def is_term(self, exp):
        indices = np.where(exp['mask'])
        goal = exp['goal'][indices]
        s1_proj = exp['state1'][indices]
        s0_proj = exp['state0'][indices]
        return ((s1_proj == goal).all() and (s0_proj != goal).any())

    def reset(self):
        self.object = self.get_idx()
        # self.attempts[self.object] += 1
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
        return 14,

    @property
    def goal_dim(self):
        return 14,

    @property
    def action_dim(self):
        return 10
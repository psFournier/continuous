from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from .base import CPBased

class Taxi2GM(CPBased):
    def __init__(self, env, args):
        super(Taxi2GM, self).__init__(env, args)
        self.goals = ['agent', 'passenger', 'taxi']
        self.object = None
        self.mask = None
        self.init()
        self.obj_feat = [[0, 1], [2, 3], [4]]
        self.state_low = [0, 0, 0, 0, 0]
        self.state_high = [self.env.nR - 1, self.env.nC - 1, self.env.nR - 1, self.env.nC - 1, 1]
        self.init_state = [2, 2, 0, 0, 0]
        self.test_goals = [(np.array([0, 0, 0, 0, 1]), 2),
                           (np.array([0, 0, 0, 4, 0]), 1),
                           (np.array([0, 0, 4, 0, 0]), 1),
                           (np.array([0, 0, 4, 3, 0]), 1)]

    def step(self, exp):
        self.steps[self.object] += 1
        exp['goal'] = self.goal
        exp['mask'] = self.mask
        exp['state1'] = self.env.step(exp['action'])
        exp = self.eval_exp(exp)
        if exp['terminal']:
            self.dones[self.object] += 1
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
        return 5,

    @property
    def goal_dim(self):
        return 5,

    @property
    def action_dim(self):
        return 6
from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased


class Reacher_e(CPBased):
    def __init__(self, env, args):
        super(Reacher_e, self).__init__(env, args)

        self.goals = [[0.02], [0.03], [0.04], [0.05]]
        self.idx = None
        self.init()

    def is_term(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        term = d < exp['g']
        return term

    def step(self, exp):
        self.steps[self.idx] += 1
        exp['s1'] = self.env.step(exp['a'])[0]
        exp['g'] = self.goal
        exp = self.eval_exp(exp)
        return exp

    def reset(self, goal=None):

        if goal is None:
            self.idx = self.get_idx()
        else:
            self.idx = goal

        self.goal = np.array(self.goals[self.idx])

        state = self.env.reset()
        return state

    @property
    def state_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 2,

    @property
    def goal_dim(self):
        return 1,


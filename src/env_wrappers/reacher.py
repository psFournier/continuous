from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import Base


class ReacherWrap(Base):
    def __init__(self, env, args):
        super(ReacherWrap, self).__init__(env, args)
        self.init()
        self.minQ, self.maxQ = -np.inf, np.inf

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < 0.02:
            exp['r'] = 0
            exp['t'] = False
        else:
            exp['r'] = -1
            exp['t'] = False
        # exp['r'] += (- np.square(exp['a']).sum())
        return exp

    @property
    def state_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 2,

class ReacherWrapShaped(Base):
    def __init__(self, env, args):
        super(ReacherWrapShaped, self).__init__(env, args)
        self.init()
        self.minQ, self.maxQ = -np.inf, np.inf
        self.test_goals = [np.array([0.02])] * 10

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < 0.02:
            exp['r'] = 0
            exp['t'] = False
        else:
            exp['r'] = -1
            exp['t'] = False
        # exp['r'] += (- np.square(exp['a']).sum())
        exp['r'] += (-self.gamma * np.linalg.norm(exp['s1'][[6, 7]]) + np.linalg.norm(exp['s0'][[6, 7]]))
        return exp

    @property
    def state_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 2,


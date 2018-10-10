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
        self.minQ = 0
        self.maxQ = 100

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < 0.04:
            exp['r'] = 1
        else:
            exp['r'] = 0
        exp['t'] = False
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
        self.minQ = -np.inf
        self.maxQ = np.inf

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < 0.04:
            exp['r'] = 1
        else:
            exp['r'] = 0

        exp['r'] += (-self.gamma * np.linalg.norm(exp['s1'][[6, 7]]) + np.linalg.norm(exp['s0'][[6, 7]]))
        exp['t'] = False
        return exp

    @property
    def state_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 2,


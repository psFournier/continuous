from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class Labyrinth2(CPBased):
    def __init__(self, env, args):
        super(Labyrinth2, self).__init__(env, args)
        self.goals = [0]
        self.goal = 0
        self.init()
        self.destinations = [np.array([0, 4]), np.array([0, 3]), np.array([4,0]), np.array([2,2])]

    def is_term(self, exp):
        return (exp['state1'] == np.array([0, 4])).all()

    def eval_exp(self, exp):
        term = self.is_term(exp)
        i = 0
        while i<4:
            if (exp['state1'] == self.destinations[i]).all():
                break
            i += 1
        r = 1 if i < 4 else 0
        r = self.transform_r(r, term)
        exp['reward'] = r
        exp['terminal'] = term
        return exp

    def reset(self):
        self.env.unwrapped.destrow = self.destinations[0][0]
        self.env.unwrapped.destcol = self.destinations[0][1]
        state = self.env.reset()
        return state

    def hindsight(self):
        return []

    @property
    def state_dim(self):
        return 2,

    @property
    def action_dim(self):
        return 4
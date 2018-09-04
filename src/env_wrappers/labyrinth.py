from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class Labyrinth(CPBased):
    def __init__(self, env, args):
        super(Labyrinth, self).__init__(env, args)
        self.goals = [0]
        self.goal = 0
        self.init()
        self.destination = np.array([0, 4])

    def is_term(self, exp):
        return (exp['state1'] == self.destination).all()

    def eval_exp(self, exp):
        exp = super(Labyrinth, self).eval_exp(exp)
        if self.shaping:
            dist0 = -np.linalg.norm(exp['state0'] - self.destination)
            dist1 = -np.linalg.norm(exp['state1'] - self.destination)
            shaping = self.gamma * dist1 - dist0
            exp['reward'] += shaping
        return exp

    def reset(self):
        self.env.unwrapped.destrow = self.destination[0]
        self.env.unwrapped.destcol = self.destination[1]
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
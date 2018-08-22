from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class LabyrinthG(CPBased):
    def __init__(self, env, args):
        super(LabyrinthG, self).__init__(env, args)
        self.goals = range(4)
        self.init()
        self.destination = np.array([0, 4])

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, exp):
        if self.posInit:
            r = -1
        else:
            r = 0
        term = False
        dist1 = np.linalg.norm(exp['state1'] - self.destination)
        if dist1 <= self.goals[self.goal]:
            r += 1
            term = True
        return r, term

    def reset(self):
        self.goal = self.get_idx()
        obs = self.env.reset()
        state = np.array(self.decode(obs))
        return state

    def decode(self, state):
        return list(self.env.decode(state))

    def encode(self, state):
        return self.env.encode(*state)

    def hindsight(self):
        return []

    @property
    def state_dim(self):
        return 2,

    @property
    def goal_dim(self):
        return 1,

    @property
    def action_dim(self):
        return 4
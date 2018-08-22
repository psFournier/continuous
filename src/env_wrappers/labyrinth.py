from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class Labyrinth(CPBased):
    def __init__(self, env, args):
        super(Labyrinth, self).__init__(env, args)
        self.gamma = 0.99
        self.goals = [0]
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

        if (exp['state1'] == self.destination).all():
            r += 1
            term = True

        if self.shaping:
            dist0 = -np.linalg.norm(exp['state0'] - self.destination)
            dist1 = -np.linalg.norm(exp['state1'] - self.destination)
            shaping = self.gamma * dist1 - dist0
            r += shaping

        return r, term

    def reset(self):

        self.env.unwrapped.destrow = self.destination[0]
        self.env.unwrapped.destcol = self.destination[1]

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
    def action_dim(self):
        return 4
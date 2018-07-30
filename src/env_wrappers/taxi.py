from gym import Wrapper
import numpy as np
import math
from samplers.competenceQueue import CompetenceQueue
import random as rnd

class Taxi(Wrapper):
    def __init__(self, env, args):
        super(Taxi, self).__init__(env)

        self.queue = CompetenceQueue()

        self.episode_exp = []
        self.buffer = None
        self.exploration_steps = 0

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, state0, action, state1):
        term = False
        r = 0
        vec = state1 - self.goal
        if (vec == 0).all():
            r = 1
            term = True
        return r, term

    def stats(self):
        stats = {}
        stats['comp'] = float("{0:.3f}".format(self.queue.competence))
        stats['cp'] = float("{0:.3f}".format(self.queue.CP))
        return stats

    def reset(self):

        if self.episode_exp:
            self.queue.append((self.goal, int(self.episode_exp[-1]['terminal'])))

        self.goal = np.array((0, 0, 4))

        obs = self.env.reset()
        state = np.array(self.decode(obs))

        for idx, buffer_item in enumerate(self.episode_exp):
            self.buffer.append(buffer_item)

        self.episode_exp.clear()

        return state

    def decode(self, state):
        return list(self.env.decode(state))

    def encode(self, state):
        return self.env.encode(*state)

    def hindsight(self):
        return []

    @property
    def state_dim(self):
        return 3,

    @property
    def action_dim(self):
        return [self.env.action_space.n]

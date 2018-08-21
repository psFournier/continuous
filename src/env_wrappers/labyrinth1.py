from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
from utils.linearSchedule import LinearSchedule

class Labyrinth1(Wrapper):
    def __init__(self, env, args):
        super(Labyrinth1, self).__init__(env)
        self.gamma = 0.99
        self.destination = np.array([0, 4])
        self.shaping = bool(args['shaping'])
        self.queues = [CompetenceQueue()]
        self.goals = [0]
        self.exploration = [LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1) for _ in self.goals]

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, exp):
        r = 0
        term = False

        if (exp['state1'] == self.destination).all():
            r = 1
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

    def get_stats(self):
        stats = {}
        for goal in self.goals:
            stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[goal].R_mean))
            stats['T_{}'.format(goal)] = float("{0:.3f}".format(self.queues[goal].T_mean))
            stats['L_{}'.format(goal)] = float("{0:.3f}".format(self.queues[goal].L_mean))
            stats['CP_{}'.format(goal)] = float("{0:.3f}".format(self.queues[goal].CP))
        return stats


    def decode(self, state):
        return list(self.env.decode(state))

    def encode(self, state):
        return self.env.encode(*state)

    def hindsight(self):
        return []

    @property
    def state_dim(self):
        return 2,
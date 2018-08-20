from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math

class Labyrinth2(Wrapper):
    def __init__(self, env, args):
        super(Labyrinth2, self).__init__(env)
        self.gamma = 0.99
        self.theta = float(args['theta'])
        self.goals = range(0, 9, 2)
        self.goal = None
        self.destination = np.array([0, 8])
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.interests = []
        self.steps = [0 for _ in self.goals]
        # self.freqs_act = [0 for _ in self.goals]
        # self.freqs_train = [0 for _ in self.goals]
        # self.freqs_act_reward = [0 for _ in self.goals]
        # self.freqs_train_reward = [0 for _ in self.goals]

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, exp):
        r = 0
        term = False
        dist1 = np.linalg.norm(exp['state1'] - self.destination)

        if dist1 <= self.goals[self.goal]:
            r = 1
            term = True

        return r, term

    def reset(self):

        CPs = [abs(q.CP) for q in self.queues]
        maxcp = max(CPs)

        if maxcp > 1:
            self.interests = [math.pow(cp / maxcp, self.theta) + 0.05 for cp in CPs]
        else:
            self.interests = [math.pow(1 - q.T_mean, self.theta) + 0.05 for q in self.queues]

        sum = np.sum(self.interests)
        mass = np.random.random() * sum
        idx = 0
        s = self.interests[0]
        while mass > s:
            idx += 1
            s += self.interests[idx]
        self.goal = idx

        # self.freqs_act[self.goal] += 1

        obs = self.env.reset()
        state = np.array(self.decode(obs))

        return state

    def get_stats(self):
        stats = {}
        for i, goal in enumerate(self.goals):
            stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].R_mean))
            stats['T_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].T_mean))
            stats['L_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].L_mean))
            stats['CP_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].CP))
            # stats['FA_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_act[i]))
            # stats['FT_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_train[i]))
            # stats['FAR_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_act_reward[i]))
            # stats['FTR_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_train_reward[i]))
            stats['I_{}'.format(goal)] = float("{0:.3f}".format(self.interests[i]))
            stats['S_{}'.format(goal)] = float("{0:.3f}".format(self.steps[i]))
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

    @property
    def goal_dim(self):
        return 1,

    # @property
    # def min_avg_length_ep(self):
    #     return np.min([q.L_mean for q in self.queues if q.L_mean != 0])
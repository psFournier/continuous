from gym import Wrapper
import numpy as np
import math
from samplers.competenceQueue import CompetenceQueue
import random as rnd

class TaxiGoal2(Wrapper):
    def __init__(self, env, args):
        super(TaxiGoal2, self).__init__(env)

        self.theta = args['theta']
        self.queues = [CompetenceQueue() for _ in range(125)]
        self.freqs = [0 for _ in range(125)]

        self.episode_exp = []
        self.buffer = None
        self.exploration_steps = 0
        self.her = args['her']

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, state0, action, state1, goal):
        term = False
        r = 0
        vec = state1 - goal
        if (vec == 0).all():
            r = 1
            term = True
        return r, term

    def sample_goal(self):
        CPs = [math.pow(queue.CP, self.theta) for queue in self.queues]
        sum = np.sum(CPs)
        mass = np.random.random() * sum
        idx = 0
        s = CPs[0]
        while mass > s:
            idx += 1
            s += CPs[idx]
        self.freqs[idx] += 1
        goal = np.array(self.decode(idx))
        return goal

    def stats(self):
        stats = {}
        CPmat = np.zeros(shape=(5,5,5))
        for i, queue in enumerate(self.queues):
            row, col, passidx = self.decode(i)
            CPmat[row, col, passidx] = float("{0:.3f}".format(queue.competence))
        avgs = np.average(CPmat, axis=(0,1))
        for i in range(5):
            stats['comp_{}'.format(i)] = avgs[i]
        return stats

    def reset(self):

        if self.episode_exp:
            goal_idx = self.encode(self.goal)
            # R = np.sum([exp['reward'] - 1 for exp in self.episode_exp])
            # self.queues[goal_idx].append((self.goal, R))
            self.queues[goal_idx].append((self.goal, int(self.episode_exp[-1]['terminal'])))

        self.goal = self.sample_goal()

        obs = self.env.reset()
        state = np.array(self.decode(obs))

        for idx, buffer_item in enumerate(self.episode_exp):
            if self.her == 'future':
                indices = range(idx, len(self.episode_exp))
                future_indices = rnd.sample(indices, np.min([4, len(indices)]))
                buffer_item['future_goals'] = [self.episode_exp[i]['state1'] for i in list(future_indices)]
            elif self.her == 'final':
                buffer_item['future_goals'] = [self.episode_exp[-1]['state1']]
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
    def goal_dim(self):
        return 3,

    @property
    def action_dim(self):
        return [self.env.action_space.n]

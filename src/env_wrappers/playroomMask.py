from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math

class PlayroomMask(Wrapper):
    def __init__(self, env, args):
        super(PlayroomMask, self).__init__(env)

        self.theta = float(args['theta'])
        self.objects = ['agent', 'passenger', 'taxi']
        self.object_idx = None
        self.goal = None
        self.obj_feat = [[0, 1], [2, 3], [4]]
        self.state_low = [0, 0, 0, 0, 0]
        self.state_high = [self.env.nR - 1, self.env.nC - 1, self.env.nR - 1, self.env.nC - 1, 1]
        self.init_state = [2, 2, 0, 0, 0]

        self.queues = [CompetenceQueue() for _ in self.objects]
        self.interests = []
        self.steps = [0 for _ in self.objects]
        self.freqs_act = [0 for _ in self.objects]
        self.freqs_train = [0 for _ in self.objects]
        self.freqs_act_reward = [0 for _ in self.objects]
        self.freqs_train_reward = [0 for _ in self.objects]
        self.test_goals = [(np.array([0, 0, 0, 0, 1]), 2),
                           (np.array([0, 0, 0, 4, 0]), 1),
                           (np.array([0, 0, 4, 0, 0]), 1),
                           (np.array([0, 0, 4, 3, 0]), 1)]

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def reset(self):

        self.update_interests()

        self.sample_goal()
        features = self.obj_feat[self.object_idx]
        # self.mask = self.feat2mask(features)
        self.goal = self.feat2val(features)

        self.freqs_act[self.object_idx] += 1

        obs = self.env.reset()
        state = np.array(self.decode(obs))

        return state

    def sample_goal(self):

        sum = np.sum(self.interests)
        mass = np.random.random() * sum
        idx = 0
        s = self.interests[0]
        while mass > s:
            idx += 1
            s += self.interests[idx]
        self.object_idx = idx

    def obj2mask(self, obj):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[obj]] = 1
        return res

    def feat2val(self, features):
        res = np.zeros(shape=self.state_dim)
        for idx in features:
            while True:
                s = np.random.randint(self.state_low[idx], self.state_high[idx]+1)
                if s != self.init_state[idx]: break
            res[idx] = s
        return res

    def update_interests(self):

        CPs = [abs(q.CP) for q in self.queues]
        maxcp = max(CPs)

        if maxcp > 1:
            self.interests = [math.pow(cp / maxcp, self.theta) + 0.0001 for cp in CPs]
        else:
            self.interests = [math.pow(1 - q.T_mean, self.theta) + 0.0001 for q in self.queues]

    def eval_exp(self, state0, action, state1, goal, object_idx):
        term = False
        r = -1
        goal_feat = self.obj_feat[object_idx]
        goal_vals = goal[goal_feat]
        s1_proj = state1[goal_feat]
        s0_proj = state0[goal_feat]
        if ((s1_proj == goal_vals).all() and (s0_proj != goal_vals).any()):
            r = 0
            term = True
        return r, term

    def make_input(self, state):
        mask = self.obj2mask(self.object_idx)
        input = [np.expand_dims(i, axis=0) for i in [state, self.goal, mask]]
        return input

    def get_stats(self):
        stats = {}
        for i, goal in enumerate(self.objects):
            stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].R_mean))
            stats['T_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].T_mean))
            stats['L_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].L_mean))
            stats['CP_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].CP))
            stats['FA_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_act[i]))
            stats['FT_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_train[i]))
            stats['FAR_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_act_reward[i]))
            stats['FTR_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_train_reward[i]))
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
        return 5,

    @property
    def goal_dim(self):
        return 5,

    @property
    def action_dim(self):
        return [self.env.action_space.n]

    @property
    def min_avg_length_ep(self):
        return np.min([q.L_mean for q in self.queues if q.L_mean != 0])
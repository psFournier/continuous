from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule

class Taxi1G(Wrapper):
    def __init__(self, env, args):
        super(Taxi1G, self).__init__(env)

        self.theta = float(args['theta'])
        self.goals = range(4)
        self.goal_states = [np.array(x) for x in [(0, 0, 4), (0, 4, 1), (4, 0, 2), (4, 3, 3)]]
        self.goal = None
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.interests = []
        self.steps = [0 for _ in self.goals]
        self.freqs_act = [0 for _ in self.goals]
        self.freqs_train = [0 for _ in self.goals]
        self.freqs_act_reward = [0 for _ in self.goals]
        self.freqs_train_reward = [0 for _ in self.goals]
        self.explorations = [LinearSchedule(schedule_timesteps=int(10000),
                                            initial_p=1.0,
                                            final_p=.1) for _ in self.goals]
        # self.trajectories = {}
        # self.trajectories[0] = [[3,3,1,1,4]
        #                         # [3,1,3,1,4],
        #                         # [3,1,1,3,4],
        #                         # [1,3,1,3,4],
        #                         # [1,3,3,1,4]
        #                         ]
        # self.trajectories[1] = [t + [0,0,0,0,5] for t in self.trajectories[0]]
        # self.trajectories[2] = [t + [0,2,2,1,2,2,5] for t in self.trajectories[0]]
        # self.trajectories[3] = [t + [0,2,0,2,2,0,0,5] for t in self.trajectories[0]]


    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, exp):
        term = False
        r = -1
        goal_state = self.goal_states[exp['goal']]
        if ((exp['state1'] == goal_state).all() and (exp['state0'] != goal_state).any()):
            r = 0
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
        self.goal = self.goals[idx]
        self.freqs_act[self.goal] += 1

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
            stats['FA_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_act[goal]))
            stats['FT_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_train[goal]))
            stats['FAR_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_act_reward[goal]))
            stats['FTR_{}'.format(goal)] = float("{0:.3f}".format(self.freqs_train_reward[goal]))
            stats['I_{}'.format(goal)] = float("{0:.3f}".format(self.interests[goal]))
            stats['S_{}'.format(goal)] = float("{0:.3f}".format(self.steps[goal]))
        return stats


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
        return 1,

    @property
    def action_dim(self):
        return 6

    @property
    def min_avg_length_ep(self):
        return np.min([q.L_mean for q in self.queues if q.L_mean != 0])
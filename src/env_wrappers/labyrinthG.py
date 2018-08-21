from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule

class LabyrinthG(Wrapper):
    def __init__(self, env, args):
        super(LabyrinthG, self).__init__(env)
        self.gamma = 0.99
        self.theta = float(args['theta'])
        self.goals = range(4)
        self.goal = None
        self.destination = np.array([0, 4])
        self.queues = [CompetenceQueue() for _ in self.goals]
        self.interests = []
        self.steps = [0 for _ in self.goals]
        self.explorations = [LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1) for _ in self.goals]

        # self.freqs_act = [0 for _ in self.goals]
        # self.freqs_train = [0 for _ in self.goals]
        # self.freqs_act_reward = [0 for _ in self.goals]
        # self.freqs_train_reward = [0 for _ in self.goals]

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, exp):
        r = -1
        term = False
        dist1 = np.linalg.norm(exp['state1'] - self.destination)

        if dist1 <= self.goals[self.goal]:
            r = 0
            term = True

        return r, term

    def reset(self):

        self.set_goal()
        obs = self.env.reset()
        state = np.array(self.decode(obs))

        return state

    def set_goal(self):
        CPs = [abs(q.CP) for q in self.queues]
        maxCP = max(CPs)
        minCP = min(CPs)

        Rs = [q.R for q in self.queues]
        maxR = max(Rs)
        minR = min(Rs)

        try:
            if maxCP > 1:
                self.interests = [math.pow((cp - minCP) / (maxCP - minCP), self.theta) + 0.1 for cp in CPs]
            else:
                self.interests = [math.pow(1 - (r - minR) / (maxR - minR), self.theta) + 0.1 for r in Rs]
        except:
            self.interests = [1.1 for _ in self.goals]

        sum = np.sum(self.interests)
        mass = np.random.random() * sum
        idx = 0
        s = self.interests[0]
        while mass > s:
            idx += 1
            s += self.interests[idx]
        self.goal = idx

    def get_stats(self):
        stats = {}
        for i, goal in enumerate(self.goals):
            stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[i].R))
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

    @property
    def action_dim(self):
        return 4


    # @property
    # def min_avg_length_ep(self):
    #     return np.min([q.L_mean for q in self.queues if q.L_mean != 0])
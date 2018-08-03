from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math

class TaxiGoal2(Wrapper):
    def __init__(self, env, args):
        super(TaxiGoal2, self).__init__(env)
        self.theta = float(args['theta'])

        self.goals = range(2*self.env.nR*self.env.nC + 1)
        self.queues = [CompetenceQueue(), CompetenceQueue(), CompetenceQueue()]
        self.interests = [0, 0, 0]
        self.freqs = [0, 0, 0]
        self.freqs_train = [0, 0, 0]
        self.freqs_reward = [0, 0, 0]
        self.goal = None

        self.trajectories = {}
        self.trajectories[0] = [[3,3,1,1,4]
                                # [3,1,3,1,4],
                                # [3,1,1,3,4],
                                # [1,3,1,3,4],
                                # [1,3,3,1,4]
                                ]
        self.trajectories[1] = [t + [0,0,0,0,5] for t in self.trajectories[0]]
        self.trajectories[2] = [t + [0,2,2,1,2,2,5] for t in self.trajectories[0]]
        self.trajectories[3] = [t + [0,2,0,2,2,0,0,5] for t in self.trajectories[0]]


    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def decode_goal(self, goal):
        out = []
        out.append(goal % self.env.nC)
        goal = goal // self.env.nC
        out.append(goal % self.env.nR)
        goal = goal // self.env.nR
        out.append(goal)
        return list(reversed(out))

    def eval_exp(self, state0, action, state1, goal):
        goal = self.decode_goal(goal)
        term = False
        r = -1
        if goal[0] == 0 and (state1[[0, 1]] == np.array(goal[1:])).all() \
                and (state0[[0, 1]] != np.array(goal[1:])).any():
                r=0
                term=True
        elif goal[0] == 1 and (state1[[2, 3]] == np.array(goal[1:])).all() \
                and state1[4] == 0 and state0[4] == 1:
                r=0
                term=True
        return r, term

    def sample_goal(self):

        self.update_interests()

        sum = np.sum(self.interests)
        mass = np.random.random() * sum
        idx = 0
        s = self.interests[0]
        while mass > s:
            idx += 1
            s += self.interests[idx]
        goal = self.env.goals[idx]

        return goal

    def update_interests(self):

        CPs = [abs(q.CP) for q in self.queues]
        maxcp = max(CPs)

        if maxcp > 10:
            self.interests = [math.pow(cp / maxcp, self.theta) + 0.0001 for cp in CPs]
        else:
            self.interests = [math.pow(1 - q.T_mean, self.theta) + 0.0001 for q in self.queues]

    def reset(self):

        self.goal = self.sample_goal()
        self.freqs[self.goal] += 1

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
        return 5,

    @property
    def goal_dim(self):
        return 1,

    @property
    def action_dim(self):
        return [self.env.action_space.n]

import numpy as np
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class Taxi1G(CPBased):
    def __init__(self, env, args):
        super(Taxi1G, self).__init__(env, args)
        self.goals = range(4)
        self.init()
        self.goal_states = [np.array(x) for x in [(0, 0, 4), (0, 4, 1), (4, 0, 2), (4, 3, 3)]]

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
        if self.posInit:
            r = -1
        else:
            r = 0
        term = False

        goal_state = self.goal_states[exp['goal']]
        if ((exp['state1'] == goal_state).all() and (exp['state0'] != goal_state).any()):
            r += 1
            term = True
        return r, term

    def reset(self):
        self.goal = self.goals[self.get_idx()]
        obs = self.env.reset()
        state = np.array(self.decode(obs))
        return state

    def decode(self, state):
        return list(self.env.decode(state))

    def encode(self, state):
        return self.env.encode(*state)

    @property
    def state_dim(self):
        return 3,

    @property
    def goal_dim(self):
        return 1,

    @property
    def action_dim(self):
        return 6
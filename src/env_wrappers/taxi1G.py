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

    def is_term(self, exp):
        goal_state = self.goal_states[exp['goal']]
        return ((exp['state1'] == goal_state).all() and (exp['state0'] != goal_state).any())

    def reset(self):
        self.goal = self.get_idx()
        state = self.env.reset()
        return state

    @property
    def state_dim(self):
        return 3,

    @property
    def goal_dim(self):
        return 1,

    @property
    def action_dim(self):
        return 6
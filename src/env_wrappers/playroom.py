from gym import Wrapper
import numpy as np

class Playroom(Wrapper):
    def __init__(self, env, args):
        super(Playroom, self).__init__(env)

        self.goals = range(4)
        self.goal_states = [np.array(x) for x in [(0, 0, 4), (0, 4, 1), (4, 0, 2), (4, 3, 3)]]
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
        state, _, _, _ = self.env.step(action)
        return state

    def eval_exp(self, state0, action, state1, goal):
        term = False
        r = -1
        if ((state1 == self.goal_states[goal]).all() and (state0 != self.goal_states[goal]).any()):
            r = 0
            term = True
        return r, term

    def reset(self):
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
        return 3,

    @property
    def goal_dim(self):
        return 1,

    @property
    def action_dim(self):
        return [self.env.action_space.n]

from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class Taxi2GM(CPBased):
    def __init__(self, env, args):
        super(Taxi2GM, self).__init__(env, args)
        self.goals = ['agent', 'passenger', 'taxi']
        self.object = None
        self.init()
        self.obj_feat = [[0, 1], [2, 3], [4]]
        self.state_low = [0, 0, 0, 0, 0]
        self.state_high = [self.env.nR - 1, self.env.nC - 1, self.env.nR - 1, self.env.nC - 1, 1]
        self.init_state = [2, 2, 0, 0, 0]

        self.test_goals = [(np.array([0, 0, 0, 0, 1]), 2),
                           (np.array([0, 0, 0, 4, 0]), 1),
                           (np.array([0, 0, 4, 0, 0]), 1),
                           (np.array([0, 0, 4, 3, 0]), 1)]

        self.explorations = [LinearSchedule(schedule_timesteps=int(10000),
                                            initial_p=1.0,
                                            final_p=.1) for _ in self.goals]

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, exp):
        term = False
        r = -1
        goal_feat = self.obj_feat[exp['object']]
        goal_vals = exp['goal'][goal_feat]
        s1_proj = exp['state1'][goal_feat]
        s0_proj = exp['state0'][goal_feat]
        if ((s1_proj == goal_vals).all() and (s0_proj != goal_vals).any()):
            r = 0
            term = True
        return r, term

    def reset(self):
        self.object = self.get_idx()
        features = self.obj_feat[self.object]
        self.goal = np.zeros(shape=self.state_dim)
        for idx in features:
            while True:
                s = np.random.randint(self.state_low[idx], self.state_high[idx] + 1)
                if s != self.init_state[idx]: break
            self.goal[idx] = s
        obs = self.env.reset()
        state = np.array(self.decode(obs))
        return state

    def obj2mask(self, obj):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[obj]] = 1
        return res

    def decode(self, state):
        return list(self.env.decode(state))

    def encode(self, state):
        return self.env.encode(*state)

    @property
    def state_dim(self):
        return 5,

    @property
    def goal_dim(self):
        return 5,

    @property
    def action_dim(self):
        return 6
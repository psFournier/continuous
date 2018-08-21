from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class PlayroomGM(CPBased):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env, args)

        self.goals = [obj.name for obj in self.env.objects]
        self.object = None
        self.init()
        self.obj_feat = [[4 + 4 * j] for j in range(len(self.goals))]
        self.state_low = self.env.state_low
        self.state_high = self.env.state_high
        self.init_state = self.env.state_init

        self.explorations = [LinearSchedule(schedule_timesteps=int(10000),
                                            initial_p=1.0,
                                            final_p=.1) for _ in self.goals]

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(obs)
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
        state = np.array(obs)
        return state

    def obj2mask(self, obj):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[obj]] = 1
        return res

    @property
    def state_dim(self):
        return 18,

    @property
    def goal_dim(self):
        return 18,

    @property
    def action_dim(self):
        return 11
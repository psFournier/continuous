from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class PlayroomGF(CPBased):
    def __init__(self, env, args):
        super(PlayroomGF, self).__init__(env, args)
        self.goals = list(range(self.state_dim[0]))
        self.feature = None
        self.init()
        self.features = self.goals
        self.state_low = self.env.state_low
        self.state_high = self.env.state_high
        self.init_state = self.env.state_init

    def step(self, exp):
        self.steps[self.feature] += 1
        exp['goal'] = self.goal
        exp['mask'] = self.obj2mask(self.feature)
        exp['state1'] = self.env.step(exp['action'])
        exp = self.eval_exp(exp)
        return exp

    def explor_eps(self):
        step = self.steps[self.feature]
        return 1 + min(float(step) / 1e4, 1) * (0.1 - 1)

    def processEp(self, episode):
        T = int(episode[-1]['terminal'])
        if T:
            print('done')
            self.dones[self.feature] += 1
        R = np.sum([exp['reward'] for exp in episode])
        S = len(episode)
        self.queues[self.feature].append({'R': R, 'S': S, 'T': T})

    def is_term(self, exp):
        indices = np.where(exp['mask'])
        goal = exp['goal'][indices]
        s1_proj = exp['state1'][indices]
        s0_proj = exp['state0'][indices]
        return ((s1_proj == goal).all() and (s0_proj != goal).any())

    def reset(self):
        self.feature = self.get_idx()
        self.goal = np.array(self.init_state)
        while True:
            self.goal[self.feature] = np.random.randint(self.state_low[self.feature], self.state_high[self.feature] + 1)
            if (self.goal != self.init_state).any():
                break
        state = self.env.reset()
        return state

    def obj2mask(self, f):
        res = np.zeros(shape=self.state_dim)
        res[self.features[f]] = 1
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
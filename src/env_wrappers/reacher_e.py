from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased


class Reacher_e(CPBased):
    def __init__(self, env, args):
        super(Reacher_e, self).__init__(env, args)

        self.goals = [0.02, 0.03, 0.04, 0.05]
        self.init()
        self.obj_feat = [8]

        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.init)

    def is_term(self, exp):
        d = np.linalg.norm(exp['state1'][[6, 7]])
        term = d < self.goals[exp['goal']]
        return term

    def reset(self, goal=None):

        if goal is None:
            self.goal = self.get_idx()
        else:
            self.goal = goal

        features = self.obj_feat[self.object]

        _ = self.env.reset()
        qpos = self.unwrapped.sim.data.qpos.flatten()
        qvel = self.unwrapped.sim.data.qvel.flatten()

        qpos[[2,3]] = self.goal
        self.reached = False

        self.unwrapped.set_state(qpos, qvel)
        obs = self.unwrapped._get_obs()
        state = self.add_goal(obs)
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()

        return state

    @property
    def state_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 2,

    @property
    def goal_dim(self):
        return 1,


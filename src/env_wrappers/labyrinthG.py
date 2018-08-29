from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class LabyrinthG(CPBased):
    def __init__(self, env, args):
        super(LabyrinthG, self).__init__(env, args)
        self.goals = range(4)
        self.init()
        self.destination = np.array([0, 4])

    def is_term(self, exp, goal):
        return np.linalg.norm(exp['state1'] - self.destination) <= self.goals[goal]

    def reset(self):
        self.goal = self.get_idx()
        state = self.env.reset()
        return state

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
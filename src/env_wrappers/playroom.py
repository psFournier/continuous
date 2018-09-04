from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class Playroom(CPBased):
    def __init__(self, env, args):
        super(Playroom, self).__init__(env, args)
        self.goals = [0]
        self.goal = 0
        self.init()
        self.state_low = self.env.state_low
        self.state_high = self.env.state_high
        self.init_state = self.env.state_init

    def is_term(self, exp):
        return (exp['state0'][16] != 4 and exp['state0'][16] == 4)

    def reset(self):
        state = self.env.reset()
        return state

    @property
    def state_dim(self):
        return 18,

    @property
    def goal_dim(self):
        return 0,

    @property
    def action_dim(self):
        return 11
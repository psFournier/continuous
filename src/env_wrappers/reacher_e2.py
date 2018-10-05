from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased, RndBased, Base


class Reacher_e2(RndBased):
    def __init__(self, env, args):
        super(Reacher_e2, self).__init__(env, args, [0.01], [0.1])
        self.init()

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < exp['g']:
            exp['r'] = 1
            exp['t'] = True
        else:
            exp['r'] = 0
            exp['t'] = False
        # exp['r'] += (- np.square(exp['a']).sum())
        return exp

    def end_episode(self, trajectory):
        R = np.sum([self.unshape(exp['r'], exp['t']) for exp in trajectory])
        self.queue.append(R)
        augmented_ep = []
        if self.args['--her'] != '0':
            min_d = min([np.linalg.norm(e['s1'][[6, 7]]) for e in trajectory])
        for i, expe in enumerate(reversed(trajectory)):
            augmented_ep.append(expe.copy())
            if self.args['--her'] != '0':
                expe['g'] = np.array([min_d + 0.01])
                expe = self.eval_exp(expe)
                augmented_ep.append(expe.copy())
        return augmented_ep

    @property
    def state_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 2,

    @property
    def goal_dim(self):
        return 1,


from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased


class Reacher_e(CPBased):
    def __init__(self, env, args):
        super(Reacher_e, self).__init__(env, args, [[0.02], [0.04], [0.06], [0.08], [0.1]])
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
        self.queues[self.idx].append(R)

        goals = []
        augmented_ep = []
        for i, expe in enumerate(reversed(trajectory)):

            augmented_ep.append(expe.copy())

            for g in goals:
                expe['g'] = g
                expe = self.eval_exp(expe)
                augmented_ep.append(expe.copy())

            if self.args['--her'] != '0':
                for goal in self.goals:
                    if goal != expe['g'] and goal not in goals:
                        expe['g'] = goal
                        expe = self.eval_exp(expe)
                        if expe['r'] == 1:
                            goals.append(goal)
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


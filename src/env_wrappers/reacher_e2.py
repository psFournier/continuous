from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased


class Reacher_e2(CPBased):
    def __init__(self, env, args):
        super(Reacher_e2, self).__init__(env, args, [[0.02], [0.04], [0.06], [0.08], [0.1]])
        self.init()
        self.minQ = -np.inf
        self.maxQ = np.inf

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < exp['g']:
            exp['r'] = 0
            exp['t'] = True
        else:
            exp['r'] = -1
            exp['t'] = False
        return exp

    def end_episode(self, trajectory):
        R = 0
        for exp in reversed(trajectory):
            R = R * self.gamma + exp['r'] + 1 + exp['t'] * 99
        self.queues[self.idx].append(R)

    def reset(self):
        self.idx = np.random.randint(len(self.goals))
        self.goal = np.array(self.goals[self.idx])
        _ = self.env.reset()
        qpos = self.unwrapped.sim.data.qpos.flatten()
        qvel = self.unwrapped.sim.data.qvel.flatten()
        qpos[2] = -0.025
        qpos[3] = 0.15
        self.unwrapped.set_state(qpos, qvel)
        state = self.unwrapped._get_obs()
        return state

    def augment_episode(self, trajectory):

        goals = []
        augmented_ep = []
        for i, expe in enumerate(reversed(trajectory)):

            augmented_ep.append(expe.copy())

            for g in goals:
                altexp = expe.copy()
                altexp['g'] = g
                altexp = self.eval_exp(altexp)
                augmented_ep.append(altexp.copy())

            if self.args['--her'] != '0':
                for i, goal in enumerate(self.goals):
                    if np.random.rand() < self.interests[i] and goal != expe['g'] and goal not in goals:
                        altexp = expe.copy()
                        altexp['g'] = goal
                        altexp = self.eval_exp(altexp)
                        if altexp['r'] == 1:
                            goals.append(goal)
                            augmented_ep.append(altexp.copy())

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


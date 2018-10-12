from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import RndBased

class PlayroomGMulti(RndBased):
    def __init__(self, env, args):
        super(PlayroomGMulti, self).__init__(env, args, None, None)
        self.mask = None
        self.init()
        self.obj_feat = [[i] for i in range(2, 11)]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.init)

    def step(self, exp):
        exp['g'] = self.goal
        exp['m'] = self.mask
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def eval_exp(self, exp):
        exp['r'] = np.sum([mi if exp['g'][i] == exp['s1'][i] else 0 for i, mi in enumerate(exp['m'])])
        if exp['r'] == 1:
            exp['t'] = True
            exp['r'] = 100
        return exp

    def reset(self):

        self.mask = np.random.randint(0,2,size=self.state_dim)
        self.goal = self.init_state.copy()
        for i, mi in enumerate(self.mask):
            if mi:
                self.goal[i] = np.random.randint(self.state_low[i], self.state_high[i] + 1)
        self.mask = self.mask / np.sum(self.mask)
        state = self.env.reset()
        return state

    def augment_episode(self, episode):

        goals = []
        masks = []
        augmented_ep = []
        trajectory_mask = []
        if 'm' in trajectory[-1].keys():
            trajectory_mask.append(trajectory[-1]['m'])
        # if self.args['--wimit'] != '0':
        #     mcrs = []

        for i, expe in enumerate(reversed(trajectory)):

            # For this way of augmenting episodes, the agent actively searches states that
            # are new in some sense, with no importance granted to the difficulty of reaching
            # such states
            for j, (g, m) in enumerate(zip(goals, masks)):
                altexp = expe.copy()
                altexp['g'] = g
                altexp['m'] = m
                altexp = self.eval_exp(altexp)
                # if self.args['--wimit'] != '0':
                #     mcrs[j] = mcrs[j] * self.gamma + expe['r']
                #     expe['mcr'] = np.expand_dims(mcrs[j], axis=1)
                augmented_ep.append(altexp.copy())

            for obj_idx, goal in enumerate(self.goals):
                # self.goals contains the objects, not their goal value
                m = self.obj2mask(obj_idx)
                # I compare the object mask to the one pursued and to those already "imagined"
                if all([(m != m2).any() for m2 in masks + trajectory_mask]):
                    # We can't test all alternative goals, and chosing random ones would bring
                    # little improvement. So we select goals as states where an object as changed
                    # its state.
                    s1m = expe['s1'][np.where(m)]
                    s0m = expe['s0'][np.where(m)]
                    if (s1m != s0m).any():
                        altexp = expe.copy()
                        altexp['g'] = expe['s1']
                        altexp['m'] = m
                        altexp = self.eval_exp(altexp)
                        # if self.args['--wimit'] != '0':
                        #     mcr = (1 - self.gamma ** (i + 1)) / (1 - self.gamma)
                        #     mcrs.append(mcr)
                        augmented_ep.append(altexp)
                        goals.append(expe['s1'])
                        masks.append(m)

        return augmented_ep

    @property
    def state_dim(self):
        return 11,

    @property
    def goal_dim(self):
        return 11,

    @property
    def action_dim(self):
        return 7
from gym import Wrapper
import numpy as np
from samplers.competenceQueue import CompetenceQueue
import math
from utils.linearSchedule import LinearSchedule
from .base import CPBased

class PlayroomGM(CPBased):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env, args, [obj.name for obj in env.objects])
        self.mask = None
        self.init()
        self.obj_feat = [[i] for i in range(2, 11)]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.init)
        self.minQ, self.maxQ = 0, 100

    def step(self, exp):
        self.steps[self.idx] += 1
        exp['g'] = self.goal
        exp['m'] = self.mask
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def eval_exp(self, exp):
        indices = np.where(exp['m'])
        goal = exp['g'][indices]
        s1_proj = exp['s1'][indices]
        if (s1_proj == goal).all():
            exp['t'] = True
            exp['r'] = self.shape(0, 1)
        else:
            exp['t'] = False
            exp['r'] = self.shape(-1, 0)
        return exp

    def reset(self):
        self.idx = self.get_idx()
        features = self.obj_feat[self.idx]
        self.goal = self.init_state.copy()
        self.mask = self.obj2mask(self.idx)
        while True:
            for f in features:
                self.goal[f] = np.random.randint(self.state_low[f], self.state_high[f] + 1)
            if (self.goal != self.init_state).any():
                break

        state = self.env.reset()
        return state

    def obj2mask(self, idx):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[idx]] = 1
        return res

    def end_episode(self, trajectory):
        R = np.sum([self.unshape(exp['r'], exp['t']) for exp in trajectory])
        self.queues[self.idx].append(R)

        goals = []
        masks = []
        augmented_ep = []
        if self.args['--wimit'] != '0':
            mcrs = []

        for i, expe in enumerate(reversed(trajectory)):

            augmented_ep.append(expe.copy())

            for j, (g, m) in enumerate(zip(goals, masks)):
                expe['g'] = g
                expe['m'] = m
                expe = self.eval_exp(expe)
                if self.args['--wimit'] != '0':
                    mcrs[j] = mcrs[j] * self.gamma + expe['r']
                    expe['mcr'] = np.expand_dims(mcrs[j], axis=1)
                augmented_ep.append(expe.copy())

            if self.args['--her'] != '0':
                for obj_idx, goal in enumerate(self.goals):
                    # self.goals contains the objects, not their goal value
                    m = self.obj2mask(obj_idx)
                    # I compare the object mask to the one pursued and to those already "imagined"
                    if m != expe['m'] and m not in masks:
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
                            if self.args['--wimit'] != '0':
                                mcr = (1 - self.gamma ** (i + 1)) / (1 - self.gamma)
                                mcrs.append(mcr)
                            augmented_ep.append(altexp)
                            goals.append(expe['s1'])
                            masks.append(m)

        return augmented_ep

    def end_episode(self, trajectory):
        R = np.sum([self.unshape(exp['r'], exp['t']) for exp in trajectory])
        self.queues[self.idx].append(R)

        augmented_ep = []
        goals = []
        masks = []
        objs = []
        if self.args['--wimit'] != '0':
            mcrs = []

        for i, expe in enumerate(reversed(self.trajectory)):

            expe['mcr'] = np.expand_dims(-100, axis=1)
            self.buffer.append(expe.copy())
            if self.args['--wimit'] != '0':
                self.bufferImit.append(expe.copy())

            if self.args['--her'] != '0':
                for obj, name in enumerate(self.env.goals):
                    mask = self.env.obj2mask(obj)
                    s1m = expe['s1'][np.where(mask)]
                    s0m = expe['s0'][np.where(mask)]
                    found = obj not in objs and (s1m != s0m).any()
                    if found:
                        goals.append(expe['s1'].copy())
                        masks.append(mask.copy())
                        objs.append(obj)
                        if self.args['--wimit'] != '0':
                            mcr = (1 - self.critic.gamma ** (i + 1)) / (1 - self.critic.gamma)
                            mcrs.append(mcr)

            for j, (g, m) in enumerate(zip(goals, masks)):
                expe['g'] = g
                expe['m'] = m
                expe = self.env.eval_exp(expe)
                if self.args['--wimit'] != '0':
                    mcrs[j] = mcrs[j] * self.critic.gamma + expe['r']
                    expe['mcr'] = np.expand_dims(mcrs[j], axis=1)
                    self.bufferImit.append(expe.copy())
                self.buffer.append(expe.copy())

        for i, expe in enumerate(reversed(trajectory)):
            augmented_ep.append(expe.copy())
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
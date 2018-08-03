import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os

RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNG
from agents import DQNG
from buffers import ReplayBuffer, PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule
import random as rnd
from samplers.competenceQueue import CompetenceQueue
import math


class DQNG2(DQNG):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNG2, self).__init__(args, sess, env, env_test, logger)

        self.names = ['state0', 'action', 'state1']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)

        self.explorations = [LinearSchedule(schedule_timesteps=int(10000),
                                           initial_p=1.0,
                                           final_p=.1) for _ in self.env.goals]

    def make_exp(self, state0, action, state1):
        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'terminal': terminal}

        return experience

    def train_autonomous(self, exp):
        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            if self.train_last_expe:
                for key in self.names:
                    experiences[key].append(exp[key])
            self.train_critic(experiences)

    def expe2array(self, experiences):
        s0 = np.array(experiences['state0'])
        a = np.array(experiences['action'])
        s1 = np.array(experiences['state1'])
        return s0, a, s1

    def preprocess(self, experiences):
        s0, a, s1 = self.expe2array(experiences)
        eval = []
        for k in range(self.batch_size + int(self.train_last_expe)):
            eval.append(self.env.eval_exp(s0[k], a[k], s1[k], self.env.goal))
        r, t = zip(*eval)
        r = np.array(r)
        t = np.array(t)
        g = np.ones(shape=a.shape, dtype='int32') * self.env.goal
        inputs = [s0, a, g]
        targets = self.compute_targets(s1, g, r, t)
        return inputs, targets

    @property
    def explore_prop(self):
        return self.explorations[self.env.goal].value(self.env_step)

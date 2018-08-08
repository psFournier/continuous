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


class DQNG0(DQNG):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNG0, self).__init__(args, sess, env, env_test, logger)

        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.beta_schedule = LinearSchedule(schedule_timesteps=int(200000),
                                          initial_p=float(args['beta0']),
                                          final_p=1.)

    def make_exp(self, state0, action, state1):
        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'reward': reward,
                      'terminal': terminal,
                      'goal': self.env.goal}

        self.trajectory.append(experience)

        return experience

    def train_autonomous(self):
        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            self.train_critic(experiences)
            self.target_train()

    def expe2array(self, experiences):
        s0 = np.array(experiences['state0'])
        a = np.array(experiences['action'])
        s1 = np.array(experiences['state1'])
        g = np.array(experiences['goal'])
        r = np.array(experiences['reward'])
        t = np.array(experiences['terminal'])
        return s0, a, s1, g, r, t

    def preprocess(self, experiences):
        s0, a, s1, g ,r, t = self.expe2array(experiences)
        inputs = [s0, a, g]
        targets = self.compute_targets(s1, g, r, t)
        weights = np.array([(self.env.min_avg_length_ep / self.env.queues[gi].L_mean) ** self.beta for gi in g])
        return inputs, targets, weights

    @property
    def beta(self):
        return self.beta_schedule.value(self.env_step)



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


class DQNG01(DQNG):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNG01, self).__init__(args, sess, env, env_test, logger)

        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.beta = float(args['beta'])
        self.exploration = [LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1) for _ in self.env.goals]

    def train_autonomous(self):
        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            self.train_critic(experiences)
            self.target_train()

    def preprocess(self, experiences):
        s0, a, s1, r, t, g = self.expe2array(experiences)
        inputs = [s0, a, g]
        targets = self.compute_targets(s1, g, r, t)
        weights = np.array([(self.env.min_avg_length_ep / self.env.queues[gi].L_mean) ** self.beta for gi in g])
        return inputs, targets, weights

    def reset(self):

        if self.trajectory:
            R, T, L = self.process_episode()
            self.env.queues[self.env.goal].append((R, T, L))
            self.env.freqs_act_reward[self.env.goal] += int(T)
            self.trajectory.clear()

        self.env.update_interests()
        if self.env_step < 100000:
            self.env.goal = 2
        else:
            self.env.goal = 3
        self.env.freqs_act[self.env.goal] += 1

        obs = self.env.env.reset()
        state = np.array(self.env.decode(obs))

        self.episode_step = 0

        return state

    @property
    def explore_prop(self):
        return self.exploration[self.env.goal].value(self.env.steps[self.env.goal])


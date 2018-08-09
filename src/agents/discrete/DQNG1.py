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



class DQNG1(DQNG):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNG1, self).__init__(args, sess, env, env_test, logger)

        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal']
        self.buffers = {goal: ReplayBuffer(limit=int(1e5), names=self.names) for goal in self.env.goals}
        self.buffers['tutor'] = ReplayBuffer(limit=int(1e2), names=self.names)

        if self.tutor_imitation:
            self.get_tutor_exp(goal=3)
            self.batch_size = int(self.batch_size / 2)

    def append_tutor_exp(self, new_expe):
        self.buffers['tutor'].append(new_expe)

    def train_autonomous(self):
        buffer = self.buffers[self.env.goal]
        if buffer.nb_entries > self.batch_size:
            experiences = buffer.sample(self.batch_size)
            self.train_critic(experiences)
            self.target_train()

    def preprocess(self, experiences):
        s0, a, s1, r, t, g = self.expe2array(experiences)
        inputs = [s0, a, g]
        targets = self.compute_targets(s1, g, r, t)
        return inputs, targets

    def train_imitation(self):
        experiences = self.buffers['tutor'].sample(self.batch_size)
        s0, a, s1, g, r, t = self.expe2array(experiences)

        targets = np.zeros((self.batch_size, 1))
        self.loss_imitation = self.critic.margin_model.train_on_batch(x=[s0, a, g], y=targets)

        targets = self.compute_targets(s1, g, r, t)
        self.critic.qValue_model.train_on_batch(x=[s0, a, g], y=targets)

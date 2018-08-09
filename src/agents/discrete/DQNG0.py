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
        self.beta = float(args['beta'])

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



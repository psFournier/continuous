import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os

from agents import DQNG
from networks import CriticDQNG
from buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQNG2(DQNG):
    def __init__(self, args, sess, env, env_test, logger):
        super(DQNG2, self).__init__(args, env, env_test, logger)

    def init(self, env):
        self.names = ['state0', 'action', 'state1']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)

    def step(self):

        self.env_step += 1
        self.episode_step += 1
        self.env.steps[self.env.goal] += 1
        self.exp['goal'] = self.env.goal
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())

        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1 = [np.array(experiences[name]) for name in self.names]

            eval = []
            for k in range(self.batch_size):
                eval.append(self.env.eval_exp(s0[k], a0[k], s1[k], self.env.goal))
            r, t = zip(*eval)
            r = np.array(r)
            t = np.array(t)
            g = np.ones(shape=a0.shape, dtype='int32') * self.env.goal

            a1 = self.critic.bestAction_model.predict_on_batch([s1, g])
            q = self.critic.target_qValue_model.predict_on_batch([s1, a1, g])
            targets = self.compute_targets(r, t, q)
            # weights = np.array([(self.env.min_avg_length_ep / self.env.queues[gi].L_mean) ** self.beta for gi in g])
            self.critic.qValue_model.train_on_batch(x=[s0, a0, g], y=targets)
            self.critic.target_train()
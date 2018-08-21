import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import CriticDQN
from agents.agent import Agent
from buffers import ReplayBuffer, PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule
import random as rnd


class DQN(Agent):

    def __init__(self, args, sess, env, env_test, logger):

        super(DQN, self).__init__(args, sess, env, env_test, logger)
        self.per = bool(args['per'])
        self.self_imitation = bool(int(args['self_imit']))
        self.tutor_imitation = bool(int(args['tutor_imit']))
        self.her = bool(int(args['her']))

        self.names = ['state0', 'action', 'state1', 'reward', 'terminal']

        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)

        self.critic = CriticDQN(sess,
                                s_dim=env.state_dim,
                                num_a=env.action_dim,
                                gamma=0.99,
                                tau=0.001,
                                learning_rate=0.001)

        self.trajectory = []

        self.loss_qVal = []
        self.critic.target_train()

    def step(self):

        self.env_step += 1
        self.episode_step += 1
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp)

        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t = [np.array(experiences[name]) for name in self.names]

            a1 = self.critic.bestAction_model.predict_on_batch([s1])
            q = self.critic.target_qValue_model.predict_on_batch([s1, a1])
            targets = self.compute_targets(r, t, q)

            self.critic.qValue_model.train_on_batch(x=[s0, a0], y=targets)
            self.critic.target_train()

    def compute_targets(self, r, t, q, clip=True):
        targets = []
        for k in range(self.batch_size):
            target = r[k] + (1 - t[k]) * self.critic.gamma * q[k]
            if clip:
                target_clip = np.clip(target, -0.99 / (1 - self.critic.gamma), 0.01)
                targets.append(target_clip)
            else:
                targets.append(target)
        targets = np.array(targets)
        return targets

    def reset(self):

        if self.trajectory:
            R = 0
            for expe in reversed(self.trajectory):
                R += int(expe['reward'])
                self.buffer.append(expe)
            self.env.queues[0].append({'step': self.env_step, 'R': R})
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.env.exploration[0].value(self.env_step):
            action = np.random.randint(0, self.env.action_space.n)
        else:
            inputs = [np.reshape(state, (1, self.critic.s_dim[0]))]
            action = self.critic.bestAction_model.predict(inputs)
            action = action[0, 0]
        return action

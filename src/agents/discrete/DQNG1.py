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

    def process_episode(self):

        R = 0
        T = False
        for expe in reversed(self.trajectory):
            R = R * self.critic.gamma + int(expe['terminal']) - 1
            expe['R'] = R
            self.buffers[self.env.goal].append(expe)
            T = T or expe['terminal']
        return R, T

    def append_tutor_exp(self, new_expe):
        self.buffers['tutor'].append(new_expe)

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
        buffer = self.buffers[self.env.goal]
        if buffer.nb_entries > self.batch_size:
            experiences = buffer.sample(self.batch_size)
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
        s0, a, s1, g, r, t = self.expe2array(experiences)
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

    def train_critic(self, experiences):

        s0, a, s1, g, r, t = self.expe2array(experiences)
        targets = self.compute_targets(s1, g, r, t)
        self.loss_qVal, q_values = self.critic.qValue_model.train_on_batch(x=[s0, a, g], y=targets)
        td_errors = targets - q_values
        #
        # for goal in range(len(self.env.goals)):
        #     if goal in g:
        #         self.td_errors[goal] = np.mean(td_errors[np.where(g == goal)])
        #         self.q_values[goal] = np.mean(q_values[np.where(g == goal)])

        return td_errors

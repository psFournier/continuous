import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os

RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents.agent import Agent
from buffers import ReplayBuffer, PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule
import random as rnd
from samplers.competenceQueue import CompetenceQueue
import math


class DQNGM(Agent):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNGM, self).__init__(args, sess, env, env_test, logger)
        self.per = bool(args['per'])
        self.self_imitation = bool(int(args['self_imit']))
        self.tutor_imitation = bool(int(args['tutor_imit']))
        self.her = bool(int(args['her']))

        self.critic = CriticDQNGM(sess,
                                 s_dim=env.state_dim,
                                 g_dim=env.goal_dim,
                                 num_a=env.action_space.n,
                                 gamma=0.99,
                                 tau=0.001,
                                 learning_rate=0.001)

        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal', 'mask']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)

        self.trajectory = []
        self.exploration = LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1)

    def make_exp(self, state0, action, state1):

        self.env_step += 1
        self.episode_step += 1

        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal, self.env.mask)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'reward': reward,
                      'terminal': terminal,
                      'goal': self.env.goal,
                      'mask': self.env.mask}

        self.trajectory.append(experience)

        return experience

    def reset(self):

        if self.trajectory:
            R, T, L = self.process_episode()
            self.env.queues[self.env.module].append((R,T,L))
            self.env.freqs_act_reward[self.env.module] += int(T)
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def process_episode(self):

        R = 0
        T = False
        L = 0
        for expe in reversed(self.trajectory):
            R = R * self.critic.gamma + int(expe['terminal']) - 1
            L += 1
            expe['R'] = R
            self.buffer.append(expe)
            T = T or expe['terminal']
        return R, T, L

    def expe2array(self, experiences):
        exp = [np.array(experiences[name]) for name in self.names]
        return exp

    def train(self):
        self.train_autonomous()

    def train_autonomous(self):
        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            self.train_critic(experiences)
            self.target_train()

    def train_critic(self, experiences):

        inputs, targets, sample_weights = self.preprocess(experiences)
        self.loss_qVal, q_values = self.critic.qValue_model.train_on_batch(x=inputs,
                                                                           y=targets,
                                                                           sample_weight=sample_weights)
        td_errors = targets - q_values

        return td_errors

    def preprocess(self, experiences):
        s0, a, s1, r, t, g, m = self.expe2array(experiences)
        inputs = [s0, a, g, m]
        targets = self.compute_targets(s1, g, m, r, t)
        weights = np.ones(shape=a.shape)
        return inputs, targets, weights

    def compute_targets(self, s1, g, m, r, t):
        a = self.critic.bestAction_model.predict_on_batch([s1, g, m])
        q = self.critic.target_qValue_model.predict_on_batch([s1, a, g, m])

        targets = []
        for k in range(len(s1)):
            self.env.freqs_train[g[k]] += 1
            self.env.freqs_train_reward[g[k]] += t[k]
            target = r[k] + (1 - t[k]) * self.critic.gamma * q[k]
            if TARGET_CLIP:
                target_clip = np.clip(target, -0.99 / (1 - self.critic.gamma), 0.01)
                targets.append(target_clip)
            else:
                targets.append(target)
        targets = np.array(targets)
        return targets

    def init_targets(self):
        self.critic.target_init()

    def target_train(self):
        self.critic.target_train()

    def act_random(self, state):
        return np.random.randint(0, self.env.action_space.n)

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.explore_prop:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            inputs = [np.reshape(state, (1, self.critic.s_dim[0])),
                      np.reshape(self.env.goal, (1, self.critic.g_dim[0])),
                      np.reshape(self.env.mask, (1, self.critic.g_dim[0]))]
            action = self.critic.bestAction_model.predict(inputs)
            action = action[0, 0]
        return action

    def log(self):

        if self.env_step % self.eval_freq == 0:

            wrapper_stats = self.env.get_stats()
            self.stats['step'] = self.env_step
            self.stats['loss_qVal'] = self.loss_qVal

            for key, val in wrapper_stats.items():
                self.stats[key] = val

            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])

            self.logger.dumpkvs()

    @property
    def explore_prop(self):
        return self.exploration.value(self.env_step)
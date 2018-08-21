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
from agents import DQN
from buffers import ReplayBuffer, PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule
import random as rnd
from samplers.competenceQueue import CompetenceQueue
import math


class DQNGM(DQN):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNGM, self).__init__(args, sess, env, env_test, logger)
        self.critic = CriticDQNGM(sess,
                                 s_dim=env.state_dim,
                                 g_dim=env.goal_dim,
                                 num_a=env.action_dim,
                                 gamma=0.99,
                                 tau=0.001,
                                 learning_rate=0.001)

        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal', 'object']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)

        self.trajectory = []
        self.explorations = [LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1) for _ in self.env.objects]

    def step(self):

        self.env_step += 1
        self.episode_step += 1
        self.exp['goal'] = self.env.goal
        self.exp['object'] = self.env.object_idx
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp)

        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, g, o = [np.array(experiences[name]) for name in self.names]
            m = np.array([self.env.obj2mask(o[k]) for k in range(self.batch_size)])

            a1 = self.critic.bestAction_model.predict_on_batch([s1, g, m])
            q = self.critic.target_qValue_model.predict_on_batch([s1, a1, g, m])

            targets = self.compute_targets(r, t, q)

            self.loss_qVal, q_values = self.critic.qValue_model.train_on_batch(x=[s0, a0, g, m], y=targets)
            self.critic.target_train()

            # td_errors = targets - q_values
            # for k in range(self.batch_size):
            #     self.env.freqs_train[o[k]] += 1
            #     self.env.freqs_train_reward[o[k]] += t[k]
            #     self.env.td_errors[o[k]] += td_errors[k][0]
            #     self.env.td_errors2[o[k]] += td_errors[k][0]


    def reset(self):

        if self.trajectory:
            R = 0
            T = False
            L = 0
            for expe in reversed(self.trajectory):
                R = R * self.critic.gamma + int(expe['terminal']) - 1
                L += 1
                expe['R'] = R
                self.buffer.append(expe)
                T = T or expe['terminal']
            self.env.queues[self.env.object_idx].append((R, T, L))
            self.trajectory.clear()
            # for o in range(len(self.env.objects)):
            #     self.env.queues[o].appendTD(self.env.td_errors2[o])
            # self.env.freqs_act_reward[self.env.object_idx] += int(T)
            # self.env.td_errors2 = [0 for _ in self.env.objects]
        state = self.env.reset()
        self.episode_step = 0

        return state

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.explorations[self.env.object_idx].value(self.env_step):
            action = np.random.randint(0, self.env.action_dim)
        else:
            mask = self.env.obj2mask(self.env.object_idx)
            input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, mask]]
            action = self.critic.bestAction_model.predict(input, batch_size=1)
            action = action[0, 0]
        return action
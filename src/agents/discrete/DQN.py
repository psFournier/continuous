import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os
RENDER_TRAIN = False
TARGET_CLIP = True
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
                                num_a=env.action_space.n,
                                gamma=0.99,
                                tau=0.001,
                                learning_rate=0.001)

        self.exploration = LinearSchedule(schedule_timesteps=int(1000),
                                          initial_p=1.0,
                                          final_p=.1)
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

            targets = []
            for k in range(len(s0)):
                target = r[k] + (1 - t[k]) * self.critic.gamma * q[k]
                if TARGET_CLIP:
                    target_clip = np.clip(target, -0.99 / (1 - self.critic.gamma), 0.01)
                    targets.append(target_clip)
                else:
                    targets.append(target)
            targets = np.array(targets)

            self.critic.qValue_model.train_on_batch(x=[s0, a0], y=targets)
            self.critic.target_train()

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
            self.env.queues[0].append((R, T, L))
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.exploration.value(self.env_step):
            action = np.random.randint(0, self.env.action_space.n)
        else:
            inputs = [np.reshape(state, (1, self.critic.s_dim[0]))]
            action = self.critic.bestAction_model.predict(inputs)
            action = action[0, 0]
        return action

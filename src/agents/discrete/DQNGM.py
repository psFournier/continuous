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

        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal', 'object']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)

        self.trajectory = []
        self.exploration = LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1)

    def make_exp(self, state0, action, state1):

        self.env_step += 1
        self.episode_step += 1

        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal, self.env.object_idx)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'reward': reward,
                      'terminal': terminal,
                      'goal': self.env.goal,
                      'object': self.env.object_idx}

        self.trajectory.append(experience)

        return experience

    def reset(self):

        if self.trajectory:
            R, T, L = self.process_episode()
            self.env.queues[self.env.object_idx].append((R,T,L))
            self.env.freqs_act_reward[self.env.object_idx] += int(T)
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
        s0, a, s1, r, t, g, o = self.expe2array(experiences)
        m = np.array([self.env.obj2mask(o[k]) for k in range(self.batch_size)])
        inputs = [s0, a, g, m]
        targets = self.compute_targets(s1, g, m, r, t, o)
        weights = np.ones(shape=a.shape)
        return inputs, targets, weights

    def expe2array(self, experiences):
        exp = [np.array(experiences[name]) for name in self.names]
        return exp

    def compute_targets(self, s1, g, m, r, t, o):
        a = self.critic.bestAction_model.predict_on_batch([s1, g, m])
        q = self.critic.target_qValue_model.predict_on_batch([s1, a, g, m])

        targets = []
        for k in range(len(s1)):
            self.env.freqs_train[o[k]] += 1
            self.env.freqs_train_reward[o[k]] += t[k]
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
            input = self.env.make_input(state)
            action = self.critic.bestAction_model.predict(input, batch_size=1)
            action = action[0, 0]
        return action

    def log(self):

        if self.env_step % self.eval_freq == 0:

            for i, goal in enumerate(self.env_test.test_goals):
                obs = self.env_test.env.reset()
                state0 = np.array(self.env_test.decode(obs))
                mask = self.env_test.obj2mask(goal[1])
                R = 0
                for _ in range(200):
                    input = [np.expand_dims(i, axis=0) for i in [state0, goal[0], mask]]
                    action = self.critic.bestAction_model.predict(input, batch_size=1)[0, 0]
                    state1 = self.env_test.step(action)
                    reward, terminal = self.env_test.eval_exp(state0, action, state1, goal[0], goal[1])
                    R += reward
                    if terminal:
                        break
                    state0 = state1
                self.stats['testR_{}'.format(i)] = R

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
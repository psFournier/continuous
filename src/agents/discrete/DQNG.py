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
from agents.agent import Agent
from buffers import ReplayBuffer, PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule
import random as rnd
from samplers.competenceQueue import CompetenceQueue
import math


class DQNG(Agent):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNG, self).__init__(args, sess, env, env_test, logger)
        self.per = bool(args['per'])
        self.self_imitation = bool(int(args['self_imit']))
        self.tutor_imitation = bool(int(args['tutor_imit']))
        self.train_last_expe = bool(int(args['train_last_expe']))
        self.her = bool(int(args['her']))

        self.critic = CriticDQNG(sess,
                                 s_dim=env.state_dim,
                                 g_dim=env.goal_dim,
                                 num_a=env.action_space.n,
                                 gamma=0.99,
                                 tau=0.001,
                                 learning_rate=0.001)

        self.names = []

        self.loss_qVal = 0
        self.loss_imitation = 0
        self.trajectory = []

    def get_tutor_exp(self, goal):
        for i in range(10):
            state0 = self.env.reset()
            actions = rnd.choice(self.env.trajectories[goal])
            episode = []
            for a in actions:
                state1 = self.env.step(a)
                experience = {'state0': state0.copy(),
                              'action': a,
                              'state1': state1.copy(),
                              'reward': None,
                              'terminal': None,
                              'goal': None}
                state0 = state1
                episode.append(experience)
            self.process_tutor_episode(episode)

    def process_tutor_episode(self, episode):
        reached_goals = []
        for expe in reversed(episode):
            s0, a, s1 = expe['state0'], expe['action'], expe['state1']
            for goal in self.env.goals:
                is_new = goal not in reached_goals
                r, t = self.env.eval_exp(s0, a, s1, goal)
                if is_new and t:
                    reached_goals.append(goal)
                if not is_new or t:
                    new_expe = {'state0': s0,
                                'action': a,
                                'state1': s1,
                                'reward': r,
                                'terminal': t,
                                'goal': goal,
                                'R': None}
                    self.append_tutor_exp(new_expe)

    def append_tutor_exp(self, new_expe):
        pass

    def make_exp(self, state0, action, state1):
        pass

    def process_episode(self):

        R = 0
        T = False
        for expe in reversed(self.trajectory):
            R = R * self.critic.gamma + int(expe['terminal']) - 1
            expe['R'] = R
            self.buffer.append(expe)
            T = T or expe['terminal']
        return R, T

    def preprocess(self, experiences):
        return None, None

    def train(self, exp):

        self.trajectory.append(exp)
        self.train_autonomous(exp)
        if self.tutor_imitation:
            self.train_imitation(exp)
        self.target_train()

    def train_autonomous(self, exp):
        pass

    def train_imitation(self, exp):
        pass

    def compute_targets(self, s1, g, r, t):
        a = self.critic.bestAction_model.predict_on_batch([s1, g])
        q = self.critic.target_qValue_model.predict_on_batch([s1, a, g])

        targets = []
        for k in range(len(s1)):
            self.env.freqs_train[g[k]] += 1
            self.env.freqs_reward[g[k]] += t[k]
            target = r[k] + (1 - t[k]) * self.critic.gamma * q[k]
            if TARGET_CLIP:
                target_clip = np.clip(target, -0.99 / (1 - self.critic.gamma), 0.01)
                targets.append(target_clip)
            else:
                targets.append(target)
        targets = np.array(targets)
        return targets

    def train_critic(self, experiences):

        inputs, targets = self.preprocess(experiences)
        self.loss_qVal, q_values = self.critic.qValue_model.train_on_batch(x=inputs, y=targets)
        td_errors = targets - q_values

        return td_errors

    def reset(self):

        if self.trajectory:
            R, T = self.process_episode()
            self.env.queues[self.env.goal].append((self.env.goal, R, T))
            self.trajectory.clear()

        state = self.env.reset()

        return state

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
                      np.reshape(self.env.goal, (1, self.critic.g_dim[0]))]
            action = self.critic.bestAction_model.predict(inputs)
            action = action[0, 0]
        return action

    @property
    def explore_prop(self):
        return 1

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

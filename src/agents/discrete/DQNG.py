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
from agents import DQN
from buffers import ReplayBuffer, PrioritizedReplayBuffer
import random as rnd
from samplers.competenceQueue import CompetenceQueue
import math


class DQNG(DQN):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNG, self).__init__(args, sess, env, env_test, logger)
        self.beta = float(args['beta'])
        self.critic = CriticDQNG(sess,
                                 s_dim=env.state_dim,
                                 g_dim=env.goal_dim,
                                 num_a=env.action_dim,
                                 gamma=0.99,
                                 tau=0.001,
                                 learning_rate=0.001)
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.tutor_buffer = None
        self.trajectory = []
        if self.tutor_imitation:
            self.get_tutor_exp(goal=3)

    def step(self):

        self.env_step += 1
        self.episode_step += 1
        self.env.steps[self.env.goal] += 1
        self.exp['goal'] = self.env.goal
        self.exp['reward'], self.exp['terminal'] = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())

        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, g = [np.array(experiences[name]) for name in self.names]

            a1 = self.critic.bestAction_model.predict_on_batch([s1, g])
            q = self.critic.target_qValue_model.predict_on_batch([s1, a1, g])
            targets = self.compute_targets(r, t, q)
            weights = np.array([self.env.interests[gi] ** self.beta for gi in g])
            self.critic.qValue_model.train_on_batch(x=[s0, a0, g], y=targets, sample_weight=weights)
            self.critic.target_train()

    def reset(self):

        if self.trajectory:
            R = 0
            for expe in reversed(self.trajectory):
                R += int(expe['reward'])
                self.buffer.append(expe)
            self.env.queues[self.env.goal].append({'step': self.env_step, 'R': R})
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.env.explorations[self.env.goal].value(self.env_step):
            action = np.random.randint(0, self.env.action_space.n)
        else:
            inputs = [np.reshape(state, (1, self.critic.s_dim[0])),
                      np.reshape(self.env.goal, (1, self.critic.g_dim[0]))]
            action = self.critic.bestAction_model.predict(inputs)
            action = action[0, 0]
        return action

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

    def train_imitation(self):
        experiences = self.tutor_buffer.sample(self.batch_size)
        s0, a0, s1, r, t, g = [np.array(experiences[name]) for name in self.names]

        targets_imit = np.zeros((self.batch_size, 1))
        self.critic.margin_model.train_on_batch(x=[s0, a0, g], y=targets_imit)

        a1 = self.critic.bestAction_model.predict_on_batch([s1, g])
        q = self.critic.target_qValue_model.predict_on_batch([s1, a1, g])
        targets_dqn = self.compute_targets(r, t, q)
        self.critic.qValue_model.train_on_batch(x=[s0, a0, g], y=targets_dqn)

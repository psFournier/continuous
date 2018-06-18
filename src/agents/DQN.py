import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import criticDqn
from agents.agent import Agent

class DQN(Agent):

    def __init__(self, args, sess, env, logger, xy_sampler, eps_sampler, buffer):

        super(DQN, self).__init__(args, sess, env, logger, xy_sampler, eps_sampler, buffer)

        self.critic = criticDqn.CriticDQN(sess,
                                         s_dim=env.state_dim,
                                         a_dim=env.action_dim,
                                         gamma=0.99,
                                         tau=0.001,
                                         learning_rate=0.001)

        self.start_epsilon = 1
        self.end_epsilon = 0.1
        self.epsilon = 1

    def one_hot(self, s):
        tab = np.zeros(shape=(1, self.env.observation_space.n))
        tab[0, s] = 1
        return tab

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            q_values = self.critic.model.predict(self.one_hot(state))
            action = np.argmax(q_values[0])
        return action

    def train(self):
        critic_stats = []
        actor_stats = []
        for _ in range(self.ep_steps):
            experiences = self.env.buffer.sample(self.batch_size)
            td_errors, stats = self.train_critic(experiences)
            critic_stats.append(stats)
            if self.env.buffer.beta != 0:
                self.env.buffer.update_priorities(experiences['indices'], np.abs(td_errors[0]))
            actor_stats.append(self.train_actor(experiences))
            self.target_train()
        return np.array(critic_stats), np.array(actor_stats)

    def target_train(self):
        self.actor.target_train()
        self.critic.target_train()
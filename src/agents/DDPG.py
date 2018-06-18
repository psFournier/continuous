import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import actorDdpg, criticDdpg
from agents.agent import Agent

class DDPG(Agent):

    def __init__(self, args, sess, env, logger, xy_sampler, eps_sampler, buffer):

        super(DDPG, self).__init__(args, sess, env, logger, xy_sampler, eps_sampler, buffer)

        self.actor = actorDdpg.ActorDDPG(sess,
                                         s_dim=env.state_dim,
                                         a_dim=env.action_dim,
                                         tau=0.001,
                                         learning_rate=0.0001)

        self.critic = criticDdpg.CriticDDPG(sess,
                                         s_dim=env.state_dim,
                                         a_dim=env.action_dim,
                                         gamma=0.99,
                                         tau=0.001,
                                         learning_rate=0.001)

    def train(self):
        critic_stats = []
        actor_stats = []
        experiences = self.buffer.sample(self.batch_size)
        td_errors = self.train_critic(experiences)
        if self.buffer.beta != 0:
            self.buffer.update_priorities(experiences['indices'], np.abs(td_errors[0]))
        self.train_actor(experiences)
        self.target_train()
        return np.array(critic_stats), np.array(actor_stats)

    def target_train(self):
        self.actor.target_train()
        self.critic.target_train()

    def train_critic(self, experiences):

        actions = self.actor.target_model.predict_on_batch(experiences['state1'])
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        target_q = self.critic.target_model.predict_on_batch([experiences['state1'], actions])
        y_i = []
        for k in range(self.batch_size):
            if TARGET_CLIP:
                target_q[k] = np.clip(target_q[k],
                                      self.env.reward_range[0] / (1 - self.critic.gamma),
                                      self.env.reward_range[1])

            if experiences['terminal'][k]:
                y_i.append(experiences['reward'][k])
            else:
                y_i.append(experiences['reward'][k] + self.critic.gamma * target_q[k])

        targets = np.reshape(y_i, (self.batch_size, 1))
        td_errors = self.critic.train([experiences['state0'],
                                                  experiences['action'],
                                                  targets,
                                                  experiences['weights']])
        return td_errors

    def train_actor(self, experiences):

        a_outs = self.actor.model.predict_on_batch(experiences['state0'])
        q_vals, grads = self.critic.gradients(experiences['state0'], a_outs)
        if INVERTED_GRADIENTS:
            """Gradient inverting as described in https://arxiv.org/abs/1511.04143"""
            low = self.env.action_space.low
            high = self.env.action_space.high
            for d in range(grads[0].shape[0]):
                width = high[d]-low[d]
                for k in range(self.batch_size):
                    if grads[k][d]>=0:
                        grads[k][d] *= (high[d]-a_outs[k][d])/width
                    else:
                        grads[k][d] *= (a_outs[k][d]-low[d])/width
        stats = self.actor.train(experiences['state0'], grads)
        return stats

    def act(self, state):
        # action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)

        action = self.actor.model.predict(np.reshape(state, (1, self.actor.s_dim[0])))
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = action.squeeze()
        return action
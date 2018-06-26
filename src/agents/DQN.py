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
from buffers.replayBuffer import ReplayBuffer
from buffers.prioritizedReplayBuffer import PrioritizedReplayBuffer

class DQN(Agent):

    def __init__(self, args, sess, env, env_test, logger):

        super(DQN, self).__init__(args, sess, env, env_test, logger)

        self.env.buffer = ReplayBuffer(limit=int(1e6),
                                       names=['state0', 'action', 'state1', 'reward', 'terminal'])

        self.critic = criticDqn.CriticDQN(sess,
                                         s_dim=env.state_dim,
                                         num_a=env.action_space.n,
                                         gamma=0.99,
                                         tau=0.001,
                                         learning_rate=0.001)

        self.start_epsilon = 1
        self.end_epsilon = 0.1
        self.epsilon = 1

    def train(self, exp):

        self.env.buffer.append(exp)

        if self.env_step > 3 * self.batch_size:
            experiences = self.env.buffer.sample(self.batch_size)
            loss = self.train_critic(experiences)
            # print(td_errors, loss)
            self.target_train()

    def train_critic(self, experiences):
        states0 = experiences['state0']
        states1 = experiences['state1']
        actions0 = experiences['action']

        actions1 = self.critic.model2.predict_on_batch([states1])
        q = self.critic.target_model1.predict_on_batch([states1, actions1])

        targets = []
        for k in range(self.batch_size):
            target = experiences['reward'][k] + (1 - experiences['terminal'][k]) * self.critic.gamma * q[k]
            targets.append(target)
        targets = np.array(targets)
        loss = self.critic.model1.train_on_batch([states0, actions0], targets)
        return loss


    # def train_critic(self, experiences):
    #     qtmax = self.critic.qtmax([experiences['state1']])[0]
    #     targets = []
    #     for k in range(self.batch_size):
    #         targets.append(experiences['reward'][k] + (1 - experiences['terminal'][k]) * self.critic.gamma * qtmax[k])
    #     td_errors, loss = self.critic.train([experiences['state0'], experiences['action'], targets])
    #     return td_errors, loss

    # def train_critic(self, experiences):
    #     pred0  = self.critic.model.predict_on_batch(experiences['state0'])
    #     pred1_target = self.critic.target_model.predict_on_batch(experiences['state1'])
    #     pred1 = self.critic.model.predict_on_batch(experiences['state1'])
    #     target = pred0.copy()
    #     for k in range(self.batch_size):
    #         if experiences['terminal'][k]:
    #             target[k][experiences['action'][k]] = experiences['reward'][k]
    #             # print('terminal')
    #         else:
    #             # target[k][experiences['action'][k]] = experiences['reward'][k] + self.critic.gamma *\
    #             #                                                                  (np.amax(pred1_target[k]))
    #             target[k][experiences['action'][k]] = experiences['reward'][k] + self.critic.gamma * \
    #                                                                          (pred1_target[k][np.argmax(pred1[k])])
    #
    #     # Update the critic given the targets
    #     loss = self.critic.model.train_on_batch(experiences['state0'], target)
    #     return loss

    def init_targets(self):
        self.critic.target_train()

    def target_train(self):
        self.critic.target_train()

    def act_random(self, state):
        return np.random.randint(0, self.env.action_space.n)

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = self.critic.model2.predict(np.reshape(state, (1, self.critic.s_dim[0])))
            action = action.squeeze()

        if noise:
            if self.epsilon > self.end_epsilon:
                self.epsilon -= (self.start_epsilon - self.end_epsilon)/100000
            # print(self.epsilon)
        return action

    def log(self):
        if self.env_step % self.eval_freq == 0:
            returns = []
            for _ in range(5):
                state = self.env_test.reset()
                r = 0
                terminal = False
                step = 0
                while (not terminal and step < self.ep_steps):
                    action = self.act(state, noise=False)
                    experience = self.env_test.step(action)
                    r += experience['reward']
                    terminal = experience['terminal']
                    state = experience['state1']
                    step += 1
                returns.append(r)
            self.stats['avg_return'] = np.mean(returns)
            self.stats['step'] = self.env_step
            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])
            self.logger.dumpkvs()


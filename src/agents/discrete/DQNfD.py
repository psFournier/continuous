import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os

RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import criticDqnfd
from agents.agent import Agent
from buffers.replayBuffer import ReplayBuffer
from buffers.prioritizedReplayBuffer import PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule


class DQNfD(Agent):
    def __init__(self, args, sess, env, env_test, logger):

        super(DQNfD, self).__init__(args, sess, env, env_test, logger)

        self.env.buffer = PrioritizedReplayBuffer(limit=int(1e6),
                                                  names=['state0', 'action', 'state1', 'reward', 'terminal'], args=args)

        self.critic = criticDqnfd.CriticDQNfD(sess,
                                              s_dim=env.state_dim,
                                              num_a=env.action_space.n,
                                              gamma=0.99,
                                              tau=0.001,
                                              learning_rate=0.001)

        self.exploration = LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1)

        self.epsilon_a = 0.001
        self.epsilon_d = 1.

    def train(self, exp):

        self.env.buffer.append(exp)

        if self.env_step > 3 * self.batch_size:
            experiences = self.env.buffer.sample(self.batch_size, self.env_step)
            loss, td_errors = self.train_critic(experiences)
            td_errors = np.abs(td_errors)
            # TODO: add e_a e_d
            self.env.buffer.update_priorities(experiences['indices'], td_errors)
            self.target_train()

    def train_critic(self, experiences):
        states0 = experiences['state0']
        states1 = experiences['state1']
        actions0 = experiences['action']
        weights = experiences['weights'].squeeze()

        actions1 = self.critic.model2.predict_on_batch([states1])
        qvals, margins = self.critic.target_model1.predict_on_batch([states1, actions1])

        targets = []
        for k in range(self.batch_size):
            target = experiences['reward'][k] + (1 - experiences['terminal'][k]) * self.critic.gamma * qvals[k]
            targets.append(target)
        targets = np.array(targets)
        loss, td_errors = self.critic.model1.train_on_batch(x=[states0, actions0],
                                                            y=[targets, targets],
                                                            sample_weight=[weights, weights])
        return loss, td_errors

    def init_targets(self):
        self.critic.target_train()

    def target_train(self):
        self.critic.target_train()

    def act_random(self, state):
        return np.random.randint(0, self.env.action_space.n)

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.exploration.value(self.env_step):
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = self.critic.model2.predict(np.reshape(state, (1, self.critic.s_dim[0])))
            action = action.squeeze()
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

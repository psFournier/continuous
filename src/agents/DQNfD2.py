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


class DQNfD2(Agent):
    def __init__(self, args, sess, env, env_test, Q_tutor, logger):

        super(DQNfD2, self).__init__(args, sess, env, env_test, logger)

        self.env.buffer = PrioritizedReplayBuffer(limit=int(1e6),
                                                  names=['state0', 'action', 'state1', 'origin'],
                                                  args=args)

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

        self.Q_tutor = Q_tutor
        self.init_buffer()

    def init_buffer(self):
        for _ in range(10):
            state0 = self.env.reset(goal=1)
            step = 0
            while step < self.ep_steps:
                action = np.argmax(self.Q_tutor[tuple(state0)][self.env.goal])
                state1 = self.env.step(action)
                exp = self.make_exp(state0, action, state1)
                exp['origin'] = 1
                self.env.buffer.append(exp)
                state0 = state1
                step += 1
                if exp['terminal']:
                    break

    def make_exp(self, state0, action, state1):
        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'reward': reward,
                      'terminal': terminal}

        return experience

    def train(self, exp):
        exp['origin'] = 0
        self.env.buffer.append(exp)
        self.env.episode_exp.append(exp)

        if self.env.buffer.nb_entries > 3 * self.batch_size:
            experiences = self.env.buffer.sample(self.batch_size, self.env_step)
            loss, td_errors = self.train_critic(experiences)
            self.env.buffer.update_priorities(experiences['indices'], td_errors)
            self.target_train()

    def train_critic(self, experiences):

        states0 = experiences['state0']
        actions = experiences['action']
        states1 = experiences['state1']
        weights = experiences['weights']
        origins = experiences['origin']
        goals = np.reshape([self.env.goal] * self.batch_size, (self.batch_size,1))

        states0 = np.append(states0, goals, axis=1)
        states1 = np.append(states1, goals, axis=1)

        actions1 = self.critic.model2.predict_on_batch([states1])
        qvals, margins = self.critic.target_model1.predict_on_batch([states1, actions1])

        targets = []
        for k in range(self.batch_size):
            reward, terminal = self.env.eval_exp(states0[k], actions[k], states1[k], self.env.goal)
            target = reward + (1 - terminal) * self.critic.gamma * qvals[k]
            targets.append(target)
        targets = np.array(targets)

        weights_qlearning = weights * (goals != 2)
        weights_largemargin = weights * origins
        sample_weights = [weights_qlearning.squeeze(), weights_largemargin.squeeze()]

        loss, td_errors = self.critic.model1.train_on_batch(x=[states0, actions],
                                                            y=[targets, targets],
                                                            sample_weight=sample_weights)
        for k in range(self.batch_size):
            td_errors[k] = self.epsilon_a if origins[k] == 0 else self.epsilon_d

        return loss, td_errors

    def init_targets(self):
        self.critic.target_train()

    def target_train(self):
        self.critic.target_train()

    def act_random(self, state):
        return np.random.randint(0, self.env.action_space.n)

    def act(self, state, noise=False):
        if noise and self.env.goal != 2 and np.random.rand(1) < self.exploration.value(self.env_step):
            action = np.random.randint(0, self.env.action_space.n)
        else:
            state = np.append(state, [self.env.goal])
            action = self.critic.model2.predict(np.reshape(state, (1, self.critic.s_dim[0])))
            action = action[0,0]
        return action

    def log(self):
        if self.env_step % self.eval_freq == 0:
            # returns = []
            # for _ in range(5):
            #     state = self.env_test.reset()
            #     r = 0
            #     terminal = False
            #     step = 0
            #     while (not terminal and step < self.ep_steps):
            #         action = self.act(state, noise=False)
            #         experience = self.env_test.step(action)
            #         r += experience['reward']
            #         terminal = experience['terminal']
            #         state = experience['state1']
            #         step += 1
            #     returns.append(r)
            #
            # self.stats['avg_return'] = np.mean(returns)

            self.stats['step'] = self.env_step
            sampler_stats = self.env.sampler.stats()
            for key, val in sampler_stats.items():
                self.stats[key] = val
            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])
            self.logger.dumpkvs()


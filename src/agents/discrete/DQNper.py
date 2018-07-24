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
from buffers.prioritizedReplayBuffer import PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule



class DQNper(Agent):

    def __init__(self, args, sess, env, env_test, logger):

        super(DQNper, self).__init__(args, sess, env, env_test, logger)

        self.env.buffer = PrioritizedReplayBuffer(limit=int(1e6),
                                                  names=['state0', 'action', 'state1', 'reward', 'terminal', 'goal'], args=args)

        self.critic = criticDqn.CriticDQN(sess,
                                          s_dim=env.state_dim,
                                          g_dim=env.goal_dim,
                                          num_a=env.action_space.n,
                                          gamma=0.99,
                                          tau=0.001,
                                          learning_rate=0.001)

        self.exploration = LinearSchedule(schedule_timesteps=int(10000),
                                          initial_p=1.0,
                                          final_p=.1)

    def make_exp(self, state0, action, state1):
        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'reward': reward,
                      'terminal': terminal,
                      'goal': self.env.goal}

        return experience

    def train(self, exp):

        self.env.buffer.append(exp)
        self.env.episode_exp.append(exp)

        if self.env_step > 3 * self.batch_size:
            experiences = self.env.buffer.sample(self.batch_size, self.env_step)
            loss, td_errors = self.train_critic(experiences)
            td_errors = np.abs(td_errors)
            self.env.buffer.update_priorities(experiences['indices'], td_errors)
            self.target_train()

    def train_critic(self, experiences):
        states0 = experiences['state0']
        states1 = experiences['state1']
        actions0 = experiences['action']
        goals = experiences['goal']
        weights = experiences['weights'].squeeze()

        actions1 = self.critic.bestAction.predict_on_batch([states1, goals])
        q = self.critic.target_qValue.predict_on_batch([states1, actions1, goals])

        targets = []
        for k in range(self.batch_size):
            target = experiences['reward'][k] + (1 - experiences['terminal'][k]) * self.critic.gamma * q[k]
            targets.append(target)
        targets = np.array(targets)
        loss, td_errors = self.critic.qValue.train_on_batch(x=[states0, actions0, goals],
                                                            y=targets,
                                                            sample_weight=weights)
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
            inputs = [np.reshape(state, (1, self.critic.s_dim[0])),
                      np.reshape(self.env.goal, (1, self.critic.g_dim[0]))]
            action = self.critic.bestAction.predict(inputs)
            action = action[0, 0]
        return action

    def log(self):
        if self.env_step % self.eval_freq == 0:
            comp_stats = self.env.stats()
            for key, val in comp_stats.items():
                self.stats[key] = val
            self.stats['step'] = self.env_step
            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])
            self.logger.dumpkvs()


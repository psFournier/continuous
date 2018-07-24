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
from buffers import ReplayBuffer, PrioritizedReplayBuffer
from utils.linearSchedule import LinearSchedule



class DQN(Agent):

    def __init__(self, args, sess, env, env_test, logger):

        super(DQN, self).__init__(args, sess, env, env_test, logger)
        self.per_alpha = float(args['per_alpha'])
        self.her = args['her']
        names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal']
        if self.her != 'no':
            names.append('future_goals')
        if self.per_alpha != 0:
            self.env.buffer = PrioritizedReplayBuffer(limit=int(1e4),
                                                      names=names, args=args)
        else:
            self.env.buffer = ReplayBuffer(limit=int(1e4),
                                           names=names)

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

        self.env.episode_exp.append(exp)

        if self.env_step > 10 * self.batch_size:
            experiences = self.env.buffer.sample(self.batch_size, self.env_step)
            loss, td_errors = self.train_critic(experiences)
            if self.per_alpha != 0:
                self.env.buffer.update_priorities(experiences['indices'], td_errors)
            self.target_train()

    def train_critic(self, experiences):

        states0 = experiences['state0']
        actions0 = experiences['action']
        states1 = experiences['state1']
        goals = experiences['goal']
        rewards = experiences['reward']
        terminal = experiences['terminal']

        weights = np.array([(s0 != s1).any() for s0,s1 in zip(states0, states1)])
        inputs = [np.array(states0), np.array(actions0), np.array(states1)]
        targets = np.zeros((self.batch_size, 1))
        loss_margin = self.critic.margin_model.train_on_batch(x=inputs,
                                                              y=targets,
                                                              sample_weight=weights)

        if self.her != 'no':
            for k in range(self.batch_size):
                for alt_goal in experiences['future_goals'][k]:
                    goals.append(alt_goal)
                    states0.append(states0[k])
                    actions0.append(actions0[k])
                    states1.append(states1[k])
                    r, t = self.env.eval_exp(states0[k], actions0[k], states1[k], alt_goal)
                    rewards.append(r)
                    terminal.append(t)

        states0 = np.array(states0)
        actions0 = np.array(actions0)
        states1 = np.array(states1)
        goals = np.array(goals)
        rewards = np.array(rewards)
        terminal = np.array(terminal)

        # if self.per_alpha != 0:
        #     weights = experiences['weights'].squeeze()
        # else:
        #     weights = np.ones(shape=(self.batch_size,1)).squeeze()

        actions1 = self.critic.bestAction_model.predict_on_batch([states1, goals])
        q = self.critic.target_qValue_model.predict_on_batch([states1, actions1, goals])

        targets = []
        for k in range(len(states0)):
            target = rewards[k] + (1 - terminal[k]) * self.critic.gamma * q[k]
            targets.append(target)
        targets = np.array(targets)

        loss_qValue, td_errors = self.critic.qValue_model.train_on_batch(x=[states0, actions0, goals],
                                                            y=targets)

        return loss_qValue, td_errors

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
            action = self.critic.bestAction_model.predict(inputs)
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


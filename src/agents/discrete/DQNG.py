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
        self.per = bool(args['per']) and False
        self.self_imitation = bool(args['self_imit']) and False
        self.tutor_imitation = bool(args['tutor_imit']) and False
        self.her = bool(args['her']) and False
        self.theta = float(args['theta'])
        # if self.per_alpha != 0:
        #     self.env.buffer = PrioritizedReplayBuffer(limit=int(1e5),
        #                                               names=names, args=args)
        # else:
        #     self.env.buffer = ReplayBuffer(limit=int(1e5),
        #                                    names=names)
        self.critic = CriticDQNG(sess,
                                 s_dim=env.state_dim,
                                 g_dim=env.goal_dim,
                                 num_a=env.action_space.n,
                                 gamma=0.99,
                                 tau=0.001,
                                 learning_rate=0.001)



        self.loss_qVal = 0
        self.loss_imitation = 0
        self.td_errors = {goal: 0 for goal in self.env.goals}
        self.q_values = {goal: 0 for goal in self.env.goals}
        names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal', 'R']
        self.buffers = {goal: ReplayBuffer(limit=int(1e5), names=names) for goal in self.env.goals}
        self.buffers['tutor'] = ReplayBuffer(limit=int(1e2), names=names)
        self.queues = {goal: CompetenceQueue() for goal in self.env.goals}
        self.interests = []
        self.freqs = {goal: 0 for goal in self.env.goals}
        self.explorations = {goal: LinearSchedule(schedule_timesteps=int(10000),
                                           initial_p=1.0,
                                           final_p=.1) for goal in self.env.goals}

        self.trajectory = []

        if self.tutor_imitation:
            self.get_tutor_exp(goal=3)

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
            self.process_episode(episode)


    def process_tutor_episode(self, episode):
        pass

    def process_episode(self, episode):

        assert episode[0]['goal'] is not None
        R = 0
        for expe in reversed(episode):
            R = R * self.critic.gamma + expe['reward']
            expe['R'] = R
            self.buffers[expe['goal']].append(expe)
        #
        # R = 0
        # alt_goals = []
        # Rs = [0]
        #
        # if self.self_imitation:
        #     for expe in reversed(episode):
        #         R = R * self.critic.gamma + expe['reward']
        #         expe['R'] = R
        #         self.buffers[expe['goal']].append(expe)
        #
        # if self.her:
        #     alt_goals = []
        #     for expe in reversed(episode):
        #
        #         for i, goal in enumerate(alt_goals):
        #             r, t = self.env.eval_exp(expe['state0'], expe['action'], expe['state1'], goal)
        #             new_expe = {'state0': expe['state0'],
        #                         'action': expe['action'],
        #                         'state1': expe['state1'],
        #                         'reward': r,
        #                         'terminal': t,
        #                         'goal': goal,
        #                         'R': Rs[1 + i]}
        #
        #         for goal in self.env.goals:
        #             if goal != expe['goal']:
        #                 r, t = self.env.eval_exp(expe['state0'], expe['action'], expe['state1'], goal)
        #                 if t:
        #                     alt_goals.append(goal)
        #
        #
        #         if tutor:
        #             self.buffers['tutor'].append(new_expe)
        #         else:
        #             self.buffers[goal].append(new_expe)

    def make_exp(self, state0, action, state1):
        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'reward': reward,
                      'terminal': terminal,
                      'goal': self.env.goal}

        return experience

    def sample_goal(self):

        CPs = [abs(q.CP) for q in self.queues.values()]
        maxcp = max(CPs)

        if maxcp > 10:
            self.interests = [math.pow(cp / maxcp, self.theta) + 0.0001 for cp in CPs]
        else:
            self.interests = [math.pow(1 - q.T_mean, self.theta) + 0.0001 for q in self.queues.values()]

        sum = np.sum(self.interests)
        mass = np.random.random() * sum
        idx = 0
        s = self.interests[0]
        while mass > s:
            idx += 1
            s += self.interests[idx]
        goal = self.env.goals[idx]

        return goal

    def train(self, exp):
        self.trajectory.append(exp)
        buffer = self.buffers[self.env.goal]
        if buffer.nb_entries > self.batch_size:
            experiences = buffer.sample(self.batch_size)
            td_errors = self.train_critic(experiences)
            if self.per:
                self.env.buffer.update_priorities(experiences['indices'], td_errors)
            self.target_train()

    def train_critic(self, experiences):

        states0 = np.array(experiences['state0'])
        actions0 = np.array(experiences['action'])
        states1 = np.array(experiences['state1'])
        goals = np.array(experiences['goal'])
        rewards = np.array(experiences['reward'])
        terminal = np.array(experiences['terminal'])
        returns = np.array(experiences['R'])

        if self.self_imitation:
            inputs = [np.array(states0),
                      np.array(actions0),
                      np.array(goals),
                      np.array(returns),
                      np.zeros((self.batch_size,1))]
            targets = np.zeros((self.batch_size, 1))
            self.loss_imitation = self.critic.margin_model.train_on_batch(x=inputs,
                                                                  y=targets)
        # if self.per_alpha != 0:
        #     weights = experiences['weights'].squeeze()
        # else:
        #     weights = np.ones(shape=(self.batch_size,1)).squeeze()

        actions1 = self.critic.bestAction_model.predict_on_batch([states1, goals])
        q = self.critic.target_qValue_model.predict_on_batch([states1, actions1, goals])

        targets = []
        for k in range(len(states0)):
            target = rewards[k] + (1 - terminal[k]) * self.critic.gamma * q[k]
            if TARGET_CLIP:
                target_clip = np.clip(target, -0.99 / (1 - self.critic.gamma), 0.01)
                targets.append(target_clip)
            else:
                targets.append(target)
        targets = np.array(targets)

        self.loss_qVal, q_values = self.critic.qValue_model.train_on_batch(x=[states0, actions0, goals],
                                                            y=targets)
        td_errors = targets - q_values


        for goal in range(len(self.env.goals)):
            if goal in goals:
                self.td_errors[goal] = np.mean(td_errors[np.where(goals == goal)])
                self.q_values[goal] = np.mean(q_values[np.where(goals == goal)])

        return td_errors

    def reset(self):

        if self.trajectory:
            self.process_episode(self.trajectory)
            self.queues[self.env.goal].append((self.env.goal,
                                               self.trajectory[0]['R'],
                                               int(self.trajectory[-1]['terminal'])))
            self.trajectory.clear()

        self.env.goal = self.sample_goal()
        self.freqs[self.env.goal] += 1

        state = self.env.reset()

        return state

    def init_targets(self):
        self.critic.target_init()

    def target_train(self):
        self.critic.target_train()

    def act_random(self, state):
        return np.random.randint(0, self.env.action_space.n)

    def act(self, state, noise=False):
        if noise and np.random.rand(1) < self.explorations[self.env.goal].value(self.env_step):
            action = np.random.randint(0, self.env.action_space.n)
        else:
            inputs = [np.reshape(state, (1, self.critic.s_dim[0])),
                      np.reshape(self.env.goal, (1, self.critic.g_dim[0]))]
            action = self.critic.bestAction_model.predict(inputs)
            action = action[0, 0]
        return action

    def log(self):
        if self.env_step % self.eval_freq == 0:

            for goal in self.env.goals:
                self.stats['R_{}'.format(goal)] = float("{0:.3f}".format(self.queues[goal].R_mean))
                self.stats['T_{}'.format(goal)] = float("{0:.3f}".format(self.queues[goal].T_mean))
                self.stats['CP_{}'.format(goal)] = float("{0:.3f}".format(self.queues[goal].CP))
                self.stats['F_{}'.format(goal)] = float("{0:.3f}".format(self.freqs[goal]))
                self.stats['I_{}'.format(goal)] = float("{0:.3f}".format(self.interests[goal]))

            self.stats['step'] = self.env_step
            self.stats['loss_qVal'] = self.loss_qVal

            for goal in self.env.goals:
                self.stats['qval_{}'.format(goal)] = self.q_values[goal]
                self.stats['tde_{}'.format(goal)] = self.td_errors[goal]

            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])

            self.logger.dumpkvs()

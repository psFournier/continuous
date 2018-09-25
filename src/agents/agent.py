import time
import numpy as np
import tensorflow as tf
import json_tricks

import os
RENDER_TRAIN = False
TARGET_CLIP = False
INVERTED_GRADIENTS = False

class Agent():
    def __init__(self, args, env, env_test, logger, buffer=None):

        self.env = env
        self.env_test = env_test
        self.buffer = buffer
        self.logger = logger
        self.log_dir = args['--log_dir']
        self.ep_steps = int(args['--ep_steps'])
        self.eval_freq = int(args['--eval_freq'])
        self.batch_size = int(args['--batchsize'])
        self.max_steps = int(args['--max_steps'])
        self.env_step = 0
        self.episode_step = 0
        self.stats = {}
        self.exp = {}
        self.metrics = {}
        self.imitMetrics = {}
        self.trajectory = []

    def run(self):
        self.exp['state0'] = self.reset()
        try:
            while self.env_step < self.max_steps:

                if RENDER_TRAIN: self.env.render(mode='human')
                self.exp['action'] = self.act(self.exp['state0'])
                self.exp = self.env.step(self.exp)
                self.trajectory.append(self.exp.copy())
                self.train()
                self.env_step += 1
                self.episode_step += 1
                self.exp['state0'] = self.exp['state1']

                if (self.exp['terminal'] or self.episode_step >= self.ep_steps):
                    self.exp['state0'] = self.reset()

                self.log()

        except KeyboardInterrupt:
            print("Keybord interruption")
            self.save_regions()
            self.save_policy()

        self.save_regions()
        self.save_policy()

    def reset(self):
        return self.env.reset()

    def train(self):
        pass

    def act(self, state, mode='train'):
        pass

    def log(self):

        if self.env_step % self.eval_freq == 0:
            R_mean = []
            for g, goal in enumerate(self.env_test.goals):
                exp = {}
                exp['state0'] = self.env_test.reset(goal=g)
                exp['terminal'] = False
                i = 0
                R = 0
                while not exp['terminal'] and i < self.ep_steps:
                    exp['action'] = self.act(self.exp['state0'], mode='test')
                    exp = self.env_test.step(self.exp)
                    self.exp['state0'] = self.exp['state1']
                    i += 1
                    R += exp['reward']
                R_mean.append(R)
                self.stats['R_{}'.format(goal)] = float("{0:.3f}".format(R))
            self.stats['R'] = float("{0:.3f}".format(np.mean(R_mean)))

            wrapper_stats = self.env.get_stats()
            for key, val in wrapper_stats.items():
                self.stats[key] = val

            self.stats['step'] = self.env_step
            for metric, val in self.metrics.items():
                self.stats[metric] = val / self.eval_freq
                self.metrics[metric] = 0
            for metric, val in self.imitMetrics.items():
                self.stats[metric] = val / self.eval_freq
                self.imitMetrics[metric] = 0

            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])

            self.logger.dumpkvs()

    def save_regions(self):
        pass

    def save_policy(self):
        pass
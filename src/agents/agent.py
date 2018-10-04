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
        t0 = time.time()
        self.exp['s0'] = self.reset()
        try:
            while self.env_step < self.max_steps:

                if 0: self.env.render(mode='human')
                self.exp['a'] = self.act(self.exp['s0'])
                self.exp = self.env.step(self.exp)
                self.trajectory.append(self.exp.copy())
                self.train()
                self.env_step += 1
                self.episode_step += 1
                self.exp['s0'] = self.exp['s1']

                if self.episode_step >= self.ep_steps:
                    t1 = time.time()
                    # print(t1 - t0)
                    t0 = t1
                    self.exp['s0'] = self.reset()

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

            exp = {}
            exp['s0'] = self.env_test.reset(goal=np.array([0.02]))
            if 0:
                self.env_test.render(mode='human')
                self.env_test.unwrapped.viewer._run_speed = 0.125
            exp['t'] = False
            R = 0
            for i in range(self.ep_steps):
                if 0:
                    self.env_test.render(mode='human')
                exp['a'] = self.act(exp['s0'], mode='test')
                exp = self.env_test.step(exp)
                exp['s0'] = exp['s1']
                R += exp['r']
            self.stats['R'] = float("{0:.3f}".format(R))

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
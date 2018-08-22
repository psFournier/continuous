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
        self.log_dir = args['log_dir']
        self.ep_steps = int(args['episode_steps'])
        self.eval_freq = int(args['eval_freq'])
        self.batch_size = 64
        self.max_steps = int(args['max_steps'])
        self.env_step = 0
        self.episode_step = 0
        self.stats = {}
        self.exp = {}

    def run(self):
        self.exp['state0'] = self.reset()
        try:
            while self.env_step < self.max_steps:

                if RENDER_TRAIN: self.env.render(mode='human')
                self.exp['action'] = self.act(self.exp['state0'], noise=True)
                self.exp['state1'] = self.env.step(self.exp['action'])
                self.step()
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

    def step(self):
        pass

    def act(self, state, noise=True):
        pass

    def log(self):

        if self.env_step % self.eval_freq == 0:
            # for i, goal in enumerate(self.env_test.test_goals):
            #     obs = self.env_test.env.reset()
            #     state0 = np.array(self.env_test.decode(obs))
            #     mask = self.env_test.obj2mask(goal[1])
            #     R = 0
            #     for _ in range(200):
            #         input = [np.expand_dims(i, axis=0) for i in [state0, goal[0], mask]]
            #         action = self.critic.bestAction_model.predict(input, batch_size=1)[0, 0]
            #         state1 = self.env_test.step(action)
            #         reward, terminal = self.env_test.eval_exp(state0, action, state1, goal[0], goal[1])
            #         R += reward
            #         if terminal:
            #             break
            #         state0 = state1
            #     self.stats['testR_{}'.format(i)] = R
            wrapper_stats = self.env.get_stats()
            self.stats['step'] = self.env_step

            for key, val in wrapper_stats.items():
                self.stats[key] = val

            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])

            self.logger.dumpkvs()

    def save_regions(self):
        pass

    def save_policy(self):
        pass
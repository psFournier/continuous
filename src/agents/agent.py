import time
import numpy as np
import tensorflow as tf
import json_tricks
import pickle

import os
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True

class Agent():
    def __init__(self, args, sess, env, logger, xy_sampler, eps_sampler, buffer):

        self.sess = sess
        self.env = env
        self.xy_sampler = xy_sampler
        self.eps_sampler = eps_sampler
        self.buffer = buffer
        self.logger = logger
        self.log_dir = args['log_dir']
        self.ep_steps = args['episode_steps']
        self.eval_freq = args['eval_freq']
        self.batch_size = 64
        self.max_steps = args['max_steps']
        self.her_xy = args['her_xy']
        self.her_eps = args['her_eps']

        self.env_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.goal_reached = 0
        self.stats = {}
        self.episode_exp = []

    def init_variables(self):
        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if not hasattr(v,
                           '_keras_initialized') or not v._keras_initialized:
                uninitialized_variables.append(v)
                v._keras_initialized = True
        self.sess.run(tf.variables_initializer(uninitialized_variables))

    def act(self, state):
        action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)
        return action

    def run(self):
        self.init_variables()
        self.start_time = time.time()
        self.target_train()

        region = self.xy_sampler.sample(rnd_prop=1)
        self.env.goal = region.sample().flatten()

        self.env.epsilon = self.eps_sampler.sample()

        state0 = self.env.reset()
        try:
            while self.env_step < self.max_steps:

                if RENDER_TRAIN: self.env.render(mode='human')

                action = self.act(state0)

                state1, reward, terminal, _ = self.env.step(action)

                exp = {'state0': state0,
                       'action': action,
                       'state1': state1,
                       'reward': reward,
                       'terminal': terminal}
                self.episode_exp.append(exp)
                self.buffer.append(exp)

                self.episode_reward += reward
                self.env_step += 1
                self.episode_step += 1
                state0 = state1

                if self.env_step > 3 * self.batch_size:
                    self.train()

                if (terminal or self.episode_step >= self.ep_steps):

                    self.episode += 1

                    if self.episode > 0:
                        self.xy_sampler.append((self.env.goal, int(terminal)))
                        self.eps_sampler.append((self.env.epsilon, int(terminal)))

                    if self.her_xy != 'no' or self.her_eps != 'no':
                        virtual_exp = self.env.hindsight(self.episode_exp, self.her_xy, self.her_eps)
                        for exp in virtual_exp:
                            self.buffer.append(exp)

                    region = self.xy_sampler.sample(rnd_prop=max(0.1, 1 - self.episode // 200))
                    self.env.goal = region.sample().flatten()

                    self.env.epsilon = self.eps_sampler.sample()

                    state0 = self.env.reset()

                    self.episode_step = 0
                    self.episode_reward = 0

                self.log()

        except KeyboardInterrupt:
            print("Keybord interruption")
            self.save_regions()

        self.save_regions()

    def log(self):
        if self.env_step % self.eval_freq == 0:
            returns = []
            for _ in range(10):
                state = self.env.reset()
                r = 0
                for _ in range(self.ep_steps):
                    action = self.act(state)
                    state, reward, terminal, _ = self.env.step(action[0])
                    r += reward
                returns.append(r)
            self.stats['avg_return'] = np.mean(returns)
            self.stats['step'] = self.env_step
            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])
            self.logger.dumpkvs()

    def save_regions(self):
        with open(os.path.join(self.log_dir, 'regionTree.pkl'), 'wb') as output:
            pickle.dump(self.env.regionTree, output)

    def train(self):
        pass

    def target_train(self):
        pass
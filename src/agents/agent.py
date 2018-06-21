import time
import numpy as np
import tensorflow as tf
import json_tricks

import os
RENDER_TRAIN = False
TARGET_CLIP = False
INVERTED_GRADIENTS = False

class Agent():
    def __init__(self, args, sess, env, env_test, logger, buffer=None):

        self.sess = sess
        self.env = env
        self.env_test = env_test
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
        self.goal_reached = 0
        self.stats = {}

    def init_variables(self):
        variables = tf.global_variables()
        uninitialized_variables = []
        for v in variables:
            if not hasattr(v,
                           '_keras_initialized') or not v._keras_initialized:
                uninitialized_variables.append(v)
                v._keras_initialized = True
        self.sess.run(tf.variables_initializer(uninitialized_variables))

    def run(self):
        self.init_variables()
        self.start_time = time.time()
        self.init_targets()

        state0 = self.env.reset()
        try:
            while self.env_step < self.max_steps:

                if RENDER_TRAIN: self.env.render(mode='human')

                if self.env_step < self.env.exploration_steps:
                    action = self.act_random(state0)
                else:
                    action = self.act(state0, noise=True)

                experience = self.env.step(action)

                self.train(experience)

                self.env_step += 1
                self.episode_step += 1
                state0 = experience['state1']

                if (experience['terminal'] or self.episode_step >= self.ep_steps):

                    self.episode += 1
                    state0 = self.env.reset()
                    self.episode_step = 0

                self.log()

        except KeyboardInterrupt:
            print("Keybord interruption")
            self.save_regions()

        self.save_regions()

    def act_random(self, state):
        action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)
        return action

    def act(self, state, noise=True):
        pass

    def log(self):
        pass

    def save_regions(self):
        pass

    def train(self, exp):
        pass

    def init_targets(self):
        pass

    def target_train(self):
        pass

    def sample_goal(self):
        pass

    def sample_epsilon(self):
        pass

    def train_goal(self):
        pass

    def hindsight(self):
        pass
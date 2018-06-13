import time
import numpy as np
import tensorflow as tf

RENDER_TRAIN = False
TARGET_CLIP = False
INVERTED_GRADIENTS = True

class TD3():
    def __init__(self,
                 sess,
                 actor,
                 critic,
                 env,
                 logger_step,
                 logger_episode,
                 ep_steps,
                 max_steps,
                 eval_freq):

        self.sess = sess
        self.actor = actor
        self.critic = critic
        self.env = env
        self.logger_step = logger_step
        self.logger_episode = logger_episode
        self.step_stats = {}
        self.episode_stats = {}

        self.ep_steps = ep_steps
        self.eval_freq = eval_freq
        self.batch_size = 100
        self.max_steps = max_steps

        self.env_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_reward = 0
        self.goal_reached = 0

    def train(self):
        critic_stats = []
        actor_stats = []
        batch_idxs, experiences = self.env.buffer.sample(self.batch_size)
        critic_stats.append(self.train_critic(experiences))
        if self.env_step % 2 == 0:
            actor_stats.append(self.train_actor(experiences))
            self.actor.target_train()
            self.critic.target_train()
        return np.array(critic_stats), np.array(actor_stats)

    def train_critic(self, experiences):

        # Calculate targets
        actions = self.actor.target_model.predict_on_batch(experiences['state1'])
        actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)
        noise = np.clip(np.random.normal(0., 0.2, size=actions.shape), -0.5, 0.5)
        actions = actions + noise
        target_q1 = self.critic.target_model1.predict_on_batch([experiences['state1'], actions])
        target_q2 = self.critic.target_model2.predict_on_batch([experiences['state1'], actions])
        target_q = np.minimum(target_q1, target_q2)
        y_i = []
        for k in range(self.batch_size):
            if TARGET_CLIP:
                target_q[k] = np.clip(target_q[k],
                                      self.env.reward_range[0] / (1 - self.critic.gamma),
                                      self.env.reward_range[1])

            if experiences['terminal'][k]:
                y_i.append(experiences['reward'][k])
            else:
                y_i.append(experiences['reward'][k] + self.critic.gamma * target_q[k])

        # Update the critics given the targets
        self.critic.model1.train_on_batch([experiences['state0'], experiences['action']], np.reshape(y_i, (self.batch_size, 1)))
        self.critic.model2.train_on_batch([experiences['state0'], experiences['action']], np.reshape(y_i, (self.batch_size, 1)))
        critic_stats1 = self.sess.run(self.critic.stat_ops, feed_dict={
            self.critic.state1: experiences['state0'],
            self.critic.action1: experiences['action']})

        return critic_stats1

    def train_actor(self, experiences):

        a_outs = self.actor.model.predict_on_batch(experiences['state0'])
        q_vals, grads = self.critic.gradients(experiences['state0'], a_outs)
        if INVERTED_GRADIENTS:
            """Gradient inverting as described in https://arxiv.org/abs/1511.04143"""
            low = self.env.action_space.low
            high = self.env.action_space.high
            for d in range(grads[0].shape[0]):
                width = high[d]-low[d]
                for k in range(self.batch_size):
                    if grads[k][d]>=0:
                        grads[k][d] *= (high[d]-a_outs[k][d])/width
                    else:
                        grads[k][d] *= (a_outs[k][d]-low[d])/width
        stats = self.actor.train(experiences['state0'], grads)
        return stats

    def log(self, stats, logger):
        for key in sorted(stats.keys()):
            logger.logkv(key, stats[key])
        logger.dumpkvs()

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
        self.actor.target_train()
        self.critic.target_train()
        self.start_time = time.time()

        state0 = self.env.reset()

        while self.env_step < 10000:

            if RENDER_TRAIN: self.env.render(mode='human')

            action = np.random.uniform(self.env.action_space.low, self.env.action_space.high)
            state1, reward, terminal, _ = self.env.step(action[0])

            self.episode_reward += reward
            self.env_step += 1
            self.episode_step += 1
            state0 = state1

            if self.env_step > 3 * self.batch_size:
                self.critic_stats, self.actor_stats = self.train()

            if (terminal or self.episode_step >= self.ep_steps):
                self.episode += 1
                if terminal: self.goal_reached += 1
                state0 = self.env.reset()
                self.log_episode_stats()
                self.episode_step = 0
                self.episode_reward = 0

            self.log_step_stats()

        while self.env_step < self.max_steps:

            if RENDER_TRAIN: self.env.render(mode='human')

            action = self.actor.model.predict(np.reshape(state0, (1, self.actor.s_dim[0])))
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

            state1, reward, terminal, _ = self.env.step(action[0])

            self.episode_reward += reward
            self.env_step += 1
            self.episode_step += 1
            state0 = state1

            if self.env_step > 3 * self.batch_size:
                self.critic_stats, self.actor_stats = self.train()

            if (terminal or self.episode_step >= self.ep_steps):
                self.episode += 1
                if terminal: self.goal_reached += 1
                state0 = self.env.reset()
                self.log_episode_stats()
                self.episode_step = 0
                self.episode_reward = 0

            self.log_step_stats()

    def log_step_stats(self):
        if self.env_step % self.eval_freq == 0:
            critic_stats_mean = self.critic_stats.mean(axis=0)
            actor_stats_mean = self.actor_stats.mean(axis=0)
            for name, stat in zip(self.critic.stat_names, critic_stats_mean):
                self.step_stats[name] = stat
            for name, stat in zip(self.actor.stat_names, actor_stats_mean):
                self.step_stats[name] = stat
            self.step_stats['env_step'] = self.env_step

            self.step_stats['accuracy'] = self.goal_reached / 20
            self.log(self.step_stats, self.logger_step)
            self.goal_reached = 0

    def log_episode_stats(self):
        if self.episode % 100 == 0:
            self.episode_stats['Episode'] = self.episode
            if self.env.goal_parameterized:
                self.episode_stats['Goal'] = self.env.goal
            self.episode_stats['Train_reward'] = self.episode_reward
            self.episode_stats['Episode_steps'] = self.episode_step
            self.episode_stats['Duration'] = time.time() - self.start_time
            self.episode_stats['Env_step'] = self.env_step
            memory_stats = self.env.stats()
            for name, stat in memory_stats.items():
                self.episode_stats[name] = stat
            self.log(self.episode_stats, self.logger_episode)
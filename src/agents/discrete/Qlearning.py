import numpy as np
from agents.agent import Agent
import os, pickle

class Qlearning(Agent):

    def __init__(self, args, sess, env, env_test, logger):

        super(Qlearning, self).__init__(args, sess, env, env_test, logger)

        self.Q = np.zeros(shape=env.state_dim + (self.env.action_space.n,))
        self.gamma = 0.99
        self.lr = 0.5

    def make_exp(self, state0, action, state1):

        reward, terminal = self.env.eval_exp(state0, action, state1, self.env.goal)

        experience = {'state0': state0.copy(),
                      'action': action,
                      'state1': state1.copy(),
                      'reward': reward,
                      'terminal': terminal}

        return experience

    def train(self, exp):

        self.env.episode_exp.append(exp)

        state0 = exp['state0']
        action = exp['action']
        state1 = exp['state1']
        target = (exp['reward'] - 1) + (1 - exp['terminal']) * self.gamma * np.max(self.Q[tuple(state1)][self.env.goal])
        self.Q[tuple(state0)][self.env.goal][action] = self.lr * target + \
                                               (1 - self.lr) * self.Q[tuple(state0)][self.env.goal][action]

    def act(self, state, noise=True):
        if noise and np.random.rand() < 0:
            action = np.random.randint(self.env.action_space.n)
        else:
            action = np.argmax(self.Q[tuple(state)][self.env.goal])
        return action

    def hindsight(self):
        virtual_exp = self.env.hindsight()
        for exp in virtual_exp:
            self.buffer.append(exp)

    def save_policy(self):
        dir = os.path.dirname(self.logger.get_dir())
        with open(os.path.join(dir, 'policy.pkl'), 'wb') as output:
            pickle.dump(self.Q, output)

    def log(self):
        if self.env_step % self.eval_freq == 0:
            # return_per_goal = []
            # for goal in self.env.goal_space:
            #     returns = []
            #     for _ in range(10):
            #         state = self.env_test.reset()
            #         self.env_test.prev_state[self.env.goal_idx] = goal
            #         state[self.env.goal_idx] = goal
            #         self.env_test.goal = goal
            #         r = 0
            #         terminal = False
            #         step = 0
            #         while (not terminal and step < self.ep_steps):
            #             action = self.act(state, noise=False)
            #             experience = self.env_test.step(action)
            #             self.train(experience) ## ATTENTION ON TRICHE
            #             r += experience['reward']
            #             terminal = experience['terminal']
            #             state = experience['state1']
            #             step += 1
            #         returns.append(r)
            #     return_per_goal.append(np.mean(returns))

            # self.stats['avg_returns'] = return_per_goal
            self.stats['step'] = self.env_step
            sampler_stats = self.env.sampler.stats()
            for key, val in sampler_stats.items():
                self.stats[key] = val
            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])
            self.logger.dumpkvs()

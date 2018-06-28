import numpy as np
from agents.agent import Agent
from buffers.replayBuffer import ReplayBuffer

class Qlearning_offPolicy(Agent):

    def __init__(self, args, sess, env, env_test, logger):

        super(Qlearning_offPolicy, self).__init__(args, sess, env, env_test, logger)

        self.env.buffer = ReplayBuffer(limit=int(1e6),
                                       names=['state0', 'action', 'state1', 'reward', 'terminal'])
        self.batch_size = 32
        self.gamma = 0.99
        self.lr = 0.1
        self.Q = np.zeros(shape=(env.state_dim + env.action_dim))

    def train(self, exp):

        self.env.buffer.append(exp)

        if self.env_step > 3 * self.batch_size:
            experiences = self.env.buffer.sample(self.batch_size)
            self.train_critic(experiences)

    def train_critic(self, exp):

        for k in range(self.batch_size):
            target = exp['reward'][k][0]
            if not exp['terminal'][k]:
                target += self.gamma * np.max(self.Q[tuple(exp['state1'][k])])
            self.Q[tuple(exp['state0'][k])][exp['action'][k]] = self.lr * target + \
                                                   (1 - self.lr) * self.Q[tuple(exp['state0'][k])][exp['action'][k]]


    def act(self, state, noise=True):
        if noise and np.random.rand() < 0.2:
            action = np.random.randint(self.env.action_space.n)
        else:
            action = np.argmax(self.Q[tuple(state)])
        return action

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
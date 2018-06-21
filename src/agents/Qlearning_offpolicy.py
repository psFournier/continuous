import numpy as np
from agents.agent import Agent
from buffers.replayBuffer import ReplayBuffer

class Qlearning_offPolicy(Agent):

    def __init__(self, args, sess, env, env_test, env_tutor, logger):

        super(Qlearning_offPolicy, self).__init__(args, sess, env, env_test, logger)

        self.env.buffer = ReplayBuffer(limit=int(1e6),
                                       names=['state0', 'action', 'state1', 'reward', 'terminal'])
        self.batch_size = 32
        self.gamma = 0.99
        self.lr = 0.1
        self.lr_tutor = 1

        self.env_tutor = env_tutor
        self.Q_tutor = np.zeros(shape=(env.state_dim + env.action_dim))
        self.train_tutor()
        self.init_buffer()
        self.Q = np.zeros(shape=(env.state_dim + env.action_dim))


    def train_tutor(self):
        for _ in range(1000):
            state = self.env_tutor.reset()
            r = 0
            terminal = False
            step = 0
            while (not terminal and step < self.ep_steps):

                action = self.tutor_act(state, noise=False)
                exp = self.env_tutor.step(action)

                reward = exp['reward'] - 1 #optimistic init
                target = reward
                if not exp['terminal']:
                    target += exp['gamma'] * np.max(self.Q_tutor[tuple(exp['state1'])])
                self.Q_tutor[tuple(exp['state0'])][exp['action']] = self.lr_tutor * target + \
                                                              (1 - self.lr_tutor) * self.Q_tutor[tuple(exp['state0'])][
                                                                  exp['action']]

                r += reward
                terminal = exp['terminal']
                state = exp['state1']
                step += 1

            print(r)

    def init_buffer(self):
        # TODO: mettre des expériences avec Q_tutor dans le replay buffer
        pass

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
        # new_s0 = exp['state0'].copy()
        # new_s0[1] = 1 - new_s0[1]
        # new_s1 = exp['state1'].copy()
        # new_s1[1] = 1 - new_s1[1]
        # r, t, g = self.env.eval_exp(new_s0, exp['action'], new_s1, exp['reward'], exp['terminal'])
        # target = r + g * np.max(self.Q[tuple(new_s1)])
        # self.Q[tuple(new_s0)][exp['action']] = self.lr * target + \
        #                                               (1 - self.lr) * self.Q[tuple(new_s0)][exp['action']]

    def act(self, state, noise=True):
        if noise and np.random.rand() < 0.2:
            action = np.random.randint(self.env.action_space.n)
        else:
            action = np.argmax(self.Q[tuple(state)])
        return action

    def tutor_act(self, state, noise=True):
        if noise and np.random.rand() < 0.2:
            action = np.random.randint(self.env.action_space.n)
        else:
            action = np.argmax(self.Q_tutor[tuple(state)])
        return action

    def hindsight(self):
        virtual_exp = self.env.hindsight()
        for exp in virtual_exp:
            self.buffer.append(exp)

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

#
# grid0 = np.fromfunction(lambda i, j: np.max(Q[env.encode(i, j, 1, 3), :], axis=2), (5, 5), dtype=int)
# grid1 = np.fromfunction(lambda i, j: np.max(Q[env.encode(i, j, 4, 3), :], axis=2), (5, 5), dtype=int)
#
# def updatefig(i):
#     global grid0, grid1
#
#     new_grid0 = np.fromfunction(lambda i, j: np.max(Q[env.encode(i, j, 1, 3), :], axis=2), (5, 5), dtype=int)
#     min0 = new_grid0.min()
#     max0 = new_grid0.max()
#     if max0 - min0 != 0:
#         new_grid0 = (new_grid0 - min0) * 255.0 / (max0 - min0)
#     mat0.set_data(new_grid0)
#     grid0 = new_grid0
#
#     new_grid1 = np.fromfunction(lambda i, j: np.max(Q[env.encode(i, j, 4, 3), :], axis=2), (5, 5), dtype=int)
#     min1 = new_grid1.min()
#     max1 = new_grid1.max()
#     if max1 - min1 != 0:
#         new_grid1 = (new_grid1 - min1) * 255.0 / (max1 - min1)
#     mat1.set_data(new_grid1)
#     title.set_text("episode {}".format(i))
#     grid1 = new_grid1
#
#     return mat0, mat1, title,
#
# fig, axes = plt.subplots(2)
# mat0 = axes[0].imshow(grid0, vmin=0., vmax=255.)
# mat1 = axes[1].imshow(grid1, vmin=0., vmax=255.)
# title = axes[1].text(0.5,0.85, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5},
#                 transform=axes[1].transAxes, ha="center")
#
# ani = animation.FuncAnimation(fig, updatefig, interval=1, blit=True)
# plt.show()

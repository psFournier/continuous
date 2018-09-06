import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import CriticDQN
from agents.agent import Agent
from buffers import ReplayBuffer

class Qoff(Agent):

    def __init__(self, args, env, env_test, logger):
        super(Qoff, self).__init__(args, env, env_test, logger)
        self.args = args
        self.gamma = 0.99
        self.lr = 0.1
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal']
        self.init(args, env)

    def init(self, args ,env):
        self.critic = np.zeros(shape=(5, 5, 4))
        self.buffer = ReplayBuffer(limit=int(1e6),
                                   names=self.names)

    def train(self):
        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, g, m = [exp[name] for name in self.names]
            for k in range(self.batch_size):
                target = r[k] + (1 - t[k]) * self.gamma * np.max(self.critic[tuple(s1[k])])
                self.critic[tuple(s0[k])][a0[k]] = self.lr * target + \
                                                       (1 - self.lr) * self.critic[tuple(s0[k])][a0[k]]

    def act(self, state):
        if np.random.rand() < 0.2:
            action = np.random.randint(self.env.action_space.n)
        else:
            action = np.argmax(self.critic[tuple(state)])
        return action

    def reset(self):

        if self.trajectory:
            self.env.processEp(self.trajectory)
            for expe in reversed(self.trajectory):
                self.buffer.append(expe.copy())
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state



import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import CriticDQN
from agents.agent import Agent
from buffers import ReplayBuffer

class DQN(Agent):

    def __init__(self, args, env, env_test, logger):
        super(DQN, self).__init__(args, env, env_test, logger)
        self.args = args
        self.init(args, env)
        for metric in self.critic.qvalModel.metrics_names:
            self.metrics[metric] = 0

    def init(self, args ,env):
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal']
        if args['--imit'] != '0':
            self.names.append('expVal')
        self.buffer = ReplayBuffer(limit=int(1e6),
                                   names=self.names)
        self.critic = CriticDQN(args, env)

    def train(self):
        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t = [exp[name] for name in self.names]
            temp = np.expand_dims([1], axis=0)

            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1, temp])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1])
            targets_dqn = self.compute_targets(r, t, q)

            if self.args['--imit'] == '0':
                targets = targets_dqn
                inputs = [s0, a0]
            else:
                e = exp['expVal']
                targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
                inputs = [s0, a0, e]
            loss = self.critic.qvalModel.train_on_batch(inputs, targets)

            for i, metric in enumerate(self.critic.qvalModel.metrics_names):
                self.metrics[metric] += loss[i]

            self.critic.target_train()

    def compute_targets(self, r, t, q, clip=True):
        targets = []
        for k in range(self.batch_size):
            target = r[k] + (1 - t[k]) * self.critic.gamma * q[k]
            if clip:
                target_clip = np.clip(target, self.env.minR / (1 - self.critic.gamma), self.env.maxR)
                targets.append(target_clip)
            else:
                targets.append(target)
        targets = np.array(targets)
        return targets

    def reset(self):

        if self.trajectory:
            T = int(self.trajectory[-1]['terminal'])
            R = np.sum([self.env.unshape(exp['reward'], exp['terminal']) for exp in self.trajectory])
            S = len(self.trajectory)
            self.env.processEp(R, S, T)
            if self.args['--imit'] == '0':
                for expe in reversed(self.trajectory):
                    self.buffer.append(expe.copy())
            else:
                E = 0
                for expe in reversed(self.trajectory):
                    if self.trajectory[-1]['terminal']:
                        E = E * self.critic.gamma + expe['reward']
                        expe['expVal'] = E
                    else:
                        expe['expVal'] = -self.ep_steps
                    self.buffer.append(expe.copy())

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def make_input(self, state, t):
        input = [np.reshape(state, (1, self.critic.s_dim[0]))]
        # temp = self.env.explor_temp(t)
        input.append(np.expand_dims([0.5], axis=0))
        return input

    def act(self, state):
        input = self.make_input(state, self.env_step)
        actionProbs = self.critic.actionProbsModel.predict(input, batch_size=1)
        if self.args['--explo'] == '1':
            action = np.random.choice(range(self.env.action_dim), p=actionProbs[0])
        else:
            # eps = self.env.explor_eps()
            eps = 1 + min(float(self.env_step) / 1e4, 1) * (0.1 - 1)
            if np.random.random() < eps:
                action = np.random.choice(range(self.env.action_dim))
            else:
                action = np.argmax(actionProbs[0], axis=0)
        return action



import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from .critic import Critic1
from utils.util import softmax

class Dqn1():
    def __init__(self, args, env, logger):
        self.env = env
        self.logger = logger
        self.log_dir = args['--log_dir']
        self.batch_size = int(args['--batchsize'])
        self.stats = {'qval'+str(f): 0 for f in self.env.feat}
        self.stats['step'] = 0
        self.exp = {}
        self.trajectory = []
        self.critic = Critic1(args, env)

    def reset(self, state):
        self.exp = {}
        self.exp['s0'] = state
        self.exp = self.env.reset(self.exp)
        self.exp['r0'] = self.env.get_r(self.exp['s0'], self.env.g, self.env.vs).squeeze()

    def process_trajectory(self, t):
        self.env.process_trajectory(t)

    def act(self):
        v = self.env.vs[self.env.idx]
        input = [np.expand_dims(i, axis=0) for i in [self.exp['s0'], v, v]]
        qvals = self.critic.qvals(input)[0].squeeze()
        self.stats['qval'+str(self.env.feat[self.env.idx])] += np.mean(np.squeeze(qvals))
        action = np.random.choice(range(self.env.action_dim), p=softmax(qvals, theta=1))
        action = np.expand_dims(action, axis=1)
        self.exp['a'] = action
        return action

    def step(self, state):
        self.exp['o'] = 0
        self.exp['s1'] = state
        self.exp['r1'] = self.env.get_r(state, self.env.g, self.env.vs).squeeze()
        self.trajectory.append(self.exp.copy())
        self.train()
        self.exp['s0'] = self.exp['s1']
        self.exp['r0'] = self.exp['r1']
        return self.exp['r1'][self.env.idx] == self.env.R

    def train(self):

        samples = self.env.sample(self.batch_size)
        if samples is not None:
            u0, u1 = np.where(samples['u'])
            s1 = samples['s1'][u0]
            r1 = samples['r1'][u0, u1]
            g = samples['g'][u0]
            v = self.env.vs[u1]
            targets = self.critic.get_targets_dqn(s1, r1, g, v)
            s0 = samples['s0'][u0]
            a = samples['a'][u0]
            inputs = [s0, a, g, v, targets]
            metrics = self.critic.train(inputs)
            # for i, name in enumerate(self.critic.metrics_dqn_names):
            #     self.env.train_metrics[name] += np.mean(np.squeeze(metrics[i]))
            self.critic.target_train()

    def end_episode(self):
        self.env.end_episode(self.trajectory)
        self.trajectory.clear()

    def imit(self):

        idx, samples = self.env.sampleT(self.batch_size)
        v = np.repeat(np.expand_dims(self.env.vs[idx], 0), self.batch_size, axis=0)
        if samples is not None:
            targets = self.critic.get_targets_dqn(samples['r1'][:, idx], samples['t'], samples['s1'], v, v)
            inputs = [samples['s0'], samples['a'], v, v, targets, samples['mcr'][:, [idx]]]
            metrics = self.critic.imit(inputs)
            metrics[2] = 1/(np.where(np.argmax(metrics[2], axis=1) == samples['a'][:, 0],
                                     0.99, 0.01 / self.env.action_dim))
            for i, name in enumerate(self.critic.metrics_imit_names):
                self.env.train_metrics[name][idx] += np.mean(np.squeeze(metrics[i]))
            self.critic.target_train()

    def log(self, step):

        for key, val in self.env.get_stats().items():
            self.stats[key] = val

        for f in self.env.feat:
            self.stats['qval'+str(f)] /= (self.stats['envstep'+str(f)] + 0.01)

        self.stats['step'] = step

        for key in sorted(self.stats.keys()):
            self.logger.logkv(key, self.stats[key])
        self.logger.dumpkvs()

        for f in self.env.feat:
            self.stats['qval'+str(f)] = 0






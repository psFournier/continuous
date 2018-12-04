import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from .critic import Critic2
from utils.util import softmax

class Dqn2():
    def __init__(self, args, env, logger, short_logger):
        self.env = env
        self.logger = logger
        self.short_logger = short_logger
        self.log_dir = args['--log_dir']
        self.batch_size = int(args['--batchsize'])
        self.stats = {}
        self.short_stats = {}
        self.exp = {}
        self.trajectory = []
        self.critic = Critic2(args, env)
        self.env.train_metrics = {name: [0] * self.env.N for name in
                                  self.critic.metrics_dqn_names + self.critic.metrics_imit_names}

    def reset(self, state):
        self.exp = {}
        self.exp['s0'] = state
        self.exp = self.env.reset(self.exp)
        self.exp['r0'] = self.env.get_r(self.exp['s0'], self.env.g, self.env.vs).squeeze()

    def process_trajectory(self, t):
        self.env.process_trajectory(t)

    def act(self):
        input = [np.expand_dims(self.exp['s0'], axis=0)]
        qvals = self.critic.qvals(input)[0].squeeze()[self.env.idx]
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
            targets = self.critic.get_targets_dqn(s1, r1, np.expand_dims(u1, axis=1))
            s0 = samples['s0'][u0]
            a = samples['a'][u0]
            inputs = [s0, np.expand_dims(u1, axis=1), a, targets]
            metrics = self.critic.train(inputs)
            # for i, name in enumerate(self.critic.metrics_dqn_names):
            #     self.env.train_metrics[name][idx] += np.mean(np.squeeze(metrics[i]))
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
        self.stats, self.short_stats = self.env.get_stats()
        self.stats['step'] = step
        self.short_stats['step'] = step
        # for i, f in enumerate(self.env.feat):
        #     for name, metric in self.env.train_metrics.items():
        #         self.stats[name + str(f)] = float("{0:.3f}".format(metric[i]))
        #         metric[i] = 0
        #     self.env.queues[i].init_stat()
        for key in sorted(self.stats.keys()):
            self.logger.logkv(key, self.stats[key])
        for key in sorted(self.short_stats.keys()):
            self.short_logger.logkv(key, self.short_stats[key])
        self.logger.dumpkvs()
        self.short_logger.dumpkvs()





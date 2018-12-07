import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from .critic import Critic2
from utils.util import softmax

class Dqn2():
    def __init__(self, args, env, logger):
        self.env = env
        self.logger = logger
        self.log_dir = args['--log_dir']
        self.batch_size = int(args['--batchsize'])
        self.stats = {}
        self.initstats()
        self.stats['step'] = 0
        self.exp = {}
        self.trajectory = []
        self.critic = Critic2(args, env)

    def initstats(self):
        for f in self.env.feat:
            self.stats['qval'+str(f)] = 0
            self.stats['imitstep'+str(f)] = 0
        self.stats['qvalAll'] = 0
        self.stats['loss'] = 0

    def reset(self, state):
        self.exp = {}
        self.exp['s0'] = state
        self.exp = self.env.reset(self.exp)
        self.exp['r0'] = self.env.get_r(self.exp['s0'], self.env.g, self.env.vs).squeeze()

    def process_trajectory(self, t):
        self.env.process_trajectory(t)

    def act(self):
        input = [np.expand_dims(self.exp['s0'], axis=0)]
        qvals = self.critic.qvals(input)[0].squeeze()
        for i, f in enumerate(self.env.feat):
            self.stats['qval'+str(f)] += np.mean(np.squeeze(qvals[i]))
        action = np.random.choice(range(self.env.action_dim), p=softmax(qvals[self.env.idx], theta=1))
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
        samples = self.env.buffer.sample(self.batch_size)
        if samples is not None:
            u0, u1 = np.where(samples['u'])
            s1 = samples['s1'][u0]
            r1 = samples['r1'][u0, u1]
            targets = self.critic.get_targets_dqn(s1, np.expand_dims(u1, axis=1), r1)
            s0 = samples['s0'][u0]
            a = samples['a'][u0]
            inputs = [s0, np.expand_dims(u1, axis=1), a, targets]
            loss, qval = self.critic.train(inputs)
            self.stats['qvalAll'] += np.mean(np.squeeze(qval))
            self.stats['loss'] += np.mean(loss)
            self.critic.target_train()

    def end_episode(self):
        self.env.end_episode(self.trajectory)
        self.trajectory.clear()

    def imit(self):
        samples, t = self.env.sampleT(self.batch_size)
        if samples is not None:
            # u0, u1 = np.where(samples['u'])
            u0, u1 = np.array(range(self.batch_size)), np.array([t]*self.batch_size)
            # idx = []
            # cps = self.env.get_cps()
            # m = max(cps)
            # for i, t in enumerate(u1):
            #     if cps[t] != 0 or m == 0 or np.random.rand() < self.eps3:
            #         idx.append(i)
            # u0 = u0[np.array(idx)]
            # u1 = u1[np.array(idx)]
            s1 = samples['s1'][u0]
            r1 = samples['r1'][u0, u1]
            targets = self.critic.get_targets_dqn(s1, np.expand_dims(u1, axis=1), r1)
            s0 = samples['s0'][u0]
            a = samples['a'][u0]
            inputs = [s0, np.expand_dims(u1, axis=1), a, targets]
            _ = self.critic.imit(inputs)
            for i, f in enumerate(self.env.feat):
                self.stats['imitstep' + str(f)] += 1
            self.critic.target_train()

    def log(self, step):

        for key, val in self.env.get_stats().items():
            self.stats[key] = val

        for f in self.env.feat:
            self.stats['qval'+str(f)] /= (step - self.stats['step'])

        self.stats['qvalAll'] /= (step - self.stats['step'])
        self.stats['loss'] /= (step - self.stats['step'])
        self.stats['step'] = step

        for key in sorted(self.stats.keys()):
            self.logger.logkv(key, self.stats[key])
        self.logger.dumpkvs()

        self.initstats()






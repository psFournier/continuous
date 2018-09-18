import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents import DQNG
from buffers import ReplayBuffer, PrioritizedReplayBuffer

class DQNGM(DQNG):
    def __init__(self, args, env, env_test, logger):
        super(DQNGM, self).__init__(args, env, env_test, logger)

    def init(self, args ,env):
        names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal', 'mask']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy())
        if args['--imit'] != '0':
            names.append('expVal')
            self.bufferImit = ReplayBuffer(limit=int(1e6), names=names.copy())
        self.critic = CriticDQNGM(args, env)

    def train(self):
        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, g, m = [exp[name] for name in self.buffer.names]
            targets_dqn = self.critic.get_targets_dqn(r, t, s1, g, m)
            inputs = [s0, a0, g, m]
            loss = self.critic.qvalModel.train_on_batch(inputs, targets_dqn)
            for i, metric in enumerate(self.critic.qvalModel.metrics_names):
                self.metrics[metric] += loss[i]

            if self.args['--imit'] == '1':
                self.trainImit()

            self.critic.target_train()

    def trainImit(self):
        if self.bufferImit.nb_entries > self.batch_size:
            exp = self.bufferImit.sample(self.batch_size)
            s0, a0, s1, r, t, g, m, e = [exp[name] for name in self.bufferImit.names]
            targets_dqn = self.critic.get_targets_dqn(r, t, s1, g, m)
            targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
            inputs = [s0, a0, g, m, np.repeat(0.5, self.batch_size, axis=0), e]
            loss = self.critic.imitModel.train_on_batch(inputs, targets)
            for i, metric in enumerate(self.critic.imitModel.metrics_names):
                self.imitMetrics[metric] += loss[i]

    def make_input(self, state, t):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        input.append(np.expand_dims([0.5], axis=0))
        return input

    def reset(self):

        if self.trajectory:
            T = int(self.trajectory[-1]['terminal'])
            R = np.sum([self.env.unshape(exp['reward'], exp['terminal']) for exp in self.trajectory])
            S = len(self.trajectory)
            self.env.processEp(R, S, T)
            for expe in reversed(self.trajectory):
                self.buffer.append(expe.copy())

            if self.args['--imit'] == '1':
                Es = [0]
                goals = []
                masks = []
                for i, expe in enumerate(reversed(self.trajectory)):

                    if self.trajectory[-1]['terminal']:
                        Es[0] = Es[0] * self.critic.gamma + expe['reward']
                        expe['expVal'] = Es[0]
                    else:
                        expe['expVal'] = -self.ep_steps
                    self.bufferImit.append(expe.copy())

                    if np.random.rand() < 0.1:
                        goals.append(expe['state1'])
                        object = np.random.choice(len(self.env.goals))
                        masks.append(self.env.obj2mask(object))
                        Es.append(0)

                    for j, (g, m) in enumerate(zip(goals, masks)):
                        expe['goal'] = g
                        expe['mask'] = m
                        expe = self.env.eval_exp(expe)
                        Es[1 + j] = Es[1 + j] * self.critic.gamma + expe['reward']
                        expe['expVal'] = Es[1 + j]
                        self.bufferImit.append(expe.copy())

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state
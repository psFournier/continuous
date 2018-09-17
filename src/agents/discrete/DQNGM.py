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
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal', 'mask']
        if args['--imit'] != '0':
            self.names.append('expVal')
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.critic = CriticDQNGM(args, env)

    def train(self):
        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            if self.args['--imit'] == '0':
                s0, a0, s1, r, t, g, m = [exp[name] for name in self.names]
            else:
                s0, a0, s1, r, t, g, m, e = [exp[name] for name in self.names]
            temp = np.expand_dims([1], axis=0)
            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1, g, m, temp])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1, g, m])
            targets_dqn = self.compute_targets(r, t, q)

            if self.args['--imit'] == '0':
                targets = targets_dqn
                inputs = [s0, a0, g, m]
            elif self.args['--imit'] == '2':
                e = exp['expVal']
                targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
                inputs = [s0, a0, g, m, e]
            else:
                e = exp['expVal']
                targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
                inputs = [s0, a0, g, m, np.repeat(0.5, self.batch_size, axis=0), e]
            loss = self.critic.qvalModel.train_on_batch(inputs, targets)

            for i, metric in enumerate(self.critic.qvalModel.metrics_names):
                self.metrics[metric] += loss[i]

            self.critic.target_train()

    def make_input(self, state, t):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        # temp = self.env.explor_temp(t)
        input.append(np.expand_dims([0.5], axis=0))
        return input

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
                Es = [0]
                goals = []
                masks = []
                for i, expe in enumerate(reversed(self.trajectory)):

                    if self.trajectory[-1]['terminal']:
                        Es[0] = Es[0] * self.critic.gamma + expe['reward']
                        expe['expVal'] = Es[0]
                    else:
                        expe['expVal'] = -self.ep_steps
                    self.buffer.append(expe.copy())

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
                        self.buffer.append(expe.copy())



            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state
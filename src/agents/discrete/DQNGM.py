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
            s0, a0, s1, r, t, g, m = [exp[name] for name in self.names]
            temp = np.expand_dims([1], axis=0)
            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1, g, m, temp])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1, g, m])
            targets_dqn = self.compute_targets(r, t, q)

            if self.args['--imit'] == '0':
                targets = targets_dqn
                inputs = [s0, a0, g, m]
            else:
                e = exp['expVal']
                targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
                inputs = [s0, a0, g, m, e]
            loss = self.critic.qvalModel.train_on_batch(inputs, targets)

            for i, metric in enumerate(self.critic.qvalModel.metrics_names):
                self.metrics[metric] += loss[i]

            self.critic.target_train()

    def make_input(self, state, t):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        # temp = self.env.explor_temp(t)
        input.append(np.expand_dims([0.5], axis=0))
        return input
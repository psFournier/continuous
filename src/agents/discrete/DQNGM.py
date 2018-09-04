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

    def init(self, env, args):
        self.names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goalVals', 'goal']
        if args['--imit'] != '0':
            self.names.append('expVal')
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.critic = CriticDQNGM(args, env)

    def step(self):

        self.exp['goal'] = self.env.goal
        self.env.steps[self.env.goal] += 1
        self.exp['goalVals'] = self.env.goalVals

        self.exp = self.env.eval_exp(self.exp)
        self.trajectory.append(self.exp.copy())

        if self.buffer.nb_entries > self.batch_size:
            experiences = self.buffer.sample(self.batch_size)
            s1 = experiences['state1']
            s0 = experiences['state0']
            r = experiences['reward']
            t = experiences['terminal']
            a0 = experiences['action']
            g = experiences['goal']
            gv = experiences['goalVals']
            m = np.array([self.env.obj2mask(g[k]) for k in range(self.batch_size)])

            a1Probs = self.critic.actionProbsModel.predict_on_batch([s1, gv, m])
            a1 = np.argmax(a1Probs, axis=1)
            q = self.critic.qvalTModel.predict_on_batch([s1, a1, gv, m])
            targets_dqn = self.compute_targets(r, t, q)

            if self.args['--imit'] == 0:
                targets = targets_dqn
                inputs = [s0, a0, gv, m]
            else:
                e = experiences['expVal']
                targets = [targets_dqn, np.zeros((self.batch_size, 1)), np.zeros((self.batch_size, 1))]
                inputs = [s0, a0, gv, m, e]
            loss = self.critic.qvalModel.train_on_batch(inputs, targets)

            for i, metric in enumerate(self.critic.qvalModel.metrics_names):
                self.metrics[metric] += loss[i]

            self.critic.target_train()

    def make_input(self, state):
        return [np.expand_dims(i, axis=0) for i in [state, self.env.goalVals, self.env.mask]]

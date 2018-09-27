import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents import DQNG
from buffers import ReplayBuffer, PrioritizedReplayBuffer
import time

class DQNGM(DQNG):
    def __init__(self, args, env, env_test, logger):
        super(DQNGM, self).__init__(args, env, env_test, logger)

    def init(self, args ,env):
        names = ['state0', 'action', 'state1', 'reward', 'terminal', 'goal', 'mask']
        self.buffer = ReplayBuffer(limit=int(1e5), names=names.copy(), args=args)
        names.append('expVal')
        # self.bufferOff = PrioritizedReplayBuffer(limit=int(1e4), names=names.copy(), args=args)
        self.bufferOff = ReplayBuffer(limit=int(1e5), names=names.copy(), args=args)
        self.critic = CriticDQNGM(args, env)
        for metric_name in ['loss_dqn', 'qval', 'val', 'loss_dqn_off', 'loss_imit_off','good_exp_off', 'val_off','qval_off']:
            self.metrics[metric_name] = 0

    def train(self):

        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t, g, m = [exp[name] for name in self.buffer.names]
            targets_dqn = self.critic.get_targets_dqn(r, t, s1, g, m)
            inputs = [s0, a0, g, m, targets_dqn]
            loss_dqn, val, qval = self.critic.trainDqn(inputs)
            self.metrics['loss_dqn'] += np.squeeze(loss_dqn)
            self.metrics['val'] += np.mean(val)
            self.metrics['qval'] += np.mean(qval)

        if self.bufferOff.nb_entries > self.batch_size:
            exp = self.bufferOff.sample(self.batch_size)
            s0, a0, s1, r, t, g, m, e = [exp[name] for name in self.bufferOff.names]
            i = exp['indices']
            targets_dqn = self.critic.get_targets_dqn(r, t, s1, g, m)
            inputs = [s0, a0, g, m, 0.5 * np.ones(shape=(self.batch_size, 1)), e, targets_dqn]
            loss_dqn_off, loss_imit_off, good_exp_off, val_off, qval_off, adv = self.critic.trainAll(inputs)
            self.metrics['loss_dqn_off'] += np.squeeze(loss_dqn_off)
            self.metrics['loss_imit_off'] += np.squeeze(loss_imit_off)
            self.metrics['good_exp_off'] += np.squeeze(good_exp_off)
            self.metrics['val_off'] += np.mean(val_off)
            self.metrics['qval_off'] += np.mean(qval_off)
            # self.bufferOff.update_priorities(i, adv)



        if self.buffer.nb_entries > self.batch_size or self.bufferOff.nb_entries > self.batch_size:
            self.critic.target_train()

    def make_input(self, state, mode):
        if mode == 'train':
            input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask, [0.5]]]
        else:
            input = [np.expand_dims(i, axis=0) for i in [state, self.env_test.goal, self.env_test.mask, [0.5]]]
        return input

    def reset(self):

        if self.trajectory:
            R = np.sum([self.env.unshape(exp['reward'], exp['terminal']) for exp in self.trajectory])
            self.env.queues[self.env.object].append(R)

            Es = []
            objs = []
            goals = []
            masks = []
            counts = []
            exp_batch = []

            for i, expe in enumerate(reversed(self.trajectory)):

                self.buffer.append(expe.copy())

                if np.random.rand() < 0.05:
                    for obj, name in enumerate(self.env.goals):
                        m = self.env.obj2mask(obj)
                        s1m = expe['state1'][np.where(m)]
                        sIm = self.env.init_state[np.where(m)]
                        max = np.max(self.env.Rs)
                        if (s1m!=sIm).any() and self.env.Rs[obj] < 0.9 * max:
                            goals.append(expe['state1'].copy())
                            masks.append(m.copy())
                            Es.append(0)

                for j, (g, m) in enumerate(zip(goals, masks)):
                    altExp = expe.copy()
                    altExp['goal'] = g
                    altExp['mask'] = m
                    altExp = self.env.eval_exp(altExp)
                    Es[j] = Es[j] * self.critic.gamma + altExp['reward']
                    altExp['expVal'] = np.expand_dims(Es[j], axis=1)

                    exp_batch.append(altExp)
                    # s_batch.append(s0)
                    # g_batch.append(g)
                    # m_batch.append(m)
                    # e_batch.append(Es[j])

                    # if 0:
                    #     self.bufferOff.append(altExp.copy())
                    # else:
                    #     input = [np.expand_dims(i, axis=0) for i in [s0, g, m]]
                    #     val = self.critic.val(input)
                    #     filter = Es[j] > val[0].squeeze()
                    #     if filter:
                    #         self.bufferOff.append(altExp.copy())
            if exp_batch:
                s0 = np.stack([exp['state0'] for exp in exp_batch])
                g = np.stack([exp['goal'] for exp in exp_batch])
                m = np.stack([exp['mask'] for exp in exp_batch])
                e = np.stack([exp['expVal'] for exp in exp_batch])
                val = self.critic.val([s0, g, m])[0]
                for idx in np.where(e > val)[0]:
                    self.bufferOff.append(exp_batch[idx])

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state
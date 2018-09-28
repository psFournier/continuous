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

            imagined_Es = []
            imagined_goals = []
            imagined_masks = []
            imagined_expe = []
            imagined_obj = []

            filtering = 1
            if filtering:
                s0List = []
                gList = []
                mList = []
                eList = []

            for i, expe in enumerate(reversed(self.trajectory)):

                FLAG = self.buffer.append(expe.copy())
                if FLAG:
                    self.bufferOff._next_idx = 0

                for obj, name in enumerate(self.env.goals):
                    mask = self.env.obj2mask(obj)
                    s1m = expe['state1'][np.where(mask)]
                    s0m = expe['state0'][np.where(mask)]
                    sIm = self.env.init_state[np.where(mask)]
                    # found = (s1m!=sIm).any() and (s0m==sIm).all()
                    found = obj not in imagined_obj and (s1m != sIm).any()
                    mastered = self.env.queues[obj].T[-1]
                    # mastered = 0
                    if found and np.random.rand() < (1.1 - mastered):
                        imagined_goals.append(expe['state1'].copy())
                        imagined_masks.append(mask.copy())
                        imagined_Es.append(0)
                        imagined_obj.append(obj)

                for j, (g, m) in enumerate(zip(imagined_goals, imagined_masks)):
                    expe['goal'] = g
                    expe['mask'] = m
                    expe = self.env.eval_exp(expe)
                    imagined_Es[j] = imagined_Es[j] * self.critic.gamma + expe['reward']
                    expe['expVal'] = np.expand_dims(imagined_Es[j], axis=1)
                    imagined_expe.append(expe.copy())

                    if filtering:
                        s0List.append(expe['state0'])
                        gList.append(g)
                        mList.append(m)
                        eList.append(expe['expVal'])

                    # if 0:
                    #     self.bufferOff.append(altExp.copy())
                    # else:
                    #     input = [np.expand_dims(i, axis=0) for i in [s0, g, m]]
                    #     val = self.critic.val(input)
                    #     filter = Es[j] > val[0].squeeze()
                    #     if filter:
                    #         self.bufferOff.append(altExp.copy())
            if imagined_expe:
                if filtering:
                    e = np.array(eList)
                    val = self.critic.val([np.array(l) for l in [s0List, gList, mList]])[0]
                    for idx in np.where(e > val)[0]:
                        self.bufferOff.append(imagined_expe[idx])
                else:
                    for exp in imagined_expe:
                        self.bufferOff.append(exp)

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state
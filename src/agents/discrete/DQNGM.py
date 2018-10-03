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
        names = ['s0', 'a', 's1', 'r', 't', 'g', 'm']
        metrics = ['loss_dqn', 'qval', 'val']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
        if self.args['--wimit'] != '0':
            names.append('mcr')
            metrics += ['loss_dqn2', 'val2', 'qval2', 'loss_imit', 'good_exp']
            self.bufferImit = PrioritizedReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
        self.critic = CriticDQNGM(args, env)
        for metric in metrics:
            self.metrics[metric] = 0

    def train(self):

        if self.buffer.nb_entries > 100 * self.batch_size:

            exp = self.buffer.sample(self.batch_size)
            targets = self.critic.get_targets_dqn(exp['r'], exp['t'], exp['s1'], exp['g'], exp['m'])
            inputs = [exp['s0'], exp['a'], exp['g'], exp['m'], targets]
            metrics = self.critic.train_dqn(inputs)
            self.metrics['loss_dqn'] += np.squeeze(metrics[0])
            self.metrics['val'] += np.mean(metrics[1])
            self.metrics['qval'] += np.mean(metrics[2])

            if self.args['--wimit'] != '0':
                exp = self.bufferImit.sample(self.batch_size)
                targets = self.critic.get_targets_dqn(exp['r'], exp['t'], exp['s1'], exp['g'], exp['m'])
                inputs = [exp['s0'], exp['a'], exp['g'], exp['m'], targets, exp['mcr']]
                metrics = self.critic.train_imit(inputs)
                self.metrics['loss_dqn2'] += np.squeeze(metrics[0])
                self.metrics['val2'] += np.mean(metrics[1])
                self.metrics['qval2'] += np.mean(metrics[2])
                self.metrics['loss_imit'] += np.squeeze(metrics[3])
                self.metrics['good_exp'] += np.squeeze(metrics[4])
                self.bufferImit.update_priorities(exp['indices'], metrics[-1])

            self.critic.target_train()

            # self.bufferOff.update_priorities(i, adv)

    def make_input(self, state, mode):
        if mode == 'train':
            input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        else:
            input = [np.expand_dims(i, axis=0) for i in [state, self.env_test.goal, self.env_test.mask]]
        return input

    def reset(self):

        if self.trajectory:
            R = np.sum([self.env.unshape(exp['r'], exp['t']) for exp in self.trajectory])
            self.env.queues[self.env.object].append(R)

            goals = []
            masks = []
            objs = []
            if self.args['--wimit'] != '0':
                mcrs = []

            for i, expe in enumerate(reversed(self.trajectory)):

                expe['mcr'] = np.expand_dims(-100, axis=1)
                self.buffer.append(expe.copy())
                if self.args['--wimit'] != '0':
                    self.bufferImit.append(expe.copy())

                if self.args['--her'] != '0':
                    for obj, name in enumerate(self.env.goals):
                        mask = self.env.obj2mask(obj)
                        s1m = expe['s1'][np.where(mask)]
                        s0m = expe['s0'][np.where(mask)]
                        found = obj not in objs and (s1m != s0m).any()
                        if found:
                            goals.append(expe['s1'].copy())
                            masks.append(mask.copy())
                            objs.append(obj)
                            if self.args['--wimit'] != '0':
                                mcr = (1 - self.critic.gamma ** (i+1)) / (1 - self.critic.gamma)
                                mcrs.append(mcr)

                for j, (g, m) in enumerate(zip(goals, masks)):
                    expe['g'] = g
                    expe['m'] = m
                    expe = self.env.eval_exp(expe)
                    if self.args['--wimit'] != '0':
                        mcrs[j] = mcrs[j] * self.critic.gamma + expe['r']
                        expe['mcr'] = np.expand_dims(mcrs[j], axis=1)
                        self.bufferImit.append(expe.copy())
                    self.buffer.append(expe.copy())

            # imagined_MCR = []
            # imagined_goals = []
            # imagined_masks = []
            # imagined_expe = []
            # imagined_obj = []
            #
            # filtering = 1
            # if filtering:
            #     s0List = []
            #     gList = []
            #     mList = []
            #     eList = []
            #
            # for i, expe in enumerate(reversed(self.trajectory)):
            #
            #     FLAG = self.buffer.append(expe.copy())
            #     if FLAG:
            #         self.bufferOff._next_idx = 0
            #
            #     for obj, name in enumerate(self.env.goals):
            #         mask = self.env.obj2mask(obj)
            #         s1m = expe['state1'][np.where(mask)]
            #         s0m = expe['state0'][np.where(mask)]
            #         sIm = self.env.init_state[np.where(mask)]
            #         # found = (s1m!=sIm).any() and (s0m==sIm).all()
            #         found = obj not in imagined_obj and (s1m != sIm).any()
            #         mastered = self.env.queues[obj].T[-1]
            #         # mastered = 0
            #         if found and np.random.rand() < (1.1 - mastered):
            #             imagined_goals.append(expe['state1'].copy())
            #             imagined_masks.append(mask.copy())
            #             imagined_Es.append(0)
            #             imagined_obj.append(obj)
            #
            #     for j, (g, m) in enumerate(zip(imagined_goals, imagined_masks)):
            #         expe['goal'] = g
            #         expe['mask'] = m
            #         expe = self.env.eval_exp(expe)
            #         imagined_Es[j] = imagined_Es[j] * self.critic.gamma + expe['reward']
            #         expe['expVal'] = np.expand_dims(imagined_Es[j], axis=1)
            #         imagined_expe.append(expe.copy())
            #
            #         if filtering:
            #             s0List.append(expe['state0'])
            #             gList.append(g)
            #             mList.append(m)
            #             eList.append(expe['expVal'])

                    # if 0:
                    #     self.bufferOff.append(altExp.copy())
                    # else:
                    #     input = [np.expand_dims(i, axis=0) for i in [s0, g, m]]
                    #     val = self.critic.val(input)
                    #     filter = Es[j] > val[0].squeeze()
                    #     if filter:
                    #         self.bufferOff.append(altExp.copy())
            # if imagined_expe:
            #     if filtering:
            #         e = np.array(eList)
            #         val = self.critic.val([np.array(l) for l in [s0List, gList, mList]])[0]
            #         for idx in np.where(e > val)[0]:
            #             self.bufferOff.append(imagined_expe[idx])
            #     else:
            #         for exp in imagined_expe:
            #             self.bufferOff.append(exp)

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state
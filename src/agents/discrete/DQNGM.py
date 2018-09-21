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
            inputs = [s0, a0, g, m, targets_dqn]
            loss_dqn, val, qval = self.critic.trainCritic(inputs)
            self.metrics['loss_dqn'] += np.squeeze(loss_dqn)
            self.metrics['val'] += np.mean(val)
            self.metrics['qval'] += np.mean(qval)

            if self.args['--imit'] != '0':
                self.trainImit()

            self.critic.target_train()

    def trainImit(self):
        if self.bufferImit.nb_entries > self.batch_size:
            exp = self.bufferImit.sample(self.batch_size)
            s0, a0, s1, r, t, g, m, e = [exp[name] for name in self.bufferImit.names]
            targets_dqn = self.critic.get_targets_dqn(r, t, s1, g, m)
            inputs = [s0, a0, g, m, 0.5 * np.ones(shape=(self.batch_size,1)), e, targets_dqn]
            loss_dqn2, loss_imit, good_exp, val = self.critic.trainImit(inputs)
            self.metrics['loss_dqn2'] += np.squeeze(loss_dqn2)
            self.metrics['loss_imit'] += np.squeeze(loss_imit)
            self.metrics['good_exp'] += np.squeeze(good_exp)

    def make_input(self, state, t):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask, [0.5]]]
        return input

    def reset(self):

        if self.trajectory:
            T = int(self.trajectory[-1]['terminal'])
            R = np.sum([self.env.unshape(exp['reward'], exp['terminal']) for exp in self.trajectory])
            S = len(self.trajectory)
            self.env.processEp(R, S, T)
            for expe in reversed(self.trajectory):
                self.buffer.append(expe.copy())

            if self.args['--imit'] != '0':
                Es = []
                objs = []
                goals = []
                masks = []
                counts = []
                for i, expe in enumerate(reversed(self.trajectory)):

                    # for obj, name in enumerate(self.env.goals):
                    #     m = self.env.obj2mask(obj)
                    #     if (expe['state1'][np.where(m)] != expe['state0'][np.where(m)]).any():
                    #         altExp = expe.copy()
                    #         altExp['goal'] = expe['state1'].copy()
                    #         altExp['mask'] = m
                    #         altExp = self.env.eval_exp(altExp)
                    #         altExp['expVal'] = np.expand_dims(altExp['reward'], axis=1)
                    #         self.bufferImit.append(altExp.copy())

                    if expe['terminal'] or i==0 or np.random.rand() < 0.1:
                        for obj, name in enumerate(self.env.goals):
                            m = self.env.obj2mask(obj)
                            diff = (expe['state1'][np.where(m)] != self.env.init_state[np.where(m)]).any()
                            T = self.env.queues[obj].T
                            mastered = T and T[-1] > 0.8
                            if diff and not mastered:
                                objs.append(obj)
                                goals.append(expe['state1'].copy())
                                masks.append(self.env.obj2mask(obj).copy())
                                Es.append(0)
                                counts.append(0)

                    for j, (g, m) in enumerate(zip(goals, masks)):
                        if counts[j] <= 20:
                            altExp = expe.copy()
                            altExp['goal'] = g
                            altExp['mask'] = m
                            altExp = self.env.eval_exp(altExp)
                            Es[j] = Es[j] * self.critic.gamma + altExp['reward']
                            counts[j] += 1
                            altExp['expVal'] = np.expand_dims(Es[j], axis=1)
                            self.bufferImit.append(altExp.copy())

                for obj, count in zip(objs, counts):
                    self.env.replays[obj] += count

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state
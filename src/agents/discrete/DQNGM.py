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
            self.env.end_episode(self.trajectory)
            for expe in reversed(self.trajectory):
                self.buffer.append(expe.copy())
            if self.args['--her'] != '0':
                augmented_ep = self.env.augment_episode(self.trajectory)
                for e in augmented_ep:
                    self.buffer.append(e)
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def demo(self):
        if self.env_step % self.demo_freq == 0:
            demo = self.get_demo(rndprop=0.1)
            augmented_demo = self.env.augment_episode(demo)
            for exp in augmented_demo:
                self.buffer.append(exp)
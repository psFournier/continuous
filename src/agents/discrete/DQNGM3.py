import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents import DQNG
import time

class DQNGM3(DQNG):
    def __init__(self, args, env, env_test, logger):
        super(DQNGM3, self).__init__(args, env, env_test, logger)

    def init(self, args ,env):
        metrics = ['loss_dqn', 'qval', 'val']
        self.critic = CriticDQNGM(args, env)
        for metric in metrics:
            self.metrics[metric] = 0
        self.goalcounts = np.zeros((len(self.env.goals),))

    def train(self):

        if self.env.buffer.nb_entries > 100 * self.batch_size:
            samples = self.env.buffer.sample(self.batch_size)
            samples = self.env.augment_samples(samples)
            targets = self.critic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
            inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets]
            metrics = self.critic.train_dqn(inputs)
            self.metrics['loss_dqn'] += np.squeeze(metrics[0])
            self.metrics['val'] += np.mean(metrics[1])
            self.metrics['qval'] += np.mean(metrics[2])
            self.goalcounts += np.bincount(samples['task'], minlength=len(self.env.goals))
            self.critic.target_train()

    def get_stats(self):
        sumsamples = np.sum(self.goalcounts)
        if sumsamples != 0:
            for i, goal in enumerate(self.env.goals):
                self.stats['samplecount_{}'.format(goal)] = float("{0:.3f}".format(self.goalcounts[i] / sumsamples))

    def make_input(self, state, mode):
        if mode == 'train':
            input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        else:
            input = [np.expand_dims(i, axis=0) for i in [state, self.env_test.goal, self.env_test.mask]]
        return input

    def reset(self):

        if self.trajectory:
            self.env.end_episode(self.trajectory)
            for expe in self.trajectory:
                self.env.buffer.append(expe.copy())
            if self.args['--her'] != '0':
                augmented_ep = self.env.augment_episode(self.trajectory)
                for e in augmented_ep:
                    self.env.buffer.append(e)
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state



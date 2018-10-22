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
        names = ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'task', 'mcr']
        metrics = ['loss_dqn', 'qval', 'val']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
        if self.args['--wimit'] != '0':
            metrics += ['loss_dqn2', 'val2', 'qval2', 'loss_imit', 'good_exp']
            self.bufferImit = ReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
            self.imitBatchsize = 32
        self.critic = CriticDQNGM(args, env)
        for metric in metrics:
            self.metrics[metric] = 0
        self.goalcounts = np.zeros((len(self.env.goals),))

    def train(self):

        if self.buffer.nb_entries > 100 * self.batch_size:

            samples = self.buffer.sample(self.batch_size)
            samples = self.env.augment_samples(samples)
            targets = self.critic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
            inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets]
            metrics = self.critic.train_dqn(inputs)
            self.metrics['loss_dqn'] += np.squeeze(metrics[0])
            self.metrics['val'] += np.mean(metrics[1])
            self.metrics['qval'] += np.mean(metrics[2])
            self.goalcounts += np.bincount(samples['task'], minlength=len(self.env.goals))

            if self.args['--wimit'] != '0':
                samples = self.bufferImit.sample(self.imitBatchsize)
                samples = self.env.augment_tutor_samples(samples)
                targets = self.critic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
                inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets, samples['mcr']]
                metrics = self.critic.train_imit(inputs)
                self.metrics['loss_dqn2'] += np.squeeze(metrics[0])
                self.metrics['val2'] += np.mean(metrics[1])
                self.metrics['qval2'] += np.mean(metrics[2])
                self.metrics['loss_imit'] += np.squeeze(metrics[3])
                self.metrics['good_exp'] += np.squeeze(metrics[5])

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

    def act(self, exp, mode='train'):
        input = self.make_input(exp['s0'], mode)
        actionProbs = self.critic.actionProbs(input)[0].squeeze()
        if mode=='train':
            action = np.random.choice(range(self.env.action_dim), p=actionProbs)
        else:
            action = np.argmax(actionProbs[0])
        prob = actionProbs[action]
        action = np.expand_dims(action, axis=1)
        exp['a'] = action
        # exp['p_a'] = prob
        return exp

    def reset(self):

        if self.trajectory:
            augmented_episode = self.env.end_episode(self.trajectory)
            for expe in augmented_episode:
                self.buffer.append(expe)
            # for expe in self.trajectory:
            #     self.buffer.append(expe.copy())
            # augmented_ep = self.env.augment_episode(self.trajectory)
            # for e in augmented_ep:
            #     self.buffer.append(e)
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def get_demo(self, rndprop):
        demo = []
        exp = {}
        exp['s0'] = self.env_test.env.reset()
        # obj = None
        # goal = np.random.randint(8, size=2)
        obj = self.env_test.env.chest1
        goal = 2
        while True:
            if np.random.rand() < rndprop:
                a = np.random.randint(self.env_test.action_dim)
                done = False
            else:
                a, done = self.env_test.env.opt_action(obj, goal)
            if not done:
                exp['a'] = np.expand_dims(a, axis=1)
                exp['s1'] = self.env_test.env.step(exp['a'])[0]
                demo.append(exp.copy())
                exp['s0'] = exp['s1']
            else:
                break
        return demo

    def demo(self):
        if self.env_step % self.demo_freq == 0 and self.args['--wimit'] != '0':
            for i in range(5):
                demo = self.get_demo(rndprop=0.)
                augmented_demo = self.env.augment_demo(demo)
                for exp in augmented_demo:
                    self.bufferImit.append(exp)
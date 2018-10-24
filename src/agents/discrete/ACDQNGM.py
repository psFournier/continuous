import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import ActorCriticDQNGM
from agents import DQNG
from buffers import ReplayBuffer, PrioritizedReplayBuffer
import time

class ACDQNGM(DQNG):
    def __init__(self, args, env, env_test, logger):
        super(ACDQNGM, self).__init__(args, env, env_test, logger)

    def init(self, args ,env):
        names = ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'task', 'mcr']
        metrics = ['loss_dqn', 'qval', 'val']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
        self.actorCritic = ActorCriticDQNGM(args, env)
        for metric in metrics:
            self.metrics[metric] = 0
        self.goalcounts = np.zeros((len(self.env.goals),))

    def train(self):

        if self.buffer.nb_entries > 100 * self.batch_size:

            samples = self.buffer.sample(self.batch_size)
            samples = self.env.augment_samples(samples)
            targets = self.actorCritic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
            inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets]
            metricsCritic = self.actorCritic.trainCritic(inputs)
            self.metrics['loss_dqn'] += np.squeeze(metricsCritic[0])
            self.metrics['qval'] += np.mean(metricsCritic[1])
            self.goalcounts += np.bincount(samples['task'], minlength=len(self.env.goals))
            metricsActor = self.actorCritic.trainActor([samples['s0'], samples['g'], samples['m']])
            if self.env_step % 1000 ==0: print(metricsActor[0], metricsActor[1])
            self.metrics['val'] += np.mean(metricsActor[2])
            self.actorCritic.target_train()

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
        actionProbs = self.actorCritic.probs(input)[0].squeeze()
        # if self.env_step % 1000 == 0: print(actionProbs)
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
        # obj = np.random.choice(self.env_test.env.objects)
        # goal = np.random.randint(obj.high[2]+1)
        obj = self.env_test.env.light
        goal = 1
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
        if self.env_step % self.demo_freq == 0:
            for i in range(5):
                demo = self.get_demo(rndprop=0.)
                augmented_demo = self.env.augment_demo(demo)
                for exp in augmented_demo:
                    self.buffer.append(exp)
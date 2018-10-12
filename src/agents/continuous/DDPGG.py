import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import ActorCriticDDPGG
from buffers import ReplayBuffer
from agents import DDPG
import os

class DDPGG(DDPG):

    def __init__(self, args, env, env_test, logger):
        super(DDPGG, self).__init__(args, env, env_test, logger)

    def init(self, args ,env):
        names = ['s0', 'a', 's1', 'r', 't', 'g']
        metrics = ['loss_dqn', 'loss_actor']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
        self.actorCritic = ActorCriticDDPGG(args, env)
        for metric in metrics:
            self.metrics[metric] = 0

    def train(self):

        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            targets_dqn = self.actorCritic.get_targets_dqn(exp['r'], exp['t'], exp['s1'], exp['g'])
            inputs = [exp['s0'], exp['a'], exp['g'], targets_dqn]
            loss_dqn = self.actorCritic.trainQval(inputs)
            action, criticActionGrads, invertedCriticActionGrads = self.actorCritic.trainActor([exp['s0'], exp['g']])
            self.metrics['loss_dqn'] += np.squeeze(loss_dqn)
            self.actorCritic.target_train()

    def make_input(self, state, mode):
        if mode == 'train':
            input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal]]
        else:
            input = [np.expand_dims(i, axis=0) for i in [state, self.env_test.goal]]
        return input

    def reset(self):

        if self.trajectory:
            self.env.end_episode(self.trajectory)
            for expe in self.trajectory:
                self.buffer.append(expe.copy())
            if self.args['--her'] != '0':
                augmented_ep = self.env.augment_episode(self.trajectory)
                for e in augmented_ep:
                    self.buffer.append(e)
            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state




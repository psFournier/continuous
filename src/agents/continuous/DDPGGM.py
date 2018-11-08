import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import ActorCriticDDPGGM
from buffers import ReplayBuffer
from agents.agent import Agent
import os

class DDPGGM(Agent):

    def __init__(self, args, env, env_test, logger):
        super(DDPGGM, self).__init__(args, env, env_test, logger)
        self.init(args, env)

    def init(self, args ,env):
        metrics = ['loss1c', 'qval']
        self.actorCritic = ActorCriticDDPGGM(args, env)
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = 0
        self.rnd_demo = float(args['--rnd_demo'])
        self.demo = int(args['--demo'])

    def train(self):
        task, samples = self.env.sample(self.batch_size)
        if samples is not None:
            targets = self.actorCritic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'],
                                                      samples['m'])
            inputsCritic = [samples['s0'], samples['a'], samples['g'], samples['m'], targets, samples['mcr']]
            inputsActor = [samples['s0'], samples['g'], samples['m']]
            metrics = self.actorCritic.train(inputsCritic, inputsActor)
            self.metrics['loss1'] += np.squeeze(metrics[0])
            self.metrics['qval'] += np.mean(metrics[2])
            self.actorCritic.target_train()

    def make_input(self, state):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        return input

    def act(self, exp):
        input = self.make_input(exp['s0'])
        action = self.actorCritic.action(input)[0].squeeze()
        action = np.expand_dims(action, axis=1)
        exp['a'] = action
        return exp

    def reset(self):

        if self.trajectory:
            self.env.end_episode(self.trajectory)
            self.trajectory.clear()
        state = self.env.reset()
        self.episode_step = 0

        return state

    def log(self):

        if self.env_step % self.eval_freq == 0:

            wrapper_stats = self.env.get_stats()
            for key, val in wrapper_stats.items():
                self.stats[key] = val

            self.stats['step'] = self.env_step
            for metric, val in self.metrics.items():
                self.stats[metric] = val / self.eval_freq
                self.metrics[metric] = 0

            self.get_stats()

            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])

            self.logger.dumpkvs()

            self.save_model()


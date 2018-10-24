import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents.agent import Agent
from buffers import ReplayBuffer, PrioritizedReplayBuffer
import time

class DQNGM(Agent):
    def __init__(self, args, env, env_test, logger):
        super(DQNGM, self).__init__(args, env, env_test, logger)
        self.args = args
        self.init(args, env)

    def init(self, args ,env):

        metrics = ['l_dqn', 'qval', 'val', 'l_dqn_i', 'val_i', 'qval_i', 'l_imit', 'filter']
        self.critic = CriticDQNGM(args, env)
        for metric in metrics:
            self.metrics[metric] = 0
        self.rnd_demo = float(args['--rnd_demo'])

    def train(self):

        if len(self.env.buffer) > 100 * self.batch_size:

            samples = self.env.buffer.sample(self.batch_size)
            targets = self.critic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
            inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets]
            metrics = self.critic.train_dqn(inputs)
            self.metrics['l_dqn'] += np.squeeze(metrics[0])
            self.metrics['val'] += np.mean(metrics[1])
            self.metrics['qval'] += np.mean(metrics[2])
            self.critic.target_train()

    def train_i(self):

        if len(self.env.buffer_i) > 10 * self.batch_size:

            samples = self.env.buffer_i.sample(self.batch_size)
            targets = self.critic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
            inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets, samples['mcr']]
            metrics = self.critic.train_imit(inputs)
            self.metrics['l_dqn_i'] += np.squeeze(metrics[0])
            self.metrics['val_i'] += np.mean(metrics[1])
            self.metrics['qval_i'] += np.mean(metrics[2])
            self.metrics['l_imit'] += np.squeeze(metrics[3])
            self.metrics['filter'] += np.squeeze(metrics[4])
            self.critic.target_train()

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
            self.env.end_episode(self.trajectory)
            self.trajectory.clear()
        state = self.env.reset()
        self.episode_step = 0

        return state

    def tutor_act(self, exp, task, goal):

        if np.random.rand() < self.rnd_demo:
            a = np.random.randint(self.env_test.action_dim)
            done = False
        else:
            a, done = self.env_test.env.opt_action(task, goal)
        exp['a'] = np.expand_dims(a, axis=1)
        exp['t'] = done

        return exp

    def get_demo(self):
        demo = []
        exp = {}
        exp['s0'] = self.env_test.env.reset()
        exp['t'] = False
        task = self.env_test.env.chest1
        goal = 2

        while True:
            exp = self.tutor_act(exp, task, goal)
            exp['s1'] = self.env_test.env.step(exp['a'])[0]
            demo.append(exp.copy())
            self.train_i()

            if exp['t']:
                break
            else:
                exp['s0'] = exp['s1']

        return demo

    def demo(self):
        if self.env_step % self.demo_freq == 0:
            demo = self.get_demo()

            goal = demo[-1]['s1']
            task = 2
            mask = self.env_test.task2mask(task)
            mcr = 0
            for i, exp in enumerate(reversed(demo)):
                exp['g'] = goal
                exp['m'] = mask
                exp['task'] = task
                exp = self.env_test.eval_exp(exp)
                exp['tasks'] = [task]
                mcr = mcr * self.env.gamma + exp['r']
                exp['mcr'] = np.expand_dims(mcr, axis=1)
                self.env.buffer_i.append(exp)


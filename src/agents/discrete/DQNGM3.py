import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM3
from agents.agent import Agent

class DQNGM3(Agent):
    def __init__(self, args, env, env_test, logger):
        super(DQNGM3, self).__init__(args, env, env_test, logger)
        self.init(args, env)

    def init(self, args ,env):

        metrics = ['loss1', 'qval', 'val', 'loss2', 'prop_good']
        self.critic = CriticDQNGM3(args, env)
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = 0
        self.rnd_demo = float(args['--rnd_demo'])
        self.demo = int(args['--demo'])
        self.mode = 'train'

    def train(self):

        task, samples = self.env.sample(self.batch_size)
        if samples is not None:
            targets = self.critic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
            inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets, samples['mcr']]
            metrics = self.critic.train(inputs)
            self.metrics['loss1'] += np.squeeze(metrics[0])
            self.metrics['val'] += np.mean(metrics[1])
            self.metrics['qval'] += np.mean(metrics[2])
            self.metrics['loss2'] += np.squeeze(metrics[3])
            self.metrics['prop_good'] += np.mean(metrics[4])
            self.env.offpolicyness[task] += np.mean(metrics[5] / samples['pa'])
            self.critic.target_train()

    def make_input(self, state):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        return input

    def act(self, exp):
        input = self.make_input(exp['s0'])
        actionProbs = self.critic.actionProbs(input)[0].squeeze()
        if self.env.foreval[self.env.task]:
            action = np.argmax(actionProbs)
        else:
            action = np.random.choice(range(self.env.action_dim), p=actionProbs)
        exp['pa'] = actionProbs[action]
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

    def tutor_act(self, task, goal):

        if np.random.rand() < self.rnd_demo:
            a = np.random.randint(self.env_test.action_dim)
            done = False
        else:
            a, done = self.env_test.env.opt_action(task, goal)

        return a, done

    def get_demo(self):

        demo = []
        exp = {}
        exp['s0'] = self.env_test.env.reset(random=False)
        task = self.env_test.sample_tutor_task()
        goal = self.env_test.sample_tutor_goal(task)
        while True:
            a, done = self.tutor_act(task, goal)
            if done:
                break
            else:
                exp['a'] = np.expand_dims(a, axis=1)
                exp['s1'] = self.env_test.env.step(exp['a'])[0]
                exp['pa'] = 1
                demo.append(exp.copy())
                exp['s0'] = exp['s1']

        return demo, task

    def imitate(self):

        if self.demo != 0 and self.env_step % self.demo_freq == 0:

            demo, true_task = self.get_demo()

            if self.demo == 1:
                tasks = [true_task]
            # elif self.demo == 2:
            #     tasks = [np.random.randint(self.env.Ntasks)]
            # elif self.demo == 3:
            #     tasks = [self.env.sample_task(demo[0]['s0'])]
            else:
                tasks = range(self.env.Ntasks)

            goals = [None for _ in tasks]
            self.env.process_trajectory(demo, tasks, goals, with_mcr=True)

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





import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents.agent import Agent

class DQNGM(Agent):
    def __init__(self, args, env, env_test, logger, short_logger):
        super(DQNGM, self).__init__(args, env, env_test, logger, short_logger)
        self.init(args, env)

    def init(self, args ,env):

        self.critic = CriticDQNGM(args, env)
        self.env.train_metrics = {name: [0]*self.env.Ntasks for name in self.critic.metrics_names}
        self.rnd_demo = float(args['--rnd_demo'])
        self.demo = int(args['--demo'])

    def train(self):

        task, samples = self.env.sample(self.batch_size)
        if samples is not None:
            targets = self.critic.get_targets_dqn(samples['r'], samples['t'], samples['s1'], samples['g'], samples['m'])
            inputs = [samples['s0'], samples['a'], samples['g'], samples['m'], targets, samples['mcr']]
            metrics = self.critic.train(inputs)
            for i, name in enumerate(self.critic.metrics_names):
                self.env.train_metrics[name][task] += np.mean(np.squeeze(metrics[i]))
            # self.env.offpolicyness[task] += np.mean(metrics[5] / samples['pa'])
            self.critic.target_train()

    def make_input(self, state):
        input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal, self.env.mask]]
        return input

    def act(self, exp):
        input = self.make_input(exp['s0'])
        actionProbs = self.critic.actionProbs(input)[0].squeeze()
        # if self.env.foreval[self.env.task]:
        #     action = np.argmax(actionProbs)
        # else:
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
            a, done = self.env_test.opt_action(task, goal)

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
                exp['o'] = 1
                demo.append(exp.copy())
                exp['s0'] = exp['s1']

        return demo, task, goal

    def imitate(self):

        if self.demo != 0 and self.env_step % self.demo_freq == 0:

            demo, true_task, true_goal = self.get_demo()

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

            self.stats['step'] = self.env_step
            self.short_stats['step'] = self.env_step

            for i, task in enumerate(self.env.tasks_feat):
                for name, metric in self.env.train_metrics.items():
                    self.stats[name + '_{}'.format(task)] = float("{0:.3f}".format(
                        metric[i] / (self.env.run_metrics['trainsteps'][i] + 0.00001)
                    ))
                    metric[i] = 0

            wrapper_stats, wrapper_short_stats = self.env.get_stats()
            for key, val in wrapper_stats.items():
                self.stats[key] = val
            for key, val in wrapper_short_stats.items():
                self.stats[key] = val
                self.short_stats[key] = val

            for key in sorted(self.stats.keys()):
                self.logger.logkv(key, self.stats[key])

            for key in sorted(self.short_stats.keys()):
                self.short_logger.logkv(key, self.short_stats[key])

            self.logger.dumpkvs()
            self.short_logger.dumpkvs()

            self.save_model()





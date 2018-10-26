import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM
from agents.agent import Agent

class DQNGM(Agent):
    def __init__(self, args, env, env_test, logger):
        super(DQNGM, self).__init__(args, env, env_test, logger)
        self.init(args, env)

    def init(self, args ,env):

        metrics = ['l_dqn', 'qval', 'val']
        self.critic = CriticDQNGM(args, env)
        self.metrics = {}
        for metric in metrics:
            self.metrics[metric] = 0
        self.rnd_demo = float(args['--rnd_demo'])
        self.demo = int(args['--demo'])
        self.mode = 'train'

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
        task = self.env_test.env.chest1
        goal = 2
        # task = np.random.choice(self.env_test.env.objects)
        # goal = np.random.randint(task.high[2] + 1)
        while True:
            a, done = self.tutor_act(task, goal)
            if done:
                break
            else:
                exp['a'] = np.expand_dims(a, axis=1)
                exp['s1'] = self.env_test.env.step(exp['a'])[0]
                demo.append(exp.copy())
                exp['s0'] = exp['s1']

        return demo, task

    def imitate(self):

        if self.demo != 0 and self.env_step % self.demo_freq == 0:

            trajectory, true_task = self.get_demo()

            if self.demo == 1:
                tasks = [true_task]
            elif self.demo == 2:
                tasks = [np.random.randint(self.env.Ntasks)]
            elif self.demo == 3:
                tasks = [self.env.sample_task_goal(trajectory[0]['s0'])[0]]
            else:
                tasks = range(self.env.Ntasks)

            masks = [self.env_test.task2mask(task) for task in tasks]
            goals = [None for _ in tasks]

            for exp in reversed(trajectory):

                for i, task in enumerate(tasks):

                    if goals[i] is None and (exp['s1'][np.where(masks[i])] != exp['s0'][np.where(masks[i])]).any():
                        goals[i] = exp['s1']
                    if goals[i] is not None:
                        exp['g'] = goals[i]
                        exp['m'] = masks[i]
                        exp['task'] = task
                        exp = self.env_test.eval_exp(exp)
                        self.env.buffer.append(exp)

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





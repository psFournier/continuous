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
        self.env.train_metrics = {name: [0]*self.env.N for name in self.critic.metrics_names}
        self.rnd_demo = float(args['--rnd_demo'])
        self.demo = int(args['--demo'])

    def run(self):

        self.exp = self.env.reset()
        self.episode_step = 0

        self.env.idx = self.env.sample_v(self.exp['s0'])
        self.episode_task = 1

        try:
            while self.env_step < self.max_steps:
                # self.env.render(mode='human')
                # self.env.unwrapped.viewer._record_video = True
                # self.env.unwrapped.viewer._video_path = os.path.join(self.logger.get_dir(), "video_%07d.mp4")
                # self.env.unwrapped.viewer._run_speed = 0.125
                self.exp = self.act(self.exp)
                self.exp = self.env.step(self.exp)
                self.trajectory.append(self.exp.copy())
                self.train()
                self.env_step += 1
                self.episode_step += 1

                if self.exp['t'] or self.episode_step >= self.ep_steps:
                    self.env.end_episode(self.trajectory)
                    self.trajectory.clear()
                    if self.env.done(self.exp) or self.episode_task >= 10:
                        self.exp = self.env.reset()
                        self.episode_task = 0
                    else:
                        self.exp['s0'] = self.exp['s1']
                        self.exp['r0'] = self.exp['r1']
                        self.exp['t'] = 0
                        self.env.idx = self.env.sample_v(self.exp['s0'])
                    self.episode_step = 0
                    self.episode_task += 1
                else:
                    self.exp['s0'] = self.exp['s1']
                    self.exp['r0'] = self.exp['r1']

                if self.env_step % self.eval_freq == 0:
                    self.log()
                self.imitate()

        except KeyboardInterrupt:
            print("Keybord interruption")

    def train(self):

        idx, samples = self.env.sample(self.batch_size)
        v = np.repeat(np.expand_dims(self.env.vs[idx],0), self.batch_size, axis=0)
        mcr = np.zeros((self.batch_size,1))
        if samples is not None:
            targets = self.critic.get_targets_dqn(samples['r1'][:,idx], samples['t'], samples['s1'], v, v)
            inputs = [samples['s0'], samples['a'], v, v, targets, mcr]
            metrics = self.critic.train(inputs)
            for i, name in enumerate(self.critic.metrics_names):
                self.env.train_metrics[name][idx] += np.mean(np.squeeze(metrics[i]))
            self.critic.target_train()

    def make_input(self, state):
        v = self.env.vs[self.env.idx]
        input = [np.expand_dims(i, axis=0) for i in [state, v, v]]
        return input

    def act(self, exp):
        input = self.make_input(exp['s0'])
        actionProbs = self.critic.actionProbs(input)[0].squeeze()
        action = np.random.choice(range(self.env.action_dim), p=actionProbs)
        action = np.expand_dims(action, axis=1)
        exp['a'] = action
        return exp

    def imitate(self):

        if self.demo != 0 and self.env_step % self.demo_freq == 0:

            demo, true_task = self.env_test.get_demo()

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
        self.stats, self.short_stats = self.env.get_stats()
        self.stats['step'] = self.env_step
        self.short_stats['step'] = self.env_step
        for i, f in enumerate(self.env.feat):
            for name, metric in self.env.train_metrics.items():
                self.stats[name + str(f)] = float("{0:.3f}".format(metric[i]))
                metric[i] = 0
        for key in sorted(self.stats.keys()):
            self.logger.logkv(key, self.stats[key])
        for key in sorted(self.short_stats.keys()):
            self.short_logger.logkv(key, self.short_stats[key])
        self.logger.dumpkvs()
        self.short_logger.dumpkvs()
        self.save_model()





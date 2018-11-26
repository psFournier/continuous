import numpy as np
RENDER_TRAIN = False
TARGET_CLIP = True
INVERTED_GRADIENTS = True
from networks import CriticDQNGM, CriticDQNGMall
from agents.agent import Agent
from networks.critic2 import Critic

class DQNGM(Agent):
    def __init__(self, args, env, env_test, logger, short_logger):
        super(DQNGM, self).__init__(args, env, env_test, logger, short_logger)
        self.init(args, env)

    def init(self, args ,env):

        self.critic = CriticDQNGM(args, env)
        self.env.train_metrics = {name: [0]*self.env.N for name in self.critic.metrics_dqn_names+self.critic.metrics_imit_names}
        self.rnd_demo = float(args['--rnd_demo'])
        self.demo = int(args['--demo'])
        self.prop_demo = float(args['--prop_demo'])
        self.freq_demo = int(args['--freq_demo'])

    def run(self):

        self.exp = self.env.reset()
        self.episode_step = 0

        self.env.idx = self.env.sample_v(self.exp['s0'])
        self.episode_task = 1

        if self.demo != 0:
            for _ in range(self.demo * 100):
                demo, true_task = self.env_test.get_demo()
                self.env.process_trajectory(demo)

        try:
            while self.env_step < self.max_steps:

                if self.env_step % self.freq_demo == 0 and self.demo != 0:
                    for _ in range(int(self.freq_demo * self.prop_demo)):
                        self.imit()

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
                    if self.env.done(self.exp) or self.episode_task >= self.ep_tasks:
                        self.exp = self.env.reset()
                        self.episode_task = 0
                    else:
                        self.exp['s0'] = self.exp['s1']
                        self.exp['r0'] = self.exp['r1']
                        self.exp['t'] = 0
                    self.episode_step = 0
                    self.env.idx = self.env.sample_v(self.exp['s0'])
                    self.episode_task += 1
                else:
                    self.exp['s0'] = self.exp['s1']
                    self.exp['r0'] = self.exp['r1']
                if self.env_step % self.eval_freq == 0:
                    self.log()



        except KeyboardInterrupt:
            print("Keybord interruption")

    def train(self):

        idx, samples = self.env.sample(self.batch_size)
        v = np.repeat(np.expand_dims(self.env.vs[idx],0), self.batch_size, axis=0)
        if samples is not None:
            targets = self.critic.get_targets_dqn(samples['r1'][:,idx], samples['t'], samples['s1'], v, v)
            inputs = [samples['s0'], samples['a'], v, v, targets]
            metrics = self.critic.train(inputs)
            for i, name in enumerate(self.critic.metrics_dqn_names):
                self.env.train_metrics[name][idx] += np.mean(np.squeeze(metrics[i]))
            self.critic.target_train()

    def imit(self):

        idx, samples = self.env.sampleT(self.batch_size)
        v = np.repeat(np.expand_dims(self.env.vs[idx], 0), self.batch_size, axis=0)
        if samples is not None:
            targets = self.critic.get_targets_dqn(samples['r1'][:, idx], samples['t'], samples['s1'], v, v)
            inputs = [samples['s0'], samples['a'], v, v, targets, samples['mcr'][:, [idx]]]
            metrics = self.critic.imit(inputs)
            metrics[2] = 1/(np.where(np.argmax(metrics[2], axis=1) == samples['a'][:, 0],
                                     0.99, 0.01 / self.env.action_dim))
            for i, name in enumerate(self.critic.metrics_imit_names):
                self.env.train_metrics[name][idx] += np.mean(np.squeeze(metrics[i]))
            self.critic.target_train()

    def act(self, exp):
        v = self.env.vs[self.env.idx]
        input = [np.expand_dims(i, axis=0) for i in [exp['s0'], v, v]]
        actionProbs = self.critic.actionProbs(input)[0].squeeze()
        if np.random.rand() < 1 - (1 - 0.01) * max(1, self.env_step / 1e4):
            action = np.random.randint(self.env.action_dim)
        else:
            action = np.argmax(actionProbs)
        # action = np.random.choice(range(self.env.action_dim), p=actionProbs)
        action = np.expand_dims(action, axis=1)
        exp['a'] = action
        return exp

    def log(self):
        self.stats, self.short_stats = self.env.get_stats()
        self.stats['step'] = self.env_step
        self.short_stats['step'] = self.env_step
        for i, f in enumerate(self.env.feat):
            for name, metric in self.env.train_metrics.items():
                self.stats[name + str(f)] = float("{0:.3f}".format(metric[i]))
                metric[i] = 0
            self.env.queues[i].init_stat()
        for key in sorted(self.stats.keys()):
            self.logger.logkv(key, self.stats[key])
        for key in sorted(self.short_stats.keys()):
            self.short_logger.logkv(key, self.short_stats[key])
        self.logger.dumpkvs()
        self.short_logger.dumpkvs()
        self.save_model()





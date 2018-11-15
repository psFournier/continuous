import numpy as np
from gym import Wrapper
from samplers.competenceQueue import CompetenceQueue
from buffers import MultiTaskReplayBuffer, ReplayBuffer
import math

from envs.playroom import Actions

class PlayroomGM(Wrapper):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env)

        self.gamma = float(args['--gamma'])
        self.eps1 = float(args['--eps1'])
        self.eps2 = float(args['--eps2'])
        self.selfImit = bool(int(args['--selfImit']))

        # self.tasks_feat = [[i] for i in range(12)]
        self.tasks_feat = [[i] for i in range(2, 8)]

        self.Ntasks = len(self.tasks_feat)
        self.mask = None
        self.task = None
        self.goal = None
        self.queues = [CompetenceQueue() for _ in self.tasks_feat]

        self.run_metrics_names = ['envsteps', 'attempts', 'trainsteps']
        self.run_metrics = {name: [0] * self.Ntasks for name in self.run_metrics_names}

        self.samples_metrics_names = ['termstates', 'tutorsamples']
        self.samples_metrics = {name: [0] * self.Ntasks for name in self.samples_metrics_names}

        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)
        self.r_done = 100
        self.r_notdone = 0
        self.terminal = True
        self.minQ = self.r_notdone / (1 - self.gamma)
        self.maxQ = self.r_done if self.terminal else self.r_done / (1 - self.gamma)

        self.names = ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'pa', 'mcr', 'o']
        self.buffer = MultiTaskReplayBuffer(limit=int(1e6), Ntasks=self.Ntasks, names=self.names)

    def step(self, exp):
        self.run_metrics['envsteps'][self.task] += 1
        exp['o'] = 0
        exp['s1'] = self.env.step(exp['a'])[0]
        indices = np.where(self.mask)
        goal = self.goal[indices]
        s1_proj = exp['s1'][indices]
        if (s1_proj == goal).all():
            exp['t'] = self.terminal
        else:
            exp['t'] = False
        return exp

    def eval_exp(self, exp):
        exp['rs'], exp['ts'] = [], []
        for g, m in zip(exp['goals'], exp['masks']):
            indices = np.where(m)
            goal = g[indices]
            s1_proj = exp['s1'][indices]
            if (s1_proj == goal).all():
                exp['ts'].append(self.terminal)
                exp['rs'].append(self.r_done)
            else:
                exp['ts'].append(False)
                exp['rs'].append(self.r_notdone)
        return exp

    def end_episode(self, episode):

        T = episode[-1]['t']
        self.queues[self.task].append(T)
        self.run_metrics['attempts'][self.task] += 1

        tasks = range(self.Ntasks)
        goals = [self.goal if t==self.task else None for t in tasks]
        self.process_trajectory(episode, tasks, goals, with_mcr=self.selfImit)

    def process_trajectory(self, trajectory, tasks, goals, with_mcr=False):

        mcrs = [np.zeros(1)] * self.Ntasks
        masks = [self.task2mask(task) for task in tasks]

        for exp in reversed(trajectory):

            exp['tasks'] = []
            exp['mcrs'] = []
            exp['goals'] = []
            exp['masks'] = []

            for i, task in enumerate(tasks):

                if goals[i] is None and (exp['s1'][np.where(masks[i])] != exp['s0'][np.where(masks[i])]).any():
                    goals[i] = exp['s1']
                if goals[i] is not None:
                    exp['tasks'].append(task)
                    exp['goals'].append(goals[i])
                    exp['masks'].append(masks[i])

            if exp['tasks']:
                exp = self.eval_exp(exp)
                for i, task in enumerate(exp['tasks']):
                    if with_mcr:
                        mcrs[task] = mcrs[task] * self.gamma + exp['rs'][i]
                        exp['mcrs'].append(mcrs[task])
                    else:
                        exp['mcrs'].append(np.zeros(1))
                self.buffer.append(exp.copy())

    def reset(self):
        state = self.env.reset(random=False)
        return state

    def sample_task(self, state):
        remaining_tasks = [i for i in range(self.Ntasks) if state[self.tasks_feat[i]] != 1]
        probs = self.get_probs(idxs=remaining_tasks, eps=self.eps1)
        self.task = np.random.choice(remaining_tasks, p=probs)
        self.goal = self.sample_goal(self.task, state)
        self.mask = self.task2mask(self.task)

    def sample_goal(self, task, state):

        goal = state.copy()
        while not (goal != state).any():
            for f in self.tasks_feat[task]:
                goal[f] = np.random.randint(self.state_low[f], self.state_high[f] + 1)

        return goal

    def get_demo(self):

        demo = []
        exp = {}
        exp['s0'] = self.env.reset(random=False)
        task = np.random.choice([4,7])
        while True:
            a, done = self.opt_action(task)
            if done:
                break
            else:
                exp['a'] = np.expand_dims(a, axis=1)
                exp['s1'] = self.env.step(exp['a'], True)[0]
                exp['pa'] = 1
                exp['o'] = 1
                demo.append(exp.copy())
                exp['s0'] = exp['s1']

        return demo, task

    def opt_action(self, t):

        # if t == 4:
        #     obj = self.env.chest1
        #     if obj.s == 1:
        #         return -1, True
        #     else:
        #         return self.env.touch(obj)
        #
        # elif t== 4:
        #     obj = self.env.chest1
        #     if obj.s == 1:
        #         return -1, True
        #     else:
        #         if self.env.door1.s == 1:
        #             return self.env.touch(obj)
        #         elif self.env.keyDoor1.s == 1:
        #             return self.env.touch(self.env.door1)
        #         else:
        #             return self.env.touch(self.env.keyDoor1)
        #
        # elif t == 7:
        #     obj = self.env.chest3
        #     if obj.s == 1:
        #         return -1, True
        #     else:
        #         if self.env.door3.s == 1:
        #             return self.env.touch(obj)
        #         elif self.env.keyDoor3.s == 1:
        #             return self.env.touch(self.env.door3)
        #         else:
        #             return self.env.touch(self.env.keyDoor3)
        # else:
        #     raise RuntimeError
        pass

    def sample(self, batchsize):
        probs = self.get_probs(idxs=range(self.Ntasks), eps=self.eps2)
        task = np.random.choice(self.Ntasks, p=probs)
        samples = self.buffer.sample(batchsize, task)
        if samples is not None:
            self.run_metrics['trainsteps'][task] += 1
            self.samples_metrics['termstates'][task] += np.mean(samples['t'])
            self.samples_metrics['tutorsamples'][task] += np.mean(samples['o'])

        return task, samples

    def get_stats(self):
        stats = {}
        short_stats = {}
        for i, task in enumerate(self.tasks_feat):
            self.queues[i].update()
            short_stats['CP_{}'.format(task)] = float("{0:.3f}".format(self.CPs[i]))
            short_stats['C_{}'.format(task)] = float("{0:.3f}".format(self.Cs[i]))

            for name, metric in self.samples_metrics.items():
                stats[name+'_{}'.format(task)] = float("{0:.3f}".format(
                    metric[i] / (self.run_metrics['trainsteps'][i] + 0.00001)
                ))
                metric[i] = 0

            for name, metric in self.run_metrics.items():
                stats[name+'_{}'.format(task)] = float("{0:.3f}".format(metric[i]))
                metric[i] = 0

        return stats, short_stats

    def task2mask(self, idx):
        res = np.zeros(shape=self.state_dim)
        res[self.tasks_feat[idx]] = 1
        return res

    def mask2task(self, mask):
        return list(np.where(mask)[0])

    def get_probs(self, idxs, eps):
        cps = self.CPs
        vals = [cps[idx] for idx in idxs]
        l = len(vals)
        s = np.sum(vals)
        if s == 0:
            probs = [1 / l] * l
        else:
            probs = [eps / l + (1 - eps) * v / s for v in vals]
        return probs

    @property
    def CPs(self):
        return [np.maximum(abs(q.CP + 0.05) - 0.05, 0) for q in self.queues]

    @property
    def Cs(self):
        return [q.C_avg[-1] for q in self.queues]

    def done(self, exp):
        return all([exp['s1'][f] == 1 for f in self.tasks_feat])

    @property
    def state_dim(self):
        return 8,

    @property
    def goal_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 5

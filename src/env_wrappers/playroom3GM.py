import numpy as np
from gym import Wrapper
from buffers import MultiTaskReplayBuffer, ReplayBuffer
from samplers.competenceQueue import CompetenceQueue
import math


class Playroom3GM(Wrapper):
    def __init__(self, env, args):
        super(Playroom3GM, self).__init__(env)

        self.gamma = float(args['--gamma'])
        self.theta1 = float(args['--theta1'])
        self.theta2 = float(args['--theta2'])
        self.selfImit = bool(int(args['--selfImit']))
        self.tutorTask = args['--tutorTask']

        self.tasks = [o.name for o in self.env.objects]
        self.Ntasks = len(self.tasks)
        self.tasks_feat = [[4], [7], [10]]
        self.mask = None
        self.task = None
        self.goal = None
        self.queues = [CompetenceQueue() for _ in self.tasks]
        self.envsteps = [0 for _ in self.tasks]
        self.trainsteps = [0 for _ in self.tasks]
        self.offpolicyness = [0 for _ in self.tasks]
        self.termstates = [0 for _ in self.tasks]
        self.attempts = [0 for _ in self.tasks]
        # self.foreval = [False for _ in self.tasks]
        self.update_interests()

        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)
        self.r_done = 100
        self.r_notdone = 0
        self.terminal = True
        self.minQ = self.r_notdone / (1 - self.gamma)
        self.maxQ = self.r_done if self.terminal else self.r_done / (1 - self.gamma)

        self.names = ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'pa', 'mcr', 'task']
        self.buffer = MultiTaskReplayBuffer(limit=int(1e6), Ntasks=self.Ntasks, names=self.names)


    def step(self, exp):
        self.envsteps[self.task] += 1
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
        self.attempts[self.task] += 1
        # self.foreval[self.task] = (self.attempts[self.task] % 10 == 0)

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
        self.update_interests()
        self.task = self.sample_task(state)
        self.goal = self.sample_goal(self.task, state)
        self.mask = self.task2mask(self.task)

        return state

    def sample_task(self, state):

        task = np.random.choice(self.Ntasks, p=self.probs1)
        features = self.tasks_feat[task]
        while all([state[f] == self.state_high[f] for f in features]):
            task = np.random.choice(self.Ntasks, p=self.probs1)
            features = self.tasks_feat[task]
        return task

    def sample_goal(self, task, state):

        goal = state.copy()
        while not (goal != state).any():
            for f in self.tasks_feat[task]:
                goal[f] = np.random.randint(state[f], self.state_high[f] + 1)

        return goal

    def sample_tutor_task(self):

        if self.tutorTask == '2':
            task = 2
        elif self.tutorTask == 'rnd':
            task = np.random.randint(self.Ntasks)
        else:
            raise RuntimeError

        return task

    def sample_tutor_goal(self, task):

        features = self.tasks_feat[task]
        goal = self.state_high[features[0]]

        return goal

    def sample(self, batchsize):

        task = np.random.choice(self.Ntasks, p=self.probs2)
        samples = self.buffer.sample(batchsize, task)
        if samples is not None:
            self.trainsteps[task] += 1
            self.termstates[task] += np.mean(samples['t'])

        return task, samples

    def get_stats(self):
        stats = {}
        for i, task in enumerate(self.tasks):
            stats['I_{}'.format(task)] = float("{0:.3f}".format(self.interests[i]))
            stats['CP_{}'.format(task)] = float("{0:.3f}".format(self.CPs[i]))
            stats['C_{}'.format(task)] = float("{0:.3f}".format(self.Cs[i]))

            stats['envstep_{}'.format(task)] = float("{0:.3f}".format(self.envsteps[i]))
            stats['trainstep_{}'.format(task)] = float("{0:.3f}".format(self.trainsteps[i]))
            stats['attempts_{}'.format(task)] = float("{0:.3f}".format(self.attempts[i]))
            stats['offpolicyness_{}'.format(task)] = float("{0:.3f}".format(
                self.offpolicyness[i] / (self.trainsteps[i] + 0.00001)))
            stats['termstates_{}'.format(task)] = float("{0:.3f}".format(
                self.termstates[i] / (self.trainsteps[i] + 0.00001)))
            self.offpolicyness[i] = 0
            self.envsteps[i] = 0
            self.trainsteps[i] = 0
            self.attempts[i] = 0
            self.termstates[i] = 0
        return stats

    def task2mask(self, idx):
        res = np.zeros(shape=self.state_dim)
        res[self.tasks_feat[idx]] = 1
        return res

    def mask2task(self, mask):
        return list(np.where(mask)[0])

    def update_interests(self):

        minCP = min(self.CPs)
        maxCP = max(self.CPs)
        widthCP = maxCP - minCP
        self.interests = [(cp - minCP) / (widthCP + 0.0001) for cp in self.CPs]

        espilon = 0.4

        interests1 = [math.pow(i, self.theta1) for i in self.interests]
        sumI1 = np.sum(interests1)
        self.probs1 = [espilon / self.Ntasks + (1 - espilon) * i / sumI1 for i in interests1]

        interests2 = [math.pow(i, self.theta2) for i in self.interests]
        sumI2 = np.sum(interests2)
        self.probs2 = [espilon / self.Ntasks + (1 - espilon) * i / sumI2 for i in interests2]



    @property
    def CPs(self):
        return [np.maximum(abs(q.CP + 0.1/math.sqrt(2)) - 0.1/math.sqrt(2), 0) for q in self.queues]

    @property
    def Cs(self):
        return [q.C_avg[-1] for q in self.queues]


    @property
    def state_dim(self):
        return 11,

    @property
    def goal_dim(self):
        return 11,

    @property
    def action_dim(self):
        return 6



    # def update_buffer(self):
    #     self._update_buffer(start=1)
    #
    # def _update_buffer(self, start):
    #     if start >= self.buffer._it_sum._capacity:
    #         task = self.buffer._it_sum._tasks[start]
    #         if task is not None:
    #             val = self.interests[task]
    #         else:
    #             val = 0
    #         self.buffer._it_sum._value[start] = val
    #     else:
    #         val = self.buffer._it_sum._operation(self._update_buffer(2*start), self._update_buffer(2*start+1))
    #         self.buffer._it_sum._value[start] = val
    #     return val
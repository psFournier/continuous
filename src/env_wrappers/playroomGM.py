import numpy as np
from gym import Wrapper
from samplers.competenceQueue import CompetenceQueue
from buffers import MultiTaskReplayBuffer, ReplayBuffer
import math

class PlayroomGM(Wrapper):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env)

        self.gamma = float(args['--gamma'])
        self.theta = float(args['--theta'])
        self.htr = int(args['--htr'])

        # self.tasks = [o.name for o in self.env.objects]
        self.tasks_feat = [[i] for i in range(12)]
        self.Ntasks = len(self.tasks_feat)
        self.mask = None
        self.task = None
        self.goal = None
        self.queues = [CompetenceQueue() for _ in self.tasks_feat]
        self.steps = [0 for _ in self.tasks_feat]
        self.attempts = [0 for _ in self.tasks_feat]
        self.foreval = [False for _ in self.tasks_feat]
        self.update_interests()

        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)
        self.r_done = 100
        self.r_notdone = 0
        self.terminal = True
        self.minQ = self.r_notdone / (1 - self.gamma)
        self.maxQ = self.r_done if self.terminal else self.r_done / (1 - self.gamma)

        self.names = ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'mcr', 'task']
        self.buffer = MultiTaskReplayBuffer(limit=int(1e6), Ntasks=self.Ntasks, names=self.names)

    def step(self, exp):
        self.steps[self.task] += 1
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

        if self.foreval[self.task]:
            T = episode[-1]['t']
            self.queues[self.task].append(T)
        self.attempts[self.task] += 1
        self.foreval[self.task] = (self.attempts[self.task] % 10 == 0)

        tasks = [self.task]
        goals = [self.goal]
        othertasks = [i for i in range(self.Ntasks) if i != self.task]

        if self.htr == 1:
            tasks += [np.random.choice(othertasks)]
            goals += [None]
        elif self.htr == 2:
            otherprobs = [self.interests[i] for i in othertasks]
            tasks += [np.random.choice(othertasks, otherprobs)]
            goals += [None]
        elif self.htr == 3:
            tasks += othertasks
            goals += [None] * len(othertasks)

        masks = [self.task2mask(task) for task in tasks]
        mcrs = [np.zeros(1) for _ in tasks]

        for exp in reversed(episode):

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
                for i in range(len(exp['tasks'])):
                    mcrs[i] = mcrs[i] * self.gamma + exp['rs'][i]
                    exp['mcrs'].append(mcrs[i])
                self.buffer.append(exp.copy())

    def reset(self):

        state = self.env.reset(random=False)
        self.update_interests()
        self.task = self.sample_task(state)
        self.goal = self.sample_goal(self.task, state)
        self.mask = self.task2mask(self.task)

        return state

    def sample_task(self, state):

        task = np.random.choice(self.Ntasks, p=self.interests)
        features = self.tasks_feat[task]
        while all([state[f] == self.state_high[f] for f in features]):
            task = np.random.choice(self.Ntasks, p=self.interests)
            features = self.tasks_feat[task]
        return task

    def sample_goal(self, task, state):

        goal = state.copy()
        while not (goal != state).any():
            for f in self.tasks_feat[task]:
                goal[f] = np.random.randint(state[f], self.state_high[f] + 1)

        return goal

    def sample(self, batchsize):

        task = np.random.choice(self.Ntasks, p=self.interests)
        samples = self.buffer.sample(batchsize, task)

        return samples

    def get_stats(self):
        stats = {}
        for i, task in enumerate(self.tasks_feat):
            stats['step_{}'.format(task)] = float("{0:.3f}".format(self.steps[i]))
            stats['I_{}'.format(task)] = float("{0:.3f}".format(self.interests[i]))
            stats['CP_{}'.format(task)] = float("{0:.3f}".format(self.CPs[i]))
            stats['C_{}'.format(task)] = float("{0:.3f}".format(self.Cs[i]))
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
        espilon = 0.4
        Ntasks = len(self.CPs)

        if widthCP <= 0.01:
            minC = min(self.Cs)
            maxC = max(self.Cs)
            widthC = maxC - minC
            Cs = [math.pow(1 - (c - minC) / (widthC + 0.0001), self.theta) for c in self.Cs]
            sumC = np.sum(Cs)
            self.interests = [espilon / Ntasks + (1 - espilon) * c / sumC for c in Cs]
        else:
            CPs = [math.pow((cp - minCP) / widthCP, self.theta) for cp in self.CPs]
            sumCP = np.sum(CPs)
            self.interests = [espilon / Ntasks + (1 - espilon) * cp / sumCP for cp in CPs]
    @property
    def CPs(self):
        return [abs(q.CP) for q in self.queues]

    @property
    def Cs(self):
        return [q.C_avg for q in self.queues]

    @property
    def state_dim(self):
        return 12,

    @property
    def goal_dim(self):
        return 12,

    @property
    def action_dim(self):
        return 7
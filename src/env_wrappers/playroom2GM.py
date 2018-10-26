import numpy as np
from gym import Wrapper
from buffers import MultiTaskReplayBuffer, ReplayBuffer
from samplers.competenceQueue import CompetenceQueue
import math


class Playroom2GM(Wrapper):
    def __init__(self, env, args):
        super(Playroom2GM, self).__init__(env)

        self.gamma = float(args['--gamma'])
        self.theta = float(args['--theta'])

        self.tasks = [o.name for o in self.env.objects]
        self.Ntasks = len(self.tasks)
        self.tasks_feat = [[4], [7], [10]]
        self.mask = None
        self.task = None
        self.goal = None
        self.queues = [CompetenceQueue() for _ in self.tasks]
        self.steps = [0 for _ in self.tasks]
        self.attempts = [0 for _ in self.tasks]
        self.foreval = [False for _ in self.tasks]
        self.update_interests()

        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)
        self.r_done = 100
        self.r_notdone = 0
        self.terminal = True
        self.minQ = self.r_notdone / (1 - self.gamma)
        self.maxQ = self.r_done if self.terminal else self.r_done / (1 - self.gamma)

        self.names = ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'task']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names.copy())

    def step(self, exp):
        self.steps[self.task] += 1
        exp['g'] = self.goal
        exp['m'] = self.mask
        exp['task'] = self.task
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def eval_exp(self, exp):
        indices = np.where(exp['m'])
        goal = exp['g'][indices]
        s1_proj = exp['s1'][indices]
        # s0_proj = exp['s0'][indices]
        if (s1_proj == goal).all():
            exp['t'] = self.terminal
            exp['r'] = self.r_done
        else:
            exp['t'] = False
            exp['r'] = self.r_notdone
        return exp

    def end_episode(self, episode):

        if self.foreval[self.task]:
            T = episode[-1]['t']
            self.queues[self.task].append(T)
        self.attempts[self.task] += 1
        self.foreval[self.task] = (self.attempts[self.task] % 10 == 0)

        goals = [self.goal]
        masks = [self.mask]
        tasks = [self.task]

        for expe in reversed(episode):

            for j, (g, m, t) in enumerate(zip(goals, masks, tasks)):

                if t != self.task:
                    expe['g'] = g
                    expe['m'] = m
                    expe['task'] = t
                    expe = self.eval_exp(expe)
                self.buffer.append(expe.copy())

            for task, _ in enumerate(self.tasks):

                if all([task != t for t in tasks]):

                    mask = self.task2mask(task)
                    s1m = expe['s1'][np.where(m)]
                    s0m = expe['s0'][np.where(m)]
                    if (s1m != s0m).any():
                        goal = expe['s1']
                        expe['g'] = goal
                        expe['m'] = mask
                        expe['task'] = task
                        expe = self.eval_exp(expe)
                        self.buffer.append(expe.copy())
                        goals.append(goal)
                        masks.append(mask)
                        tasks.append(task)


    def reset(self):

        state = self.env.reset(random=False)
        self.update_interests()
        self.task, self.goal = self.sample_task_goal(state)
        self.mask = self.task2mask(self.task)

        return state

    def sample_task_goal(self, state):

        task = np.random.choice(self.Ntasks, p=self.interests)
        features = self.tasks_feat[task]
        while all([state[f] == self.state_high[f] for f in features]):
            task = np.random.choice(self.Ntasks, p=self.interests)
            features = self.tasks_feat[task]

        goal = state.copy()
        while not (goal != state).any():
            for f in features:
                goal[f] = np.random.randint(state[f], self.state_high[f] + 1)

        return task, goal

    def get_stats(self):
        stats = {}
        for i, task in enumerate(self.tasks):
            stats['step_{}'.format(task)] = float("{0:.3f}".format(self.steps[i]))
            stats['attempts_{}'.format(task)] = float("{0:.3f}".format(self.attempts[i]))
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
        CPs = [math.pow((cp - minCP) / (widthCP + 0.0001), self.theta) for cp in self.CPs]
        sumCP = np.sum(CPs)
        Ntasks = len(self.CPs)
        espilon = 0.4
        if sumCP == 0:
            self.interests = [1 / Ntasks for _ in CPs]
        else:
            self.interests = [espilon / Ntasks + (1 - espilon) * cp / sumCP for cp in CPs]

    @property
    def CPs(self):
        return [abs(q.CP) for q in self.queues]

    @property
    def Cs(self):
        return [q.C_avg for q in self.queues]


    @property
    def state_dim(self):
        return 11,

    @property
    def goal_dim(self):
        return 11,

    @property
    def action_dim(self):
        return 6
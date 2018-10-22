import numpy as np
from .base import CPBased

class Playroom2GM(CPBased):
    def __init__(self, env, args):
        super(Playroom2GM, self).__init__(env, args, ['pos'] + [obj.name for obj in env.objects])
        self.mask = None
        self.init()
        self.obj_feat = [[0,1], [4], [7], [10]]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)
        self.r_done = 50
        self.r_notdone = -0.5
        self.terminal = True
        self.minQ = self.r_notdone / (1 - self.gamma)
        self.maxQ = self.r_done if self.terminal else self.r_done / (1 - self.gamma)

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
        if (s1_proj == goal).all():
            exp['t'] = self.terminal
            exp['r'] = self.r_done
        else:
            exp['t'] = False
            exp['r'] = self.r_notdone
        return exp

    def end_episode(self, trajectory):
        augmented_episode = []
        MCR = 0
        T = trajectory[-1]['t']

        for exp in reversed(trajectory):
            MCR = MCR * self.gamma + exp['r']
            exp['mcr'] = MCR
            augmented_episode.append(exp.copy())
        self.queues[self.task].append(MCR, T)

        return augmented_episode

    def sample_goal(self, task):
        features = self.obj_feat[task]
        goal = self.init_state.copy()
        while not (goal != self.init_state).any():
            for f in features:
                goal[f] = np.random.randint(self.state_low[f], self.state_high[f] + 1)
        return goal

    def reset(self):
        self.task = self.sample_task()
        self.mask = self.task2mask(self.task)
        self.goal = self.sample_goal(self.task)
        state = self.env.reset()
        return state

    def task2mask(self, idx):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[idx]] = 1
        return res

    def mask2task(self, mask):
        return list(np.where(mask)[0])

    def augment_demo(self, demo):
        return demo

    def augment_samples(self, samples):
        return samples

    def augment_tutor_samples(self, samples):
        return samples

    @property
    def state_dim(self):
        return 11,

    @property
    def goal_dim(self):
        return 11,

    @property
    def action_dim(self):
        return 6
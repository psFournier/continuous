import numpy as np
from ..base import CPBased

class PlayroomGM(CPBased):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env, args, [obj.name for obj in env.objects])
        self.mask = None
        self.init()
        self.obj_feat = [[i] for i in range(2, 11)]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.init)
        self.minQ = 0
        self.maxQ = 100

    def step(self, exp):
        self.steps[self.task] += 1
        exp['g'] = self.goal
        exp['m'] = self.mask
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def eval_exp(self, exp):
        indices = np.where(exp['m'])
        goal = exp['g'][indices]
        s1_proj = exp['s1'][indices]
        if (s1_proj == goal).all():
            exp['t'] = True
            exp['r'] = 0
        else:
            exp['t'] = False
            exp['r'] = -1
        return exp

    def end_episode(self, trajectory):
        MCR = 0
        T = trajectory[-1]['t']
        for exp in reversed(trajectory):
            MCR = MCR * self.gamma + exp['r'] + 1 + exp['t'] * 99
        self.queues[self.task].append(MCR, T)

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

    def augment_episode(self, episode):
        return episode

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
        return 7
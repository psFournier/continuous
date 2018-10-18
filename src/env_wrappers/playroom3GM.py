import numpy as np
from .base import CPBased
from buffers import ReplayBuffer, PrioritizedReplayBuffer, goalPrioritizedBuffer

class Playroom3GM(CPBased):
    def __init__(self, env, args):
        super(Playroom3GM, self).__init__(env, args, ['pos'] + [obj.name for obj in env.objects])
        self.mask = None
        self.init()
        self.obj_feat = [[0,1], [4], [7], [10]]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.initstate = np.array(self.env.initstate)
        self.r_done = 0
        self.r_notdone = -1
        self.terminal = True
        self.minQ = self.r_notdone / (1 - self.gamma)
        self.maxQ = self.r_done if self.terminal else self.r_done / (1 - self.gamma)
        self.names = ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'task']
        self.buffer = goalPrioritizedBuffer(limit=int(1e6), names=self.names.copy(), args=args)

    def update_buffer(self):
        self._update_buffer(start=1)

    def _update_buffer(self, start):
        if start >= self.buffer._it_sum._capacity:
            task = self.buffer._it_sum._tasks[start]
            if task is not None:
                val = self.interests[task]
            else:
                val = 0
            self.buffer._it_sum._value[start] = val
        else:
            val = self.buffer._it_sum._operation(self._update_buffer(2*start), self._update_buffer(2*start+1))
            self.buffer._it_sum._value[start] = val
        return val


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
        MCR = 0
        T = trajectory[-1]['t']
        for exp in reversed(trajectory):
            MCR = MCR * self.gamma + exp['r'] + 1 + exp['t'] * 99
        self.queues[self.task].append(MCR, T)

    def sample_goal(self, task):
        features = self.obj_feat[task]
        goal = self.initstate.copy()
        while not (goal != self.initstate).any():
            for f in features:
                goal[f] = np.random.randint(self.state_low[f], self.state_high[f] + 1)
        return goal

    def reset(self):
        self.task = self.sample_task()
        self.update_buffer()
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
        return 6
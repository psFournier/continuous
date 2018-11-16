import numpy as np
from gym import Wrapper
from samplers.competenceQueue import CompetenceQueue
from buffers import MultiReplayBuffer, ReplayBuffer
import math

from envs.playroom import Actions

class PlayroomGM(Wrapper):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env)

        self.gamma = float(args['--gamma'])
        self.eps1 = float(args['--eps1'])
        self.eps2 = float(args['--eps2'])

        self.feat = np.array(range(2,8))
        self.N = self.feat.shape[0]
        vs = np.zeros(shape=(self.N, self.state_dim[0]))
        vs[np.arange(self.N), self.feat] = 1
        self.vs = vs / np.sum(vs, axis=1, keepdims=True)

        # self.g = None
        self.v = None
        self.idx = None
        self.queues = [CompetenceQueue()] * self.N

        self.run_metrics_names = ['envsteps', 'attempts', 'trainsteps']
        self.run_metrics = {name: [0] * self.N for name in self.run_metrics_names}

        self.samples_metrics_names = ['termstates', 'tutorsamples']
        self.samples_metrics = {name: [0] * self.N for name in self.samples_metrics_names}

        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)

        self.names = ['s0', 'r0', 'a', 's1', 'r1', 't', 'u', 'o']
        self.buffer = MultiReplayBuffer(limit=int(1e6), Ntasks=self.N, names=self.names)

    def reset(self):
        exp = {}
        exp['s0'] = self.env.reset()
        exp['r0'] = - np.dot(self.vs, exp['s0'] != 1)
        return exp

    def sample_v(self, s):
        remaining_v = [i for i in range(self.N) if s[self.feat[i]] != 1]
        probs = self.get_probs(idxs=remaining_v, eps=self.eps1)
        idx = np.random.choice(remaining_v, p=probs)
        return idx

    def step(self, exp):
        exp['o'] = 0
        exp['s1'] = self.env.step(exp['a'])[0]
        exp['r1'] = -1 * np.dot(self.vs, exp['s1'] != 1)
        exp['t'] = 0
        exp['i'] = self.idx
        if exp['r1'][self.idx] == 0:
            exp['t'] = 1
        return exp

    def end_episode(self, episode):
        self.queues[self.idx].process_ep(episode)
        base_util = np.zeros(shape=(self.N,))
        base_util[self.idx] = 1
        self.process_trajectory(episode, base_util=base_util)

    def process_trajectory(self, trajectory, base_util=None):
        if base_util is None:
            u = np.zeros(shape=(self.N,))
        else:
            u = base_util
        for exp in reversed(trajectory):
            u_idx = np.where(exp['r1'] > exp['r0'])
            u[u_idx] += 1
            u_c = u.copy()
            exp['u'] = u_c
            if any(u_c!=0):
                self.buffer.append(exp.copy())

    def sample(self, batchsize):
        probs = self.get_probs(idxs=range(self.N), eps=self.eps2)
        idx = np.random.choice(self.N, p=probs)
        samples = self.buffer.sample(batchsize, idx)
        if samples is not None:
            self.queues[self.idx].process_samples(samples)
        return idx, samples

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

    def get_stats(self):
        stats = {}
        short_stats = {}
        for i, f in enumerate(self.feat):
            self.queues[i].update()
            for key, val in self.queues[i].get_short_stats().items():
                short_stats[key+str(f)] = val
            for key, val in self.queues[i].get_stats().items():
                stats[key+str(f)] = val
        return stats, short_stats

    def get_probs(self, idxs, eps):
        cps = [np.maximum(abs(q.CP + 0.05) - 0.05, 0) for q in self.queues]
        vals = [cps[idx] for idx in idxs]
        l = len(vals)
        s = np.sum(vals)
        if s == 0:
            probs = [1 / l] * l
        else:
            probs = [eps / l + (1 - eps) * v / s for v in vals]
        return probs

    def done(self, exp):
        return all([exp['s1'][f] == 1 for f in [[4], [7]]])

    @property
    def state_dim(self):
        return 8,

    @property
    def goal_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 5

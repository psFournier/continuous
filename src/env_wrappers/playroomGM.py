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
        self.eps3 = float(args['--eps3'])
        self.demo = int(args['--demo'])

        self.feat = np.array(range(2,8))
        self.N = self.feat.shape[0]
        vs = np.zeros(shape=(self.N, self.state_dim[0]))
        vs[np.arange(self.N), self.feat] = 1
        self.vs = vs / np.sum(vs, axis=1, keepdims=True)
        self.R = 100

        self.idx = None
        self.queues = [CompetenceQueue() for _ in range(self.N)]

        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)

        self.names = ['s0', 'r0', 'a', 's1', 'r1', 't', 'u', 'o', 'mcr']
        self.buffer = MultiReplayBuffer(limit=int(1e6), Ntasks=self.N, names=self.names)

    def reset(self):
        exp = {}
        exp['s0'] = self.env.reset()
        exp['r0'] = self.get_r(exp['s0'])
        exp['t'] = 0
        return exp

    def get_r(self, s):
        return self.R * np.dot(self.vs, s == 1)

    def sample_v(self, s):
        remaining_v = [i for i in range(self.N) if s[self.feat[i]] != 1]
        probs = self.get_probs(idxs=remaining_v, eps=self.eps1)
        idx = np.random.choice(remaining_v, p=probs)
        return idx

    def step(self, exp):
        exp['o'] = 0
        exp['s1'] = self.env.step(exp['a'])[0]
        exp['r1'] = self.get_r(exp['s1'])
        exp['t'] = 0
        exp['i'] = self.idx
        if exp['r1'][self.idx] == self.R:
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
        mcr = np.zeros(shape=(self.N,))
        for exp in reversed(trajectory):
            r_idx = np.where(exp['r1'] > exp['r0'])
            u[r_idx] += 1
            u_idx = np.where(u != 0)
            mcr[u_idx] = exp['r1'][u_idx] + self.gamma * mcr[u_idx]
            u_c = u.copy()
            exp['u'] = u_c
            exp['mcr'] = mcr
            if any(u_c!=0):
                self.buffer.append(exp.copy())

    def sample(self, batchsize):
        probs = self.get_probs(idxs=range(self.N), eps=self.eps2)
        idx = np.random.choice(self.N, p=probs)
        samples = self.buffer.sample(batchsize, idx)
        if samples is not None:
            self.queues[self.idx].process_samples(samples)
        return idx, samples

    def sampleT(self, batchsize):
        probs = self.get_probs(idxs=range(self.N), eps=self.eps3)
        idx = np.random.choice(self.N, p=probs)
        samples = self.buffer.sampleT(batchsize, idx)
        return idx, samples

    def get_demo(self):
        demo = []
        exp = {}
        exp['s0'] = self.env.reset()
        exp['r0'] = self.get_r(exp['s0'])
        exp['t'] = 0
        if self.demo == 1:
            task = np.random.choice([4])
        elif self.demo == 2:
            task = np.random.choice([4, 7])
        while True:
            a, done = self.opt_action(task)
            if done:
                demo[-1]['t'] = 1
                break
            else:
                exp['a'] = np.expand_dims(a, axis=1)
                exp['s1'] = self.env.step(exp['a'], True)[0]
                exp['r1'] = self.get_r(exp['s1'])
                exp['o'] = 1
                exp['t'] = 0
                demo.append(exp.copy())
                exp['s0'] = exp['s1']
                exp['r0'] = exp['r1']

        return demo, task

    def opt_action(self, t):

        if t== 4:
            obj = self.env.chest1
            if obj.s == 1:
                return -1, True
            else:
                if self.env.door1.s == 1:
                    return self.env.touch(obj)
                elif self.env.keyDoor1.s == 1:
                    return self.env.touch(self.env.door1)
                else:
                    return self.env.touch(self.env.keyDoor1)

        elif t == 7:
            obj = self.env.chest2
            if obj.s == 1:
                return -1, True
            else:
                if self.env.door2.s == 1:
                    return self.env.touch(obj)
                elif self.env.keyDoor2.s == 1:
                    return self.env.touch(self.env.door2)
                else:
                    return self.env.touch(self.env.keyDoor2)
        else:
            raise RuntimeError

    def get_stats(self):
        stats = {}
        short_stats = {}
        for i, f in enumerate(self.feat):
            self.queues[i].update()
            short = self.queues[i].get_short_stats()
            for key, val in short.items():
                short_stats[key+str(f)] = val
                stats[key + str(f)] = val
            long = self.queues[i].get_stats()
            for key, val in long.items():
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

import numpy as np
from gym import Wrapper
from samplers.competenceQueue import CompetenceQueue
from buffers import MultiReplayBuffer, ReplayBuffer

class PlayroomGM(Wrapper):
    def __init__(self, env, args):
        super(PlayroomGM, self).__init__(env)

        self.gamma = float(args['--gamma'])
        self.eps1 = float(args['--eps1'])
        self.eps2 = float(args['--eps2'])
        self.eps3 = float(args['--eps3'])
        self.demo = int(args['--demo'])
        self.deterministic = bool(int(args['--deter']))

        self.feat = np.array(range(2,8))
        self.N = self.feat.shape[0]
        vs = np.zeros(shape=(self.N, self.state_dim[0]))
        vs[np.arange(self.N), self.feat] = 1
        self.vs = vs / np.sum(vs, axis=1, keepdims=True)
        self.R = 100
        self.idx = -1
        self.v = np.zeros(shape=(self.state_dim[0], 1))
        self.g = np.ones(shape=(self.state_dim[0]))
        self.queues = [CompetenceQueue() for _ in range(self.N)]
        self.names = ['s0', 'r0', 'a', 's1', 'r1', 'g', 'v', 'o', 'u']
        self.buffer = ReplayBuffer(limit=int(1e5), names=self.names)

    def reset(self, exp):
        self.idx, self.v = self.sample_v(exp['s0'])
        exp['g'] = self.g
        exp['v'] = self.v
        return exp

    def get_r(self, s, g, v):
        return self.R * np.sum(np.multiply(v, s==g), axis=1, keepdims=True)

    def sample_v(self, s):
        remaining_v = [i for i in range(self.N) if s[self.feat[i]] != 1]
        probs = self.get_probs(idxs=remaining_v, eps=self.eps1)
        idx = np.random.choice(remaining_v, p=probs)
        v = self.vs[idx]
        return idx, v

    def end_episode(self, episode):
        term = episode[-1]['r1'][self.idx] == self.R
        self.queues[self.idx].process_ep(episode, term)
        base_util = np.zeros(shape=(self.N,))
        base_util[self.idx] = 1
        self.process_trajectory(episode, base_util=base_util)

    def process_trajectory(self, trajectory, base_util=None):
        if base_util is None:
            u = np.zeros(shape=(self.N,))
        else:
            u = base_util
        u = np.expand_dims(u, axis=1)
        # mcr = np.zeros(shape=(self.N,))
        for exp in reversed(trajectory):
            u = self.gamma * u
            u[np.where(exp['r1'] > exp['r0'])] = 1

            # u_idx = np.where(u != 0)
            # mcr[u_idx] = exp['r1'][u_idx] + self.gamma * mcr[u_idx]
            exp['u'] = u.squeeze()
            # exp['mcr'] = mcr
            if any(u!=0):
                self.buffer.append(exp.copy())

    # def sample(self, batchsize):
    #     probs = self.get_probs(idxs=range(self.N), eps=self.eps2)
    #     idx = np.random.choice(self.N, p=probs)
    #     samples = self.buffer.sample(batchsize, idx)
    #     if samples is not None:
    #         self.queues[idx].process_samples(samples)
    #     return idx, samples
    #
    # def sampleT(self, batchsize):
    #     probs = self.get_probs(idxs=range(self.N), eps=self.eps3)
    #     idx = np.random.choice(self.N, p=probs)
    #     samples = self.buffer.sampleT(batchsize, idx)
    #     if samples is not None:
    #         self.queues[idx].process_samplesT(samples)
    #     return idx, samples

    def get_demo(self):
        demo = []
        exp = {}
        exp['s0'] = self.env.reset()
        exp['r0'] = self.get_r(exp['s0'], self.g, self.vs).squeeze()
        exp['g'] = self.g
        if self.demo == 1:
            task = np.random.choice([4])
        elif self.demo == 2:
            task = np.random.choice([4, 7])
        exp['v'] = self.vs[task]
        while True:
            a, done = self.opt_action(task, deterministic=self.deterministic)
            if done:
                break
            else:
                exp['a'] = np.expand_dims(a, axis=1)
                exp['s1'] = self.env.step(exp['a'], True)[0]
                exp['r1'] = self.get_r(exp['s1'], self.g, self.vs).squeeze()
                exp['o'] = 1
                demo.append(exp.copy())
                exp['s0'] = exp['s1']
                exp['r0'] = exp['r1']

        return demo, task

    def opt_action(self, t, deterministic=False):

        if t== 4:
            obj = self.env.chest1
            if obj.s == 1:
                return -1, True
            else:
                if self.env.door1.s == 1:
                    return self.env.touch(obj, deterministic)
                elif self.env.keyDoor1.s == 1:
                    return self.env.touch(self.env.door1, deterministic)
                else:
                    return self.env.touch(self.env.keyDoor1, deterministic)

        elif t == 7:
            obj = self.env.chest2
            if obj.s == 1:
                return -1, True
            else:
                if self.env.door2.s == 1:
                    return self.env.touch(obj, deterministic)
                elif self.env.keyDoor2.s == 1:
                    return self.env.touch(self.env.door2, deterministic)
                else:
                    return self.env.touch(self.env.keyDoor2, deterministic)
        else:
            raise RuntimeError

    def get_stats(self):
        stats = {}
        for i, f in enumerate(self.feat):
            self.queues[i].update()
            for key, val in self.queues[i].get_stats().items():
                stats[key + str(f)] = val
            self.queues[i].init_stat()
        return stats

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

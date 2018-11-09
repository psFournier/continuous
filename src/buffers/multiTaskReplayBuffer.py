import numpy as np

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

class TaskRingBuffer(object):
    def __init__(self, limit):
        self._storage = []
        self._limit = limit
        self._start = 0
        self._next_idx = 0
        self._numsamples = 0

    def append(self, idx):
        if self._next_idx >= len(self._storage):
            self._storage.append(idx)
        else:
            self._storage[self._next_idx] = idx
        self._numsamples += 1
        self._next_idx = (self._next_idx + 1) % self._limit

    def pop(self):
        self._start += 1
        self._numsamples -= 1

    def sample(self, batch_size):
        res = []
        idxs = [np.random.randint(0, self._numsamples - 1) for _ in range(batch_size)]
        for i in idxs:
            idx = (self._start + i) % len(self._storage)
            res.append(self._storage[idx])
        return res

class MultiTaskReplayBuffer(object):
    def __init__(self, limit, Ntasks, names):
        self._storage = []
        self._next_idx = 0
        self._limit = limit
        self._taskBuffers = []
        self._names = names
        for _ in range(Ntasks):
            self._taskBuffers.append(TaskRingBuffer(limit))

    def __len__(self):
        return len(self._storage)

    def append(self, item):
        triplet = {'s0': item['s0'],
                   'a': item['a'],
                   's1': item['s1'],
                   'tasks': item['tasks'],
                   'pa': item['pa'],
                   'o': item['o']}
        if self._next_idx >= len(self._storage):
            self._storage.append(triplet)
        else:
            for t in self._storage[self._next_idx]['tasks']:
                self._taskBuffers[t].pop()
            self._storage[self._next_idx] = triplet
        for i, t in enumerate(item['tasks']):
            info = {'idx': self._next_idx,
                    'g': item['goals'][i],
                    'm': item['masks'][i],
                    'r': item['rs'][i],
                    't': item['ts'][i],
                    'mcr': item['mcrs'][i]}
            self._taskBuffers[t].append(info)
        self._next_idx = (self._next_idx + 1) % self._limit

    def sample(self, batchsize, task):
        exps = []
        res = None
        buffer = self._taskBuffers[task]
        if buffer._numsamples >= 100 * batchsize:
            infos = buffer.sample(batchsize)
            for info in infos:
                triplet = self._storage[info['idx']]
                dict = merge_two_dicts(triplet, info)
                exps.append(dict)
            res = {name: np.array([exp[name] for exp in exps]) for name in self._names}

        return res

class ToyMultiTaskReplayBuffer(object):
    def __init__(self, limit, Ntasks, names):
        self._storage = []
        self._next_idx = 0
        self._limit = limit
        self._taskBuffers = []
        self._names = names
        for _ in range(Ntasks):
            self._taskBuffers.append(TaskRingBuffer(limit))

    def __len__(self):
        return len(self._storage)

    def append(self, item):
        triplet = {'step': item['step'],
                   'tasks': item['tasks']}
        if self._next_idx >= len(self._storage):
            self._storage.append(triplet)
        else:
            for t in self._storage[self._next_idx]['tasks']:
                self._taskBuffers[t].pop()
            self._storage[self._next_idx] = triplet
        for i, t in enumerate(item['tasks']):
            info = {'idx': self._next_idx}
            self._taskBuffers[t].append(info)
        self._next_idx = (self._next_idx + 1) % self._limit

    def sample(self, batchsize, task):
        exps = []
        res = None
        buffer = self._taskBuffers[task]
        if buffer._numsamples >= 100 * batchsize:
            infos = buffer.sample(batchsize)
            for info in infos:
                triplet = self._storage[info['idx']]
                dict = merge_two_dicts(triplet, info)
                exps.append(dict)
            res = {name: np.array([exp[name] for exp in exps]) for name in self._names}

        return res

if __name__ == '__main__':

    B = ToyMultiTaskReplayBuffer(limit=5, Ntasks=3, names=['step', 'r'])
    for i in range(20):
        if i %2 == 0:
            tasks = [0, 1]
        else:
            tasks = [1, 2]
        B.append({'step': i, 'tasks': tasks})
        print('step ', i)
        print(B._storage)
        print(B._taskBuffers[0]._storage, B._taskBuffers[0]._start, B._taskBuffers[0]._numsamples)
        print(B._taskBuffers[1]._storage, B._taskBuffers[1]._start, B._taskBuffers[1]._numsamples)
        print(B._taskBuffers[2]._storage, B._taskBuffers[2]._start, B._taskBuffers[2]._numsamples)
        print('\n')

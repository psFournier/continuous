import numpy as np

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
            self._numsamples += 1
        else:
            self._storage[self._next_idx] = idx
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

class TaskReplayBuffer(object):
    def __init__(self, limit, tasks, names):
        self._storage = []
        self._next_idx = 0
        self._limit = limit
        self._taskBuffers = []
        self._names = names
        for _ in tasks:
            self._taskBuffers.append(TaskRingBuffer(limit))

    def __len__(self):
        return len(self._storage)

    def append(self, item):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            for t in self._storage[self._next_idx]['tasks']:
                self._taskBuffers[t].pop()
            self._storage[self._next_idx] = item
        for t in item['tasks']:
            self._taskBuffers[t].append(self._next_idx)
        self._next_idx = (self._next_idx + 1) % self._limit

    def sample(self, batch_size, task=None):
        if task is not None:
            idxs = self._taskBuffers[task].sample(batch_size)
        else:
            idxs = [np.random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        exps = []
        for i in idxs:
            exps.append(self._storage[i])
        res = {name: np.array([exp[name] for exp in exps]) for name in self._names}
        return res
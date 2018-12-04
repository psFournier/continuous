import numpy as np

class ReplayBuffer(object):
    def __init__(self, limit, names):
        self._storage = []
        self._next_idx = 0
        self._limit = limit
        self._names = names

    def __len__(self):
        return len(self._storage)

    def append(self, item):
        if self._next_idx >= len(self._storage):
            self._storage.append(item)
        else:
            self._storage[self._next_idx] = item
        self._next_idx = (self._next_idx + 1) % self._limit

    def sample(self, batchsize):
        res = None
        if len(self._storage) >= 10000:
            idxs = [np.random.randint(0, len(self._storage) - 1) for _ in range(batchsize)]
            exps = []
            for i in idxs:
                exps.append(self._storage[i])
            res = {name: np.array([exp[name] for exp in exps]) for name in self._names}
        return res
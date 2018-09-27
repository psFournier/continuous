import numpy as np
from buffers.ringBuffer import RingBuffer


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class ReplayBuffer(object):
    def __init__(self, limit, names, args):
        self.args = args
        self.names = names
        self.contents = {}
        self.limit = limit
        self._next_idx = 0
        for name in names:
            self.contents[name] = RingBuffer()

    def append(self, buffer_item):
        for name, value in self.contents.items():
            value.append(buffer_item[name], self._next_idx)
        self._next_idx = (self._next_idx + 1) % self.limit
        return self._next_idx == 1

    def sample(self, batch_size):
        idxs = [np.random.randint(0, self.nb_entries) for _ in range(batch_size)]
        result = {}
        for name, value in self.contents.items():
            result[name] = np.array(value.get_batch(idxs))
        result['indices'] = np.array(idxs)
        return result

    @property
    def nb_entries(self):
        return len(self.contents['state0'])

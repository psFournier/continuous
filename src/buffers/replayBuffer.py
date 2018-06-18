import numpy as np
from buffers.ringBuffer import RingBuffer


def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)


class ReplayBuffer(object):
    def __init__(self, limit, names):
        self.contents = {}
        self.limit = limit
        self._next_idx = 0
        self.beta = 0
        for name in names:
            self.contents[name] = RingBuffer()

    def append(self, buffer_item):
        for name, value in self.contents.items():
            value.append(buffer_item[name], self._next_idx)
        self._next_idx = (self._next_idx + 1) % self.limit

    def sample(self, batch_size):
        batch_idxs = [np.random.randint(0, self.nb_entries) for _ in range(batch_size)]
        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value.get_batch(batch_idxs))
        result['indices'] = array_min2d(batch_idxs)
        result['weights'] = array_min2d([1] * batch_size)
        return result

    @property
    def nb_entries(self):
        return len(self.contents['state0'])

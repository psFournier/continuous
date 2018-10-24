import numpy as np

class RingBuffer(object):
    def __init__(self):
        self.data = []

    def __len__(self):
        return len(self.data)

    def get_batch(self, idxs):
        return [self.data[idx] for idx in idxs]

    def append(self, v, next_idx):

        if next_idx >= len(self.data):
            self.data.append(v)
        else:
            self.data[next_idx] = v


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
        return len(list(self.contents.values())[0])

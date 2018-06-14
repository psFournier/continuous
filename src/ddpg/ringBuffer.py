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
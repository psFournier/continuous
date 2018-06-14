import numpy as np
from ddpg.ringBuffer import RingBuffer
from ddpg.segment_tree import SumSegmentTree, MinSegmentTree
from ddpg.replayBuffer import ReplayBuffer

def array_min2d(x):
    x = np.array(x)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, limit, names, alpha, beta):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        alpha: float
            how much prioritization is used
            (0 - no prioritization, 1 - full prioritization)

        See Also
        --------
        ReplayBuffer.__init__
        """
        super(PrioritizedReplayBuffer, self).__init__(limit=limit, names=names)
        assert alpha > 0
        self.alpha = alpha
        self.beta = beta
        self.epsilon = 1e-6

        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def append(self, buffer_item):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().append(buffer_item)
        self._it_sum[idx] = self._max_priority ** self.alpha
        self._it_min[idx] = self._max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = np.random.random() * self._it_sum.sum(0, self.nb_entries - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """
        assert self.beta > 0

        idxes = self._sample_proportional(batch_size)

        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value.get_batch(idxes))

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.nb_entries) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.nb_entries) ** (-self.beta)
            weights.append(weight / max_weight)

        weights = np.array(weights)
        result['indices'] = array_min2d(idxes)
        result['weights'] = array_min2d(weights)
        return result


    def update_priorities(self, idxes, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        Parameters
        ----------
        idxes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            priority = priority + self.epsilon
            assert 0 <= idx < self.nb_entries
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha
            self._max_priority = max(self._max_priority, priority)
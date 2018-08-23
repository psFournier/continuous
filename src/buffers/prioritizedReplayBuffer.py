import numpy as np
from buffers.segment_tree import SumSegmentTree, MinSegmentTree
from buffers.replayBuffer import ReplayBuffer
from utils.linearSchedule import LinearSchedule

def array_min2d(x):
    x = np.array(x, copy=False)
    if x.ndim >= 2:
        return x
    return x.reshape(-1, 1)

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, limit, names, args):
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
        self.alpha = float(args['--alpha'])
        assert self.alpha > 0

        # self.beta_schedule = LinearSchedule(int(args['--max_steps']),
        #                                initial_p=float(args['--beta0']),
        #                                final_p=1.0)
        self.epsilon = 1e-6
        # self.epsilon_a = 0.001
        # self.epsilon_d = 1.
        # self.epsilon_a = None
        # self.epsilon_d = None

        it_capacity = 1
        while it_capacity < limit:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        # self._it_min = MinSegmentTree(it_capacity)
        self.max_priority = 1.

    def append(self, buffer_item):
        """See ReplayBuffer.store_effect"""
        idx = self._next_idx
        super().append(buffer_item)
        self._it_sum[idx] = self.max_priority ** self.alpha
        # self._it_min[idx] = self.max_priority ** self.alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            sum = self._it_sum.sum(0, self.nb_entries - 1)
            mass = np.random.random() * sum
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, step=0):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.
        beta: float
            To what degree to use importance weights
            (0 - no corrections, 1 - full correction)
        """

        idxes = self._sample_proportional(batch_size)

        # beta = self.beta_schedule.value(step)
        # assert beta > 0
        # weights = []
        # p_min = self._it_min.min() / self._it_sum.sum()
        # max_weight = (p_min * self.nb_entries) ** (-beta)
        #
        # for idx in idxes:
        #     p_sample = self._it_sum[idx] / self._it_sum.sum()
        #     weight = (p_sample * self.nb_entries) ** (-beta)
        #     weights.append(weight / max_weight)
        #
        # weights = np.array(weights)

        result = {}
        for name, value in self.contents.items():
            result[name] = array_min2d(value.get_batch(idxes))
        result['indices'] = array_min2d(idxes)
        # result['weights'] = array_min2d(weights)
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
        idxes = list(idxes.squeeze())
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            priority = priority + self.epsilon
            assert 0 <= idx < self.nb_entries
            self._it_sum[idx] = priority ** self.alpha
            # self._it_min[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)

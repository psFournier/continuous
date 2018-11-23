from collections import deque
from gym.spaces import Box
import itertools
import numpy as np
from scipy.signal import lfilter


class CompetenceQueue():
    def __init__(self, window = 100, maxlen=250):
        self.window = window
        self.C = deque(maxlen=maxlen)
        self.C_avg = deque(maxlen=10)
        self.C_avg.append(0)
        self.CP = 0
        self.init_stat()

    def init_stat(self):
        self.envstep = 0
        self.trainstep = 0
        self.trainstepT = 0
        self.attempt = 0
        self.tutorsample = 0
        self.termstate = 0

    def process_ep(self, episode):
        self.C.append(episode[-1]['t'])
        self.envstep += len(episode)
        self.attempt += 1

    def process_samples(self, samples):
        self.trainstep += 1
        self.termstate += np.mean(samples['t'])
        self.tutorsample += np.mean(samples['o'])

    def process_samplesT(self, samples):
        self.trainstepT += 1

    def update(self):
        size = len(self.C)
        if size > 2:
            window = min(size, self.window)
            self.C_avg.append(np.mean(list(self.C)[-window:]))
            self.CP = self.C_avg[-1] - self.C_avg[0]

    def get_stats(self):
        dict = {'envstep': float("{0:.3f}".format(self.envstep)),
                'trainstep': float("{0:.3f}".format(self.trainstep)),
                'termstate': float("{0:.3f}".format(self.termstate)),
                'attempt': float("{0:.3f}".format(self.attempt)),
                'tutorsample': float("{0:.3f}".format(self.tutorsample))
                }
        # self.init_stat()
        return dict

    def get_short_stats(self):
        dict = {'C': float("{0:.3f}".format(self.C_avg[-1])),
                'CP': float("{0:.3f}".format(self.CP))}
        return dict
import numpy as np
from .playroomGM import PlayroomGM

class PlayroomGM4(PlayroomGM):
    def __init__(self, env, args):
        super(PlayroomGM4, self).__init__(env, args)

    def augment_demo(self, demo):

        goal = demo[-1]['s0']
        mask = demo[-1]['s0'] != demo[-2]['s0']
        augmented_demo = []

        for i, expe in enumerate(reversed(demo)):
            altexp = expe.copy()
            altexp['g'] = goal
            altexp['m'] = mask
            altexp = self.eval_exp(altexp)
            augmented_demo.append(altexp.copy())

        return augmented_demo
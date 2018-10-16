import numpy as np
from .playroomGM import PlayroomGM

class PlayroomGM4(PlayroomGM):
    def __init__(self, env, args):
        super(PlayroomGM4, self).__init__(env, args)

    def augment_demo(self, demo):

        goal = demo[-1]['s0']
        mask = demo[-1]['s0'] != demo[-2]['s0']
        augmented_demo = []

        for i, exp in enumerate(reversed(demo)):
            exp['g'] = goal
            exp['m'] = mask
            exp = self.eval_exp(exp)
            augmented_demo.append(exp.copy())

        return augmented_demo
import numpy as np
from .playroomGM import PlayroomGM

class PlayroomGM3(PlayroomGM):
    def __init__(self, env, args):
        super(PlayroomGM3, self).__init__(env, args)

    def augment_demo(self, demo):
        augmented_demo = []
        for exp in demo:
            exp['g'] = None
            exp['m'] = None
            exp['r'] = None
            exp['t'] = None
            augmented_demo.append(exp.copy())
        return augmented_demo

    def augment_exp(self, samples):
        exps = [{'s0': samples['s0'][i], 'a': samples['a'][i], 's1': samples['s1'][i]} for i in range(samples['s0'].shape[0])]
        for i, exp in enumerate(exps):
            exp['m'] = self.mask
            exp['g'] = self.goal
            exps[i] = self.eval_exp(exp)
        res = {name: np.array([exp[name] for exp in exps]) for name in ['s0', 'a', 's1', 'r', 't', 'g', 'm']}
        return res
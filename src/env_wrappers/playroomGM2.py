import numpy as np
from .playroomGM import PlayroomGM

class PlayroomGM2(PlayroomGM):
    def __init__(self, env, args):
        super(PlayroomGM2, self).__init__(env, args)

    def augment_demo(self, demo):

        goals = []
        masks = []
        augmented_demo = []

        for i, expe in enumerate(reversed(demo)):

            # For this way of augmenting episodes, the agent actively searches states that
            # are new in some sense, with no importance granted to the difficulty of reaching
            # such states
            for j, (g, m) in enumerate(zip(goals, masks)):
                altexp = expe.copy()
                altexp['g'] = g
                altexp['m'] = m
                altexp = self.eval_exp(altexp)
                # if self.args['--wimit'] != '0':
                #     mcrs[j] = mcrs[j] * self.gamma + expe['r']
                #     expe['mcr'] = np.expand_dims(mcrs[j], axis=1)
                augmented_demo.append(altexp.copy())

            s1m = expe['s1'][np.where(self.mask)]
            s0m = expe['s0'][np.where(self.mask)]
            if (s1m != s0m).any():
                altexp = expe.copy()
                altexp['g'] = expe['s1']
                altexp['m'] = self.mask
                altexp = self.eval_exp(altexp)
                # if self.args['--wimit'] != '0':
                #     mcr = (1 - self.gamma ** (i + 1)) / (1 - self.gamma)
                #     mcrs.append(mcr)
                augmented_demo.append(altexp)
                goals.append(expe['s1'])
                masks.append(self.mask)

        return augmented_demo
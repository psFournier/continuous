import numpy as np
from .playroomGM import PlayroomGM

class PlayroomGM1(PlayroomGM):
    def __init__(self, env, args):
        super(PlayroomGM1, self).__init__(env, args)

    def augment_samples(self, samples):
        batchsize = samples['s0'].shape[0]
        exps = [{key: samples[key][i] for key in samples.keys()} for i in range(batchsize)]
        for i, exp in enumerate(exps):
            task = self.sample_task()
            exp['m'] = self.task2mask(task)
            exp['g'] = self.sample_goal(task)
            exps[i] = self.eval_exp(exp)
        res = {name: np.array([exp[name] for exp in exps]) for name in ['s0', 'a', 's1', 'r', 't', 'g', 'm']}
        return res

    # def augment_episode(self, episode):
    #
    #     goals = []
    #     masks = []
    #     augmented_ep = []
    #     trajectory_mask = []
    #     if 'm' in trajectory[-1].keys():
    #         trajectory_mask.append(trajectory[-1]['m'])
    #     # if self.args['--wimit'] != '0':
    #     #     mcrs = []
    #
    #     for i, expe in enumerate(reversed(trajectory)):
    #
    #         # For this way of augmenting episodes, the agent actively searches states that
    #         # are new in some sense, with no importance granted to the difficulty of reaching
    #         # such states
    #         for j, (g, m) in enumerate(zip(goals, masks)):
    #             altexp = expe.copy()
    #             altexp['g'] = g
    #             altexp['m'] = m
    #             altexp = self.eval_exp(altexp)
    #             # if self.args['--wimit'] != '0':
    #             #     mcrs[j] = mcrs[j] * self.gamma + expe['r']
    #             #     expe['mcr'] = np.expand_dims(mcrs[j], axis=1)
    #             augmented_ep.append(altexp.copy())
    #
    #         for obj_idx, goal in enumerate(self.goals):
    #             # self.goals contains the objects, not their goal value
    #             m = self.obj2mask(obj_idx)
    #             # I compare the object mask to the one pursued and to those already "imagined"
    #             if all([(m != m2).any() for m2 in masks + trajectory_mask]):
    #                 # We can't test all alternative goals, and chosing random ones would bring
    #                 # little improvement. So we select goals as states where an object as changed
    #                 # its state.
    #                 s1m = expe['s1'][np.where(m)]
    #                 s0m = expe['s0'][np.where(m)]
    #                 if (s1m != s0m).any():
    #                     altexp = expe.copy()
    #                     altexp['g'] = expe['s1']
    #                     altexp['m'] = m
    #                     altexp = self.eval_exp(altexp)
    #                     # if self.args['--wimit'] != '0':
    #                     #     mcr = (1 - self.gamma ** (i + 1)) / (1 - self.gamma)
    #                     #     mcrs.append(mcr)
    #                     augmented_ep.append(altexp)
    #                     goals.append(expe['s1'])
    #                     masks.append(m)
    #
    #     return augmented_ep
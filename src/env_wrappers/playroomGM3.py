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

    def augment_tutor_samples(self, samples):
        batchsize = samples['s0'].shape[0]
        exps = [{key: samples[key][i] for key in samples.keys()} for i in range(batchsize)]
        for i, exp in enumerate(exps):
            task = self.sample_task()
            exp['task'] = task
            exp['m'] = self.task2mask(task)
            exp['g'] = self.sample_goal(task)
            exps[i] = self.eval_exp(exp)
        res = {name: np.array([exp[name] for exp in exps]) for name in ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'task']}
        return res

    def augment_samples(self, samples):
        batchsize = samples['s0'].shape[0]
        exps = [{key: samples[key][i] for key in samples.keys()} for i in range(batchsize)]
        for i, exp in enumerate(exps):
            task = self.sample_task()
            exp['m'] = self.task2mask(task)
            exp['g'] = self.sample_goal(task)
            exps[i] = self.eval_exp(exp)
        res = {name: np.array([exp[name] for exp in exps]) for name in
                   ['s0', 'a', 's1', 'r', 't', 'g', 'm', 'task']}
        return res
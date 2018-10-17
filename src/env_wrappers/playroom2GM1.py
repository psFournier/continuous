import numpy as np
from .playroom2GM import Playroom2GM

class Playroom2GM1(Playroom2GM):
    def __init__(self, env, args):
        super(Playroom2GM1, self).__init__(env, args)

    def augment_episode(self, episode):

        goals = []
        masks = []
        tasks = []
        augmented_ep = []

        for i, expe in enumerate(reversed(episode)):

            for j, (g, m, t) in enumerate(zip(goals, masks, tasks)):
                altexp = expe.copy()
                altexp['g'] = g
                altexp['m'] = m
                altexp['task'] = t
                altexp = self.eval_exp(altexp)
                augmented_ep.append(altexp.copy())

            if np.random.rand() < 0.1:
                task = np.random.randint(len(self.goals))
                m = self.task2mask(task)
                altexp = expe.copy()
                altexp['g'] = expe['s1']
                altexp['m'] = m
                altexp['task'] = task
                altexp = self.eval_exp(altexp)
                augmented_ep.append(altexp)
                goals.append(expe['s1'])
                masks.append(m)
                tasks.append(task)

        return augmented_ep
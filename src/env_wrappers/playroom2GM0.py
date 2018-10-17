import numpy as np
from .playroom2GM import Playroom2GM

class Playroom2GM0(Playroom2GM):
    def __init__(self, env, args):
        super(Playroom2GM0, self).__init__(env, args)

    def augment_episode(self, episode):

        goals = []
        masks = []
        tasks = []
        augmented_ep = []

        for i, expe in enumerate(reversed(episode)):

            # For this way of augmenting episodes, the agent actively searches states that
            # are new in some sense, with no importance granted to the difficulty of reaching
            # such states
            for j, (g, m, t) in enumerate(zip(goals, masks, tasks)):
                altexp = expe.copy()
                altexp['g'] = g
                altexp['m'] = m
                altexp['task'] = t
                altexp = self.eval_exp(altexp)
                augmented_ep.append(altexp.copy())

            for task, _ in enumerate(self.goals[1:]):
                # self.goals contains the objects, not their goal value
                m = self.task2mask(task)
                # I compare the object mask to the one pursued and to those already "imagined"
                if all([task != t for t in tasks + [self.task]]):
                    # We can't test all alternative goals, and chosing random ones would bring
                    # little improvement. So we select goals as states where an object as changed
                    # its state.
                    s1m = expe['s1'][np.where(m)]
                    s0m = expe['s0'][np.where(m)]
                    if (s1m != s0m).any():
                        altexp = expe.copy()
                        altexp['g'] = expe['s1']
                        altexp['m'] = m
                        altexp['task'] = task
                        altexp = self.eval_exp(altexp)
                        augmented_ep.append(altexp)
                        goals.append(expe['s1'])
                        masks.append(m)
                        tasks.append(task)

            if np.random.rand() < 0.02:
                m = self.task2mask(0)
                altexp = expe.copy()
                altexp['g'] = expe['s1']
                altexp['m'] = m
                altexp['task'] = 0
                altexp = self.eval_exp(altexp)
                augmented_ep.append(altexp)
                goals.append(expe['s1'])
                masks.append(m)
                tasks.append(0)

        return augmented_ep
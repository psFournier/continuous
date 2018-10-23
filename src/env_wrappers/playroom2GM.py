import numpy as np
from .base import CPBased

class Playroom2GM(CPBased):
    def __init__(self, env, args):
        super(Playroom2GM, self).__init__(env, args, [obj.name for obj in env.objects])
        self.mask = None
        self.init()
        self.obj_feat = [[4], [7], [10]]
        self.state_low = self.env.low
        self.state_high = self.env.high
        self.init_state = np.array(self.env.initstate)
        self.r_done = 100
        self.r_notdone = 0
        self.terminal = True
        self.minQ = self.r_notdone / (1 - self.gamma)
        self.maxQ = self.r_done if self.terminal else self.r_done / (1 - self.gamma)

    def step(self, exp):
        self.steps[self.task] += 1
        exp['g'] = self.goal
        exp['m'] = self.mask
        exp['task'] = self.task
        exp['s1'] = self.env.step(exp['a'])[0]
        exp = self.eval_exp(exp)
        return exp

    def eval_exp(self, exp):
        indices = np.where(exp['m'])
        goal = exp['g'][indices]
        s1_proj = exp['s1'][indices]
        if (s1_proj == goal).all():
            exp['t'] = self.terminal
            exp['r'] = self.r_done
        else:
            exp['t'] = False
            exp['r'] = self.r_notdone
        return exp

    def end_episode(self, episode):

        T = episode[-1]['t']
        self.queues[self.task].append(T)

        augmented_ep = self.process_trajectory(trajectory=episode,
                                               goals=[self.goal],
                                               masks=[self.mask],
                                               tasks=[self.task],
                                               mcrs=[np.array([0])])

        return augmented_ep

    def process_trajectory(self, trajectory, goals, masks, tasks, mcrs):

        augmented_trajectory = []

        for expe in reversed(trajectory):

            # For this way of augmenting episodes, the agent actively searches states that
            # are new in some sense, with no importance granted to the difficulty of reaching
            # such states
            for j, (g, m, t) in enumerate(zip(goals, masks, tasks)):
                altexp = expe.copy()
                altexp['g'] = g
                altexp['m'] = m
                altexp['task'] = t
                altexp = self.eval_exp(altexp)
                mcrs[j] = mcrs[j] * self.gamma + altexp['r']
                altexp['mcr'] = np.expand_dims(mcrs[j], axis=1)

                augmented_trajectory.append(altexp.copy())

            for task, _ in enumerate(self.goals):
                # self.goals contains the objects, not their goal value
                m = self.task2mask(task)
                # I compare the object mask to the one pursued and to those already "imagined"
                if all([task != t for t in tasks]):
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
                        altexp['mcr'] = np.expand_dims(altexp['r'], axis=1)
                        augmented_trajectory.append(altexp)
                        goals.append(expe['s1'])
                        masks.append(m)
                        tasks.append(task)
                        mcrs.append(altexp['r'])

        return augmented_trajectory

    def reset(self):
        state = self.env.reset()
        self.update_interests()

        self.task = np.random.choice(len(self.goals), p=self.interests)
        features = self.obj_feat[self.task]
        while all([state[f] == self.state_high[f] for f in features]):
            self.task = np.random.choice(len(self.goals), p=self.interests)
            features = self.obj_feat[self.task]
        self.mask = self.task2mask(self.task)

        # self.task = 0
        # features = self.obj_feat[self.task]
        # self.mask = self.task2mask(self.task)

        self.goal = state.copy()
        while not (self.goal != state).any():
            for f in features:
                self.goal[f] = np.random.randint(state[f], self.state_high[f] + 1)

        return state

    def task2mask(self, idx):
        res = np.zeros(shape=self.state_dim)
        res[self.obj_feat[idx]] = 1
        return res

    def mask2task(self, mask):
        return list(np.where(mask)[0])

    def augment_demo(self, demo):
        return demo

    def augment_samples(self, samples):
        return samples

    def augment_tutor_samples(self, samples):
        return samples

    @property
    def state_dim(self):
        return 11,

    @property
    def goal_dim(self):
        return 11,

    @property
    def action_dim(self):
        return 6
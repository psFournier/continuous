from gym import Wrapper
import numpy as np
from samplers.listSampler import ListSampler

class TaxiGoalTutor(Wrapper):
    def __init__(self, env, args):
        super(TaxiGoalTutor, self).__init__(env)

        self.goal = 0
        self.goal_space = [0,1,2]
        self.goal_idx = 4
        self.sampler = ListSampler(space=[0,1],
                             theta=float(args['theta']))

        self.episode_exp = []
        self.buffer = None
        self.exploration_steps = 0

    def step(self, action):
        obs, _, _, _ = self.env.step(action)
        state = np.array(self.decode(obs))
        return state

    def eval_exp(self, state0, action, state1, goal):
        term = False
        r = 0
        if goal == 0 and state0[2] < 4 and state1[2] == 4:
            r = 1
            term = True
        elif goal == 1 and state0[2] != state0[3] and state1[2] == state1[3]:
            r = 1
            term = True
        return r, term

    def reset(self, goal=None):

        if self.episode_exp and self.goal != 2:
            R = np.sum([exp['reward'] for exp in self.episode_exp])
            self.sampler.append((self.goal, int(self.episode_exp[-1]['terminal']), R))

        if goal is None:
            if self.sampler.max_CP < 0.1 and np.random.rand() < 0.5:
                self.goal = 2
            else:
                self.goal = self.sampler.sample()
        else:
            self.goal = goal

        obs = self.env.reset()
        state = np.array(self.decode(obs))
        self.episode_exp.clear()

        return state

    def decode(self, state):
        return list(self.env.decode(state))

    def hindsight(self):
        return []

    @property
    def state_dim(self):
        return 5,

    @property
    def action_dim(self):
        return [self.env.action_space.n]

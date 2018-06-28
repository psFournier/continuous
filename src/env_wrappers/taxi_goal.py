from gym import Wrapper
import numpy as np
from samplers.listSampler import ListSampler

class TaxiGoal(Wrapper):
    def __init__(self, env, args):
        super(TaxiGoal, self).__init__(env)

        self.goal = 0
        self.goal_space = [0,1]
        self.sampler = ListSampler(space=self.goal_space,
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

    def reset(self):

        if self.episode_exp:
            R = np.sum([exp['reward'] - 1 for exp in self.episode_exp])
            self.sampler.append((self.goal, int(self.episode_exp[-1]['terminal']), R))

        self.goal = self.sampler.sample()

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
        return (5,5,5,4,2)

    @property
    def action_dim(self):
        return [self.env.action_space.n]

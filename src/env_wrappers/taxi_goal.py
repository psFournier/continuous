from gym import Wrapper
import numpy as np
from samplers.listSampler import ListSampler

class TaxiGoal(Wrapper):
    def __init__(self, env, args):
        super(TaxiGoal, self).__init__(env)

        self.goal = 0
        self.goal_space = [0,1]
        self.goal_idx = 1
        self.sampler = ListSampler(space=self.goal_space,
                             theta=float(args['theta_eps']))

        self.episode_exp = []
        self.buffer = None
        self.exploration_steps = 0

    def add_goal(self, state):
        return np.array([state, self.goal])


    def _step(self, action):

        obs, env_reward, env_terminal, info = self.env.step(action)
        state = self.add_goal(obs)

        reward, terminal, gamma = self.eval_exp(self.prev_state, action, state, env_reward,
                                        env_terminal)

        experience = {'state0': self.prev_state,
                      'action': action,
                      'state1': state,
                      'reward': reward,
                      'terminal': terminal,
                      'gamma': gamma}

        self.prev_state = state
        self.episode_exp.append(experience)

        return experience

    def eval_exp(self, state0, action, state1, reward, terminal):
        term = False
        r = -1
        gamma = 0.99
        row0, col0, passidx0, destidx0 = self.decode(state0[0])
        row1, col1, passidx1, destidx1 = self.decode(state1[0])
        if state1[self.goal_idx] == 0 and passidx0 < 4 and passidx1 == 4:
            r = 0
            term = True
        elif state1[self.goal_idx] == 1 and passidx0 != destidx0 and passidx1 == destidx1:
            r = 0
            term = True
        if passidx0 < 4 and passidx1 == 4:
            gamma = 0
        return r, term, gamma

    def _reset(self):

        if self.episode_exp:
            R = np.sum([exp['reward'] for exp in self.episode_exp])
            self.sampler.append((self.goal, int(self.episode_exp[-1]['terminal']), R))

        # self.goal = self.sampler.sample()
        self.goal = 1

        obs = self.env.reset()
        state = self.add_goal(obs)
        self.prev_state = state
        self.episode_exp.clear()

        return state

    def decode(self, state):
        return list(self.env.decode(state))

    def hindsight(self):
        return []

    @property
    def state_dim(self):
        return [self.env.observation_space.n, len(self.goal_space)]

    @property
    def action_dim(self):
        return [self.env.action_space.n]

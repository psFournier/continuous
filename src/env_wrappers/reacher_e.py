import numpy as np
from gym import Wrapper
from gym.spaces import Box
import random as rnd
import math


class Reacher_e(Wrapper):
    def __init__(self, env):
        super(Reacher_e, self).__init__(env)

        self.goal = []
        self.goal_space = Box(np.array([-0.2, -0.2]), np.array([0.2, 0.2]))
        self.XY = [6,7]
        self.target_XY = [8,9]

        self.epsilon = 0.02
        self.epsilons = [0.02, 0.03, 0.04, 0.05]
        self.epsilon_idx = [10]

        self.rec = None

    def add_goal(self, state):
        return np.concatenate([state, self.goal, np.array([self.epsilon])])

    def _step(self,action):

        obs, env_reward, env_terminal, info = self.env.step(action)
        state = self.add_goal(obs)

        if self.rec is not None: self.rec.capture_frame()

        reward, reached = self.eval_exp(self.prev_state, action, state, env_reward,
                                         env_terminal)

        self.prev_state = state
        return state, reward, reached, info

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        reached_xy = agent_state_1[self.XY]
        target_xy = agent_state_1[self.target_XY]
        vec = reached_xy - target_xy
        d = np.linalg.norm(vec)
        term = d < self.epsilon
        r = 0
        if not term:
            r = -1
        return r, term

    def _reset(self):

        _ = self.env.reset()
        qpos = self.unwrapped.sim.data.qpos.flatten()
        qvel = self.unwrapped.sim.data.qvel.flatten()

        qpos[[2,3]] = self.goal
        self.reached = False

        self.unwrapped.set_state(qpos, qvel)
        obs = self.unwrapped._get_obs()
        state = self.add_goal(obs)
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()

        return state

    def is_reachable(self):
        return (np.linalg.norm(self.goal) < 0.2)

    def hindsight(self, episode_exp, her_xy, her_eps):

        virtual_exp = []

        for idx, buffer_item in enumerate(episode_exp):

            new_targets = []
            if her_xy == 'future':
                indices = range(idx, len(episode_exp))
                future_indices = rnd.sample(indices, np.min([4, len(indices)]))
                new_targets += [episode_exp[i]['state1'][self.XY]
                                for i in list(future_indices)]
            elif her_xy == 'final':
                new_targets += episode_exp[-1]['state1'][self.XY]

            new_eps = []
            if her_eps == 'easier':
                new_eps += [eps for eps in self.epsilons if eps > self.epsilon]
            elif her_eps == 'harder':
                new_eps += [eps for eps in self.epsilons if eps < self.epsilon]
            elif her_eps == 'all':
                new_eps += [eps for eps in self.epsilons if eps != self.epsilon]

            for t in new_targets + [self.goal]:
                for e in new_eps + [self.epsilon]:
                    if (t != self.goal).any() or e != self.epsilon:
                        res = buffer_item.copy()
                        res['state0'][self.target_XY] = t
                        res['state1'][self.target_XY] = t
                        res['state0'][self.epsilon_idx] = e
                        res['state1'][self.epsilon_idx] = e
                        res['reward'], res['terminal'] = self.eval_exp(res['state0'],
                                                                           res['action'],
                                                                           res['state1'],
                                                                           res['reward'],
                                                                           res['terminal'])
                        virtual_exp.append(res)

        return virtual_exp

    @property
    def state_dim(self):
        return (self.env.observation_space.shape[0]+self.goal_space.shape[0]+1,)

    @property
    def action_dim(self):
        return (self.env.action_space.shape[0],)

    @property
    def goal_parameterized(self):
        return True


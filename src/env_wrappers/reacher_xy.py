import numpy as np
from gym import Wrapper
from gym.spaces import Box
import random as rnd
from samplers.treeSampler import TreeSampler
from samplers.treePlotter import TreePlotter
from samplers.treeSampler2 import TreeSampler2
from samplers.treePlotter2 import TreePlotter2

#todo add training param to env_wrapper to avoir useless things during testing

class Reacher_xy(Wrapper):
    def __init__(self, env, args):
        super(Reacher_xy, self).__init__(env)

        self.goal = []
        self.goal_space = Box(np.array([-0.25, -0.25]), np.array([0.25, 0.25]), dtype='float32')
        # self.goal_space = Box(np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        self.XY = [6,7]
        self.target_XY = [8,9]
        self.sampler = TreePlotter2(space=self.goal_space,
                                   R=int(args['R']),
                                   auto=True,
                                   theta=float(args['theta_xy']))

        self.episode_exp = []
        self.buffer = None
        self.her = args['her_xy']
        self.rec = None
        self.exploration_steps = 1000

    def add_goal(self, state):
        return np.concatenate([state, self.goal])

    def step(self,action):

        obs, reward, terminal, info = self.env.step(action)
        state = self.add_goal(obs)

        if self.rec is not None: self.rec.capture_frame()

        reward, terminal = self.eval_exp(self.prev_state, action, state, reward,
                                         terminal)

        experience = {'state0': self.prev_state,
                      'action': action,
                      'state1': state,
                      'reward': reward,
                      'terminal': terminal}

        self.prev_state = state
        self.episode_exp.append(experience)

        return experience

    def eval_exp(self, _, action, agent_state_1, reward, terminal):
        reached_xy = agent_state_1[self.XY]
        target_xy = agent_state_1[self.target_XY]
        vec = reached_xy - target_xy
        d = np.linalg.norm(vec)
        term = d < 0.02
        r = -1
        if term:
            r = 0
        return r, term

    def reset(self):

        if self.episode_exp:
            R = np.sum([exp['reward'] for exp in self.episode_exp])
            self.sampler.append((self.goal, R, int(self.episode_exp[-1]['terminal'])))
            self.sampler.append((self.episode_exp[-1]['state1'][self.XY], 0, 1))
            hindsight_exp = self.hindsight()
            for exp in hindsight_exp:
                self.buffer.append(exp)

        obs = self.env.reset()
        qpos = self.unwrapped.sim.data.qpos.flatten()
        qvel = self.unwrapped.sim.data.qvel.flatten()

        self.goal = self.sampler.sample(rnd_prop=0)
        qpos[[2,3]] = self.goal

        self.unwrapped.set_state(qpos, qvel)
        obs = self.unwrapped._get_obs()
        state = self.add_goal(obs)
        self.prev_state = state
        self.episode_exp.clear()
        if self.rec is not None: self.rec.capture_frame()

        return state

    def hindsight(self):

        virtual_exp = []

        for idx, buffer_item in enumerate(self.episode_exp):

            new_targets = []
            if self.her == 'future':
                indices = range(idx, len(self.episode_exp))
                future_indices = rnd.sample(indices, np.min([4, len(indices)]))
                new_targets += [self.episode_exp[i]['state1'][self.XY]
                                for i in list(future_indices)]
            elif self.her == 'final':
                new_targets += [self.episode_exp[-1]['state1'][self.XY]]

            for t in new_targets:
                res = buffer_item.copy()
                res['state0'][self.target_XY] = t
                res['state1'][self.target_XY] = t
                res['reward'], res['terminal'] = self.eval_exp(res['state0'],
                                                                   res['action'],
                                                                   res['state1'],
                                                                   res['reward'],
                                                                   res['terminal'])
                virtual_exp.append(res)

        return virtual_exp

    def is_reachable(self):
        return (np.linalg.norm(self.goal) < 0.2)

    @property
    def state_dim(self):
        return (self.env.observation_space.shape[0]+self.goal_space.shape[0],)

    @property
    def action_dim(self):
        return (self.env.action_space.shape[0],)
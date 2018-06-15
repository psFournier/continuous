import numpy as np
from gym import Wrapper
from gym.spaces import Box
from ddpg.regionTreePlot import RegionTreePlot
from ddpg.replayBuffer import ReplayBuffer
import random as rnd


class Reacher_xy(Wrapper):
    def __init__(self, env, epsilon, nRegions, beta_xy, her, buffer_size=int(1e6)):
        super(Reacher_xy, self).__init__(env)

        self.goal = []
        self.reached = False
        self.reward = 0
        self.XY = [6,7]
        self.target_XY = [8,9]
        self.goal_space = Box(np.array([-0.6, -0.6]), np.array([0.6, 0.6]))
        self.regionTree = RegionTreePlot(space=self.goal_space,
                                     nRegions=nRegions,
                                     auto = True,
                                     beta=beta_xy)

        self.epsilon = epsilon

        self.rec = None

        self.buffer = ReplayBuffer(limit = buffer_size,
                                   names=['state0', 'action', 'state1', 'reward', 'terminal'])

        self.episode = 0
        self.episode_exp = []
        self.her_xy_strat = her

    def add_goal(self, state):
        return np.concatenate([state, self.goal])

    def _step(self,action):

        obs, env_reward, env_terminal, info = self.env.step(action)
        state = self.add_goal(obs)

        if self.rec is not None: self.rec.capture_frame()

        self.reward, self.reached = self.eval_exp(self.prev_state, action, state, env_reward,
                                         env_terminal)
        exp = {'state0': self.prev_state.copy(),
                   'action': action,
                   'state1': state.copy(),
                   'reward': self.reward,
                   'terminal': self.reached}
        self.episode_exp.append(exp)
        self.buffer.append(exp)

        self.prev_state = state
        return state, self.reward, self.reached, info

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

    def her_targets(self, idx):
        new_targets = []
        if self.her_xy_strat == 'future':
            indices = range(idx, len(self.episode_exp))
            future_indices = rnd.sample(indices, np.min([4, len(indices)]))
            new_targets += [self.episode_exp[i]['state1'][self.XY]
                            for i in list(future_indices)]
        elif self.her_xy_strat == 'final':
            new_targets += self.episode_exp[-1]['state1'][self.XY]
        else:
            pass
        return new_targets

    def hindsight(self):
        for idx, buffer_item in enumerate(self.episode_exp):
            new_targets = self.her_targets(idx)
            for t in new_targets:
                res = buffer_item.copy()
                res['state0'][self.target_XY] = t
                res['state1'][self.target_XY] = t
                res['reward'], res['terminal'] = self.eval_exp(res['state0'],
                                                                   res['action'],
                                                                   res['state1'],
                                                                   res['reward'],
                                                                   res['terminal'])
                self.buffer.append(res)

    def _reset(self):

        if self.episode > 0:
            self.regionTree.append((self.goal, int(self.reached)))
            self.regionTree.append((self.episode_exp[-1]['state1'][self.XY], 1))

        if self.her_xy_strat != 'no':
            self.hindsight()

        _ = self.env.reset()
        qpos = self.unwrapped.sim.data.qpos.flatten()
        qvel = self.unwrapped.sim.data.qvel.flatten()

        region = self.regionTree.sample(rnd_prop = max(0.1, 1 - self.episode//200))
        self.goal = region.sample().flatten()
        qpos[[2,3]] = self.goal
        self.reached = False

        self.unwrapped.set_state(qpos, qvel)
        obs = self.unwrapped._get_obs()
        state = self.add_goal(obs)
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()

        self.episode += 1
        self.episode_exp = []

        return state

    def is_reachable(self):
        return (np.linalg.norm(self.goal) < 0.2)

    def stats(self):
        return self.regionTree.stats()

    @property
    def state_dim(self):
        return (self.env.observation_space.shape[0]+self.goal_space.shape[0],)

    @property
    def action_dim(self):
        return (self.env.action_space.shape[0],)

    @property
    def goal_parameterized(self):
        return True
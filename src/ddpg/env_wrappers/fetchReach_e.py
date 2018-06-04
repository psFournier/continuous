import numpy as np
from gym import Wrapper
from gym.spaces import Box
from ddpg.replayBuffer import ReplayBuffer
from ddpg.competenceQueue import CompetenceQueue
import random as rnd
import math


class FetchReach_e(Wrapper):
    def __init__(self, env, epsilon = 0.02, nRegions = 4, beta = 1, her = 'no_no', buffer_size = 1e6):
        super(FetchReach_e, self).__init__(env)

        self.goal = []
        self.goal_space = Box(np.array([-np.inf, -np.inf, -np.inf]), np.array([np.inf, np.inf, np.inf])) # TODO
        self.reached = False
        self.reward = 0
        self.XY = [0,1,2]
        self.target_XY = [10, 11, 12]

        self.beta = beta
        self.epsilon = epsilon
        self.epsilon_idx = [13]
        self.epsilons = [0.02, 0.03, 0.04, 0.05]
        self.epsilon_queues = [CompetenceQueue() for _ in self.epsilons]
        self.epsilon_freq = [0] * len(self.epsilons)

        self.rec = None

        self.buffer = ReplayBuffer(limit = buffer_size,
                          content_shape = {'state0': self.state_dim,
                           'action': self.action_dim,
                           'state1': self.state_dim,
                           'reward': (1,),
                           'terminal': (1,)})

        self.episode = 0
        self.episode_exp = []
        self.her_xy_strat = her.split('_')[0]
        self.her_e_strat = her.split('_')[1]

    def add_goal(self, state):
        return np.concatenate([state, self.goal, np.array([self.epsilon])])

    def _step(self,action):

        dictObs, env_reward, env_terminal, info = self.env.step(action)
        state = self.add_goal(dictObs['observation'])

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

    def her_xy(self, idx):
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

    def her_e(self):
        new_eps = []
        if self.her_e_strat == 'easier':
            new_eps += [eps for eps in self.epsilons if eps > self.epsilon]
        elif self.her_e_strat == 'harder':
            new_eps += [eps for eps in self.epsilons if eps < self.epsilon]
        elif self.her_e_strat == 'all':
            new_eps += [eps for eps in self.epsilons if eps != self.epsilon]
        else:
            pass
        return new_eps

    def hindsight(self):
        for idx, buffer_item in enumerate(self.episode_exp):
            new_targets = self.her_xy(idx)
            new_eps = self.her_e()
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
                        self.buffer.append(res)


    def sample_epsilon(self):
        CPs = [math.pow(queue.CP, self.beta) for queue in self.epsilon_queues]
        sum = np.sum(CPs)
        mass = np.random.random() * sum
        idx = 0
        s = CPs[0]
        while mass > s:
            idx += 1
            s += CPs[idx]
        return self.epsilons[idx]

    def _reset(self):

        if self.episode > 0:
            self.epsilon_queues[self.epsilons.index(self.epsilon)].append((self.epsilon, int(self.reached)))

        if self.her_xy_strat != 'no' or self.her_e_strat != 'no':
            self.hindsight()

        dictObs = self.env.reset()
        obs = dictObs['observation']
        self.goal = dictObs['desired_goal']
        print('goal: ', self.goal)
        # qpos = self.unwrapped.sim.data.qpos.flatten()
        # qvel = self.unwrapped.sim.data.qvel.flatten()
        #
        # while True:
        #     self.goal = np.random.uniform(low=self.goal_space.low, high=self.goal_space.high)
        #     if self.is_reachable(): break
        # qpos[[2,3]] = self.goal
        # self.unwrapped.set_state(qpos, qvel)
        # obs = self.unwrapped._get_obs()
        self.reached = False

        self.epsilon = self.sample_epsilon()
        self.epsilon_freq[self.epsilons.index(self.epsilon)] += 1


        state = self.add_goal(obs)
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()

        self.episode += 1
        self.episode_exp = []

        return state

    # def is_reachable(self):
    #     return (np.linalg.norm(self.goal) < 0.2)

    def stats(self):
        stats = {}
        stats['list_CP'] = self.list_CP
        stats['list_comp'] = self.list_comp
        stats['list_freq'] = self.epsilon_freq
        return stats

    @property
    def state_dim(self):
        return (self.env.observation_space.spaces['observation'].shape[0]+self.goal_space.shape[0]+1,)

    @property
    def action_dim(self):
        return (self.env.action_space.shape[0],)

    @property
    def goal_parameterized(self):
        return True

    @property
    def list_CP(self):
        return [float("{0:.3f}".format(self.epsilon_queues[idx].CP)) for idx in range(len(self.epsilons))]

    @property
    def list_comp(self):
        return [float("{0:.3f}".format(self.epsilon_queues[idx].competence)) for idx in range(len(self.epsilons))]

    @property
    def list_freq(self):
        return self.epsilon_freq
import numpy as np
from .base import Base


class Reacher1(Base):
    def __init__(self, env, args):
        super(Reacher1, self).__init__(env, args)
        self.init()
        self.minQ = -np.inf
        self.maxQ = np.inf

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < 0.04:
            exp['r'] = 0
            exp['t'] = True
        else:
            exp['r'] = -1
            exp['t'] = False
        return exp

    def end_episode(self, trajectory):
        R = 0
        for exp in reversed(trajectory):
            R = R * self.gamma + exp['r'] + 1 + exp['t'] * 99
        self.queue.append(R)

    def reset(self):
        _ = self.env.reset()
        qpos = self.unwrapped.sim.data.qpos.flatten()
        qvel = self.unwrapped.sim.data.qvel.flatten()
        qpos[2] = -0.025
        qpos[3] = 0.15
        self.unwrapped.set_state(qpos, qvel)
        state = self.unwrapped._get_obs()
        return state

    @property
    def state_dim(self):
        return 8,

    @property
    def action_dim(self):
        return 2,

class Reacher1S(Reacher1):
    def __init__(self, env, args):
        super(Reacher1S, self).__init__(env, args)

    def eval_exp(self, exp):
        d = np.linalg.norm(exp['s1'][[6, 7]])
        if d < 0.02:
            exp['r'] = 0
            exp['t'] = True
        else:
            exp['r'] = -1
            exp['t'] = False
        exp['r'] += (-self.gamma * np.linalg.norm(exp['s1'][[6, 7]]) + np.linalg.norm(exp['s0'][[6, 7]]))
        return exp


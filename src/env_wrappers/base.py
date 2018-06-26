from gym import Wrapper
from buffers.replayBuffer import ReplayBuffer
from buffers.prioritizedReplayBuffer import PrioritizedReplayBuffer

# Wrappers override step, reset functions, as well as the defintion of action, observation and goal spaces.

class Base(Wrapper):
    def __init__(self, env, args):
        super(Base, self).__init__(env)
        self.rec = None
        self.buffer = ReplayBuffer(limit = int(1e6),
                                   names=['state0', 'action', 'state1', 'reward', 'terminal'])
        self.episode_exp = []
        self.sampler = None
        self.exploration_steps = 0

    def step(self,action):

        state, reward, terminal, info = self.env.step(action)

        if self.rec is not None: self.rec.capture_frame()

        experience = {'state0': self.prev_state,
                   'action': action,
                   'state1': state,
                   'reward': reward,
                   'terminal': terminal}

        self.prev_state = state

        return experience

    def reset(self):

        state = self.env.reset()
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()

        return state

    def stats(self):
        stats = {}
        return stats

    @property
    def state_dim(self):
        return (self.env.observation_space.shape[0],)

    @property
    def action_dim(self):
        return (self.env.action_space.shape[0],)
from gym import Wrapper
from buffers.replayBuffer import ReplayBuffer
from buffers.prioritizedReplayBuffer import PrioritizedReplayBuffer

# Wrappers override step, reset functions, as well as the defintion of action, observation and goal spaces.

class Base(Wrapper):
    def __init__(self, env, buffer_size=int(1e6)):
        super(Base, self).__init__(env)
        self.rec = None
        self.buffer = PrioritizedReplayBuffer(limit = buffer_size,
                                              names=['state0', 'action', 'state1', 'reward', 'terminal'],
                                              alpha=1,
                                              beta=1)
        self.episode = 0

    def _step(self,action):

        state, self.reward, self.reached, info = self.env.step(action)

        if self.rec is not None: self.rec.capture_frame()

        exp = {'state0': self.prev_state.copy(),
                   'action': action,
                   'state1': state.copy(),
                   'reward': self.reward,
                   'terminal': self.reached}
        self.buffer.append(exp)

        self.prev_state = state
        return state, self.reward, self.reached, info

    def _reset(self):

        state = self.env.reset()
        self.prev_state = state
        if self.rec is not None: self.rec.capture_frame()
        self.episode += 1

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

    @property
    def goal_parameterized(self):
        return False
from gym import Wrapper

class Taxi_gamma(Wrapper):
    def __init__(self, env, gamma=0.99):
        super(Taxi_gamma, self).__init__(env)
        self.gamma = gamma
        self.goal_space = None
        self.epsilon_space = None


    def _step(self, action):

        state, env_reward, env_terminal, info = self.env.step(action)

        x, y, passidx, _ = list(self.env.decode(self.prev_state))
        if env_terminal:
            self.gamma = 0
        elif (passidx < 4 and (x, y) == self.env.locs[passidx] and action == 4):
            self.gamma = 0
        else:
            self.gamma = 0.99

        self.prev_state = state
        return state, env_reward, env_terminal, info

    def _reset(self):

        state = self.env.reset()
        self.prev_state = state

        return state
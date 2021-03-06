import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import ActorDDPG, CriticDDPG, ActorCriticDDPG
from agents.agent import Agent
from buffers import ReplayBuffer

class DDPG(Agent):

    def __init__(self, args, env, env_test, logger):
        super(DDPG, self).__init__(args, env, env_test, logger)
        self.args = args
        self.init(args, env)
        for metric in self.critic.model.metrics_names:
            self.metrics[self.critic.model.name + '_' + metric] = 0

    def init(self, args ,env):
        names = ['state0', 'action', 'state1', 'reward', 'terminal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy())
        self.actorCritic = ActorCriticDDPG(args, env)
        # self.critic = CriticDDPG(args, env)
        # self.actor = ActorDDPG(args, env)

    def train(self):

        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t = [exp[name] for name in self.buffer.names]
            a1 = self.actor.target_model.predict_on_batch(s1)
            a1 = np.clip(a1, self.env.action_space.low, self.env.action_space.high)
            q = self.critic.Tmodel.predict_on_batch([s1, a1])
            targets = r + (1 - t) * self.critic.gamma * np.squeeze(q)
            targets = np.clip(targets, self.env.minR / (1 - self.critic.gamma), self.env.maxR)
            inputs = [s0, a0]
            loss = self.critic.model.train_on_batch(inputs, targets)
            for i, metric in enumerate(self.critic.model.metrics_names):
                self.metrics[metric] += loss[i]

            # a2 = self.actor.model.predict_on_batch(s0)
            # grads = self.critic.gradsModel.predict_on_batch([s0, a2])
            # low = self.env.action_space.low
            # high = self.env.action_space.high
            # for d in range(grads[0].shape[0]):
            #     width = high[d] - low[d]
            #     for k in range(self.batch_size):
            #         if grads[k][d] >= 0:
            #             grads[k][d] *= (high[d] - a2[k][d]) / width
            #         else:
            #             grads[k][d] *= (a2[k][d] - low[d]) / width
            # self.actor.train(s0, grads)

            self.actor.target_train()
            self.critic.target_train()

    def reset(self):

        if self.trajectory:
            T = int(self.trajectory[-1]['terminal'])
            R = np.sum([self.env.unshape(exp['reward'], exp['terminal']) for exp in self.trajectory])
            S = len(self.trajectory)
            self.env.processEp(R, S, T)
            for expe in reversed(self.trajectory):
                self.buffer.append(expe.copy())

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def make_input(self, state):
        input = [np.reshape(state, (1, self.actor.s_dim[0]))]
        return input

    def act(self, state):
        input = self.make_input(state)
        action = self.actor.model.predict(input, batch_size=1)
        noise = np.random.normal(0., 0.1, size=action.shape)
        action = noise + action
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = action.squeeze()
        return action



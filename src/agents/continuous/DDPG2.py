import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import ActorDDPG, CriticDDPG, ActorCriticDDPG
from agents.agent import Agent
from buffers import ReplayBuffer

class DDPG2(Agent):

    def __init__(self, args, env, env_test, logger):
        super(DDPG2, self).__init__(args, env, env_test, logger)
        self.args = args
        self.init(args, env)
        for metric in self.actorCritic.criticModel.metrics_names:
            self.metrics[self.actorCritic.criticModel.name + '_' + metric] = 0

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
            a1 = self.actorCritic.actorTmodel.predict_on_batch(s1)
            a1 = np.clip(a1, self.env.action_space.low, self.env.action_space.high)
            q = self.actorCritic.criticTmodel.predict_on_batch([s1, a1])
            targets = r + (1 - t) * self.actorCritic.gamma * np.squeeze(q)
            targets = np.clip(targets, self.env.minR / (1 - self.actorCritic.gamma), self.env.maxR)
            inputs = [s0, a0]
            loss = self.actorCritic.criticModel.train_on_batch(inputs, targets)
            for i, metric in enumerate(self.actorCritic.criticModel.metrics_names):
                self.metrics[self.actorCritic.criticModel.name + '_' + metric] += loss[i]
            self.actorCritic.trainActor([s0])

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

            self.actorCritic.target_train()

    def reset(self):

        if self.trajectory:
            T = int(self.trajectory[-1]['terminal'])
            R = np.sum([exp['reward'] for exp in self.trajectory])
            S = len(self.trajectory)
            self.env.processEp(R, S, T)
            for expe in reversed(self.trajectory):
                self.buffer.append(expe.copy())

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def make_input(self, state):
        input = [np.reshape(state, (1, self.actorCritic.s_dim[0]))]
        return input

    def act(self, state):
        input = self.make_input(state)
        action = self.actorCritic.actorModel.predict(input, batch_size=1)
        noise = np.random.normal(0., 0.1, size=action.shape)
        action = noise + action
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = action.squeeze()
        return action



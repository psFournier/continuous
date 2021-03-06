import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import CriticDQN
from agents.agent import Agent
from buffers import ReplayBuffer
import keras.backend as K

class DQN(Agent):

    def __init__(self, args, env, env_test, logger):
        super(DQN, self).__init__(args, env, env_test, logger)
        self.args = args
        self.init(args, env)

    def init(self, args ,env):
        names = ['state0', 'action', 'state1', 'reward', 'terminal']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy())
        self.critic = CriticDQN(args, env)
        for metric_name in ['loss_dqn', 'qval', 'val']:
            self.metrics[metric_name] = 0

    def train(self):

        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            s0, a0, s1, r, t = [exp[name] for name in self.buffer.names]
            targets_dqn = self.critic.get_targets_dqn(r, t, s1)
            inputs = [s0, a0]
            loss = self.critic.criticModel.train_on_batch(inputs, targets_dqn)
            for i, metric in enumerate(self.critic.criticModel.metrics_names):
                self.metrics[metric] += loss[i]

            self.critic.target_train()

    def reset(self):

        if self.trajectory:
            R = np.sum([self.env.unshape(exp['reward'], exp['terminal']) for exp in self.trajectory])
            self.env.processEp(R)
            for expe in reversed(self.trajectory):
                self.buffer.append(expe.copy())

            if self.args['--imit'] != '0':
                Es = [0]
                for i, expe in enumerate(reversed(self.trajectory)):
                    if self.trajectory[-1]['terminal']:
                        Es[0] = Es[0] * self.critic.gamma + expe['reward']
                        expe['expVal'] = Es[0]
                    else:
                        expe['expVal'] = -self.ep_steps
                    self.bufferImit.append(expe.copy())

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def make_input(self, state, mode):
        input = [np.reshape(state, (1, self.critic.s_dim[0]))]
        input.append(np.expand_dims([0.5], axis=0))
        return input

    def act(self, state, mode='train'):
        input = self.make_input(state, mode)
        actionProbs = self.critic.actionProbs(input)
        if mode=='train':
            action = np.random.choice(range(self.env.action_dim), p=actionProbs[0].squeeze())
        else:
            action = np.argmax(actionProbs[0])
        return np.expand_dims(action, axis=1)



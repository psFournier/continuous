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

    def init(self, args ,env):
        names = ['s0', 'a', 's1', 'r', 't']
        metrics = ['loss_dqn', 'loss_actor']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
        self.actorCritic = ActorCriticDDPG(args, env)
        for metric in metrics:
            self.metrics[metric] = 0

    def train(self):

        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            targets_dqn = self.actorCritic.get_targets_dqn(exp['r'], exp['t'], exp['s1'])
            inputs = [exp['s0'], exp['a'], targets_dqn]
            loss_dqn = self.actorCritic.trainQval(inputs)
            loss_actor = self.actorCritic.trainActor([exp['s0']])
            self.metrics['loss_dqn'] += np.squeeze(loss_dqn)
            self.metrics['loss_actor'] += np.squeeze(loss_actor)

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

    def make_input(self, state, mode):
        input = [np.expand_dims(state, axis=0)]
        return input

    def reset(self):

        if self.trajectory:

            R = np.sum([self.env.unshape(exp['reward'], exp['terminal']) for exp in self.trajectory])
            self.env.queue.append(R)

            for i, expe in enumerate(reversed(self.trajectory)):
                _ = self.buffer.append(expe.copy())

            self.trajectory.clear()

        state = self.env.reset()
        self.episode_step = 0

        return state

    def act(self, state, mode='train'):
        input = self.make_input(state, mode)
        action = self.actorCritic.action(input)[0]
        if mode == 'train':
            noise = np.random.normal(0., 0.1, size=action[0].shape)
            action = noise + action
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = action.squeeze()
        return action



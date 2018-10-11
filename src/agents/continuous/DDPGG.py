import numpy as np
RENDER_TRAIN = False
INVERTED_GRADIENTS = True
from networks import ActorCriticDDPGG
from buffers import ReplayBuffer
from agents import DDPG

class DDPGG(DDPG):

    def __init__(self, args, env, env_test, logger):
        super(DDPGG, self).__init__(args, env, env_test, logger)

    def init(self, args ,env):
        names = ['s0', 'a', 's1', 'r', 't', 'g']
        metrics = ['loss_dqn', 'loss_actor']
        self.buffer = ReplayBuffer(limit=int(1e6), names=names.copy(), args=args)
        self.actorCritic = ActorCriticDDPGG(args, env)
        for metric in metrics:
            self.metrics[metric] = 0

    def train(self):

        if self.buffer.nb_entries > self.batch_size:
            exp = self.buffer.sample(self.batch_size)
            targets_dqn = self.actorCritic.get_targets_dqn(exp['r'], exp['t'], exp['s1'], exp['g'])
            inputs = [exp['s0'], exp['a'], exp['g'], targets_dqn]
            loss_dqn = self.actorCritic.trainQval(inputs)
            action, criticActionGrads, invertedCriticActionGrads = self.actorCritic.trainActor([exp['s0'], exp['g']])
            self.metrics['loss_dqn'] += np.squeeze(loss_dqn)


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
        if mode == 'train':
            input = [np.expand_dims(i, axis=0) for i in [state, self.env.goal]]
        else:
            input = [np.expand_dims(i, axis=0) for i in [state, self.env_test.goal]]
        return input



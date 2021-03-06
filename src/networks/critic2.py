from keras.models import Model
from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape, Dropout
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum, Dot
from .criticDQNG import CriticDQNG
import numpy as np
from keras.losses import mse

class Critic(object):
    def __init__(self, args, env):
        self.g_dim = env.goal_dim
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = (1,)
        self.gamma = 0.99
        self.lr_imit = float(args['--lrimit'])
        self.w_i = float(args['--wimit'])
        self.margin = float(args['--margin'])
        self.network = float(args['--network'])
        self.filter = int(args['--filter'])
        self.num_actions = env.action_dim
        self.initModels()
        self.initTargetModels()

    def initModels(self):

        ### Inputs
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        T = Input(shape=(1,), dtype='uint8')
        TARGETS = Input(shape=(1,))
        MCR = Input(shape=(1,), dtype='float32')

        ### Q values and action models
        qvals = self.create_critic_network(S)
        qvals_a_t = K.reshape(qvals, shape=(self.env.action_dim, self.env.N))
        self.model = Model([S], qvals_a_t)
        self.qvals_a_t = K.function(inputs=[S], outputs=[qvals_a_t], updates=None)
        self.actionProbs = K.function(inputs=[S], outputs=[actionProbs], updates=None)
        taskFilter = K.squeeze(K.one_hot(T, self.env.Ntasks), axis=2)
        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        qvalTask = K.sum(taskFilter * qvals, axis=2, keepdims=True)
        qval = K.sum(actionFilter * qvalTask, axis=1, keepdims=True)
        self.qval = K.function(inputs=[S, A, T], outputs=[qval], updates=None)

        ### DQN Training
        l2errors = K.square(qval - TARGETS)
        loss_dqn = K.mean(l2errors, axis=0)
        inputs_dqn = [S, A, T, TARGETS]
        updates_dqn = Adam(lr=0.001).get_updates(loss_dqn, self.model.trainable_weights)
        self.metrics_dqn_names = ['loss_dqn', 'qval']
        metrics_dqn = [loss_dqn, qval]
        self.train = K.function(inputs_dqn, metrics_dqn, updates_dqn)

        ### Large margin loss
        onehot = 1 - K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        # onehotMargin = K.repeat_elements(self.margin, self.num_actions, axis=1) * onehot
        imit = (K.max(qvals + self.margin * onehot, axis=1, keepdims=True) - qval)

        ### Suboptimal demos
        val = K.max(qvals, axis=1, keepdims=True)
        advClip = K.cast(K.greater(MCR, val), dtype='float32')
        goodexp = K.sum(advClip)

        ### Imitation
        if self.filter == 0:
            loss_imit = K.mean(imit, axis=0)
            loss_dqn_imit = K.mean(l2errors, axis=0)
        elif self.filter == 1:
            loss_imit = K.mean(imit * advClip, axis=0)
            loss_dqn_imit = K.mean(l2errors, axis=0)
        else:
            loss_imit = K.mean(imit * advClip, axis=0)
            loss_dqn_imit = K.mean(l2errors * advClip, axis=0)
        loss = loss_dqn_imit + self.w_i * loss_imit
        inputs_imit = [S, A, G, M, TARGETS, MCR]
        self.metrics_imit_names = ['loss_imit', 'loss_dqn_imit', 'qvals', 'goodexp']
        metrics_imit = [loss_imit, loss_dqn_imit, qvals, goodexp]
        updates_imit = Adam(lr=self.lr_imit).get_updates(loss, self.model.trainable_weights)
        self.imit = K.function(inputs_imit, metrics_imit, updates_imit)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        T = Input(shape=(1,), dtype='uint8')
        A = Input(shape=(1,), dtype='uint8')
        Tqvals = self.create_critic_network(S)
        self.Tmodel = Model([S], Tqvals)
        actionFilter = K.squeeze(K.one_hot(T * self.num_actions + A, self.num_actions * self.env.Ntasks), axis=1)
        Tqval = K.sum(actionFilter * Tqvals, axis=1, keepdims=True)
        self.Tqval = K.function(inputs=[S, A, T], outputs=[Tqval], updates=None)
        self.target_train()

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.Tmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.Tmodel.set_weights(target_weights)

    def compute_targets(self, r, t, q):
        targets = r + (1 - t) * self.gamma * np.squeeze(q)
        targets = np.clip(targets, 0, self.env.R)
        return targets

    def get_targets_dqn(self, r, t, s, g=None, m=None):
        qvals = self.qvals([s, g, m])[0]
        a1 = np.expand_dims(np.argmax(qvals, axis=1), axis=1)
        q = self.Tqval([s, g, m, a1])[0]
        targets_dqn = self.compute_targets(r, t, q)
        return np.expand_dims(targets_dqn, axis=1)

    def create_critic_network(self, S):
        h1 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))(S)
        h2 = Dense(300, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))(h1)
        Q_values = Dense(self.env.action_dim * self.env.N,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         kernel_regularizer=l2(0.01),
                         bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(h2)
        return Q_values
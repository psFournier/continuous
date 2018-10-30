from keras.models import Model
from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNG import CriticDQNG
import numpy as np
from keras.losses import mse

class CriticDQNGM3(object):
    def __init__(self, args, env):
        self.g_dim = env.goal_dim
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = (1,)
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.w_i = float(args['--wimit'])
        self.margin = float(args['--margin'])
        self.num_actions = env.action_dim
        self.optimizer = Adam(lr=self.learning_rate)
        self.initModels()
        self.initTargetModels()

    def initModels(self):

        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        TARGETS = Input(shape=(1,))
        MCR = Input(shape=(1,), dtype='float32')

        qvals = self.create_critic_network(S, G, M)
        self.model = Model([S, G, M], qvals)
        self.qvals = K.function(inputs=[S, G, M], outputs=[qvals], updates=None)

        actionProbs = K.softmax(qvals)
        self.actionProbs = K.function(inputs=[S, G, M], outputs=[actionProbs], updates=None)

        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        actionProb = K.sum(actionFilter * actionProbs, axis=1, keepdims=True)
        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        self.qval = K.function(inputs=[S, G, M, A], outputs=[qval], updates=None)

        val = K.max(qvals, axis=1, keepdims=True)
        self.val = K.function(inputs=[S, G, M], outputs=[val], updates=None)

        qvalWidth = K.max(qvals, axis=1, keepdims=True) - K.min(qvals, axis=1, keepdims=True)
        onehot = 1 - K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        onehotMargin = K.repeat_elements(self.margin * qvalWidth, self.num_actions, axis=1) * onehot
        imit = (K.max(qvals + onehotMargin, axis=1, keepdims=True) - qval)
        advantage = K.maximum(MCR - val, 0)
        imitFiltered = imit * advantage
        loss_imit = K.mean(imitFiltered, axis=0)

        inputs = [S, A, G, M, TARGETS, MCR]
        outputs = [loss_dqn, val, qval, loss_imit, K.sum(advantage), actionProb]

        updates = self.optimizer.get_updates(loss_dqn + self.w_i * loss_imit, self.model.trainable_weights)
        self.train = K.function(inputs, outputs, updates)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        A = Input(shape=(1,), dtype='uint8')
        Tqvals = self.create_critic_network(S, G, M)
        self.Tmodel = Model([S, G, M], Tqvals)

        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        Tqval = K.sum(actionFilter * Tqvals, axis=1, keepdims=True)
        self.Tqval = K.function(inputs=[S, G, M, A], outputs=[Tqval], updates=None)

        self.target_train()

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.Tmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.Tmodel.set_weights(target_weights)

    def compute_targets(self, r, t, q):
        targets = r + (1 - t) * self.gamma * np.squeeze(q)
        targets = np.clip(targets, self.env.minQ, self.env.maxQ)
        return targets

    def get_targets_dqn(self, r, t, s, g=None, m=None):
        qvals = self.qvals([s, g, m])[0]
        a1 = np.expand_dims(np.argmax(qvals, axis=1), axis=1)
        q = self.Tqval([s, g, m, a1])[0]
        targets_dqn = self.compute_targets(r, t, q)
        return np.expand_dims(targets_dqn, axis=1)

    def create_critic_network(self, S, G=None, M=None):
        L1 = concatenate([multiply([subtract([S, G]), M]), S])
        L2 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))(L1)
        L3 = Dense(300, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))(L2)
        Q_values = Dense(self.env.action_dim,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         kernel_regularizer=l2(0.01),
                         bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))(L3)
        return Q_values
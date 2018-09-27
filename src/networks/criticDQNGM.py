from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNG import CriticDQNG
import numpy as np
from keras.losses import mse

class CriticDQNGM(CriticDQNG):
    def __init__(self, args, env):
        super(CriticDQNGM, self).__init__(args, env)

    def initModels(self):
        w0, w1, w2 = float(self.args['--w0']), float(self.args['--w1']), float(self.args['--w2'])

        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        T = Input(shape=(1,), dtype='float32')
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        E = Input(shape=(1,), dtype='float32')
        TARGETS = Input(shape=(1,))

        qvals = self.create_critic_network(S, G, M)
        self.model = Model([S, G, M], qvals)
        self.qvals = K.function(inputs=[S, G, M], outputs=[qvals], updates=None)

        actionProbs = K.softmax(qvals / T)
        self.actionProbs = K.function(inputs=[S, G, M, T], outputs=[actionProbs], updates=None)

        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        self.qval = K.function(inputs=[S, G, M, A], outputs=[qval], updates=None)

        val = K.max(qvals, axis=1, keepdims=True)
        self.val = K.function(inputs=[S, G, M], outputs=[val], updates=None)

        if self.args['--imit'] == '1':
            actionProb = K.sum(actionFilter * actionProbs, axis=1, keepdims=True)
            imit = -K.log(actionProb)
        elif self.args['--imit'] == '2':
            width = float(self.args['--margin']) * (
            K.max(qvals, axis=1, keepdims=True) - K.min(qvals, axis=1, keepdims=True))
            one_hot = 1 - K.squeeze(K.one_hot(A, self.num_actions), axis=1)
            width = K.repeat_elements(width, self.num_actions, axis=1)
            margin = width * one_hot
            imit = (K.max(qvals + margin, axis=1, keepdims=True) - qval)
        else:
            raise RuntimeError

        adv0 = K.maximum(E - val, 0)
        adv1 = K.cast(K.greater(E, val), dtype='float32')
        good_exp = K.sum(adv1)

        if self.args['--filter'] == '1':
            imit *= adv0
        elif self.args['--filter'] == '2':
            imit *= adv1
        else:
            raise RuntimeError

        loss = 0
        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        loss += w0 * loss_dqn
        loss_imit = K.mean(imit, axis=0)
        loss += w1 * loss_imit

        self.updatesDqn = self.optimizer.get_updates(params=self.model.trainable_weights, loss=loss_dqn)
        self.trainDqn = K.function(inputs=[S, A, G, M, TARGETS],
                                outputs=[loss_dqn, val, qval],
                                updates=self.updatesDqn)

        self.updatesAll = self.optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.trainAll = K.function(inputs=[S, A, G, M, T, E, TARGETS],
                                   outputs=[loss_dqn, loss_imit, good_exp, val, qval, adv0],
                                   updates=self.updatesAll)

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

    def get_targets_dqn(self, r, t, s, g=None, m=None):
        qvals = self.qvals([s, g, m])[0]
        a1 = np.expand_dims(np.argmax(qvals, axis=1), axis=1)
        q = self.Tqval([s, g, m, a1])[0]
        targets_dqn = self.compute_targets(r, t, q)
        return np.expand_dims(targets_dqn, axis=1)

    def create_critic_network(self, S, G=None, M=None):
        l1 = concatenate([multiply([subtract([S, G]), M]), S])
        l2 = Dense(200, activation="relu")(l1)
        l3 = Dense(200, activation="relu")(l2)
        Q_values = Dense(self.num_actions)(l3)
        return Q_values
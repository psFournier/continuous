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

class CriticDQNGM(CriticDQNG):
    def __init__(self, args, env):
        super(CriticDQNGM, self).__init__(args, env)

    def initModels(self):

        w = float(self.args['--wimit'])
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        PA = Input(shape=(1,))
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        TARGETS = Input(shape=(1,))

        qvals = self.create_critic_network(S, G, M)
        self.model = Model([S, G, M], qvals)
        self.qvals = K.function(inputs=[S, G, M], outputs=[qvals], updates=None)

        actionProbs = K.softmax(qvals)
        self.actionProbs = K.function(inputs=[S, G, M], outputs=[actionProbs], updates=None)

        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        actionProb = K.sum(actionFilter * actionProbs, axis=1, keepdims=True)
        self.qval = K.function(inputs=[S, G, M, A], outputs=[qval], updates=None)

        val = K.max(qvals, axis=1, keepdims=True)
        self.val = K.function(inputs=[S, G, M], outputs=[val], updates=None)

        # weights = PA / actionProb
        losses = K.square(qval - TARGETS)
        loss_dqn = K.mean(losses, axis=0)
        inputs = [S, A, G, M, TARGETS]
        outputs = [loss_dqn, val, qval]

        updates_dqn = self.optimizer.get_updates(loss_dqn, self.model.trainable_weights)
        self.train_dqn = K.function(inputs, outputs, updates_dqn)

        if w != 0:
            margin = float(self.args['--margin'])
            qvalWidth = K.max(qvals, axis=1, keepdims=True) - K.min(qvals, axis=1, keepdims=True)
            onehot = 1 - K.squeeze(K.one_hot(A, self.num_actions), axis=1)
            onehotMargin = K.repeat_elements(margin * qvalWidth, self.num_actions, axis=1) * onehot
            imit = (K.max(qvals + onehotMargin, axis=1, keepdims=True) - qval)

            # imit = -K.log(actionProb)

            MCR = Input(shape=(1,), dtype='float32')
            # adv = K.maximum(E - val, 0)
            advClip = K.cast(K.greater(MCR, val), dtype='float32')
            good_exp = K.sum(advClip)
            imit *= advClip

            loss_imit = K.mean(imit, axis=0)
            loss = loss_dqn + 0 * loss_imit
            inputs += [MCR]
            outputs += [loss_imit, imit, good_exp]
            # inputs.append(E)
            # outputs += [loss_imit, good_exp, adv]

            updates_imit = self.optimizer.get_updates(loss, self.model.trainable_weights)
            self.train_imit = K.function(inputs, outputs, updates_imit)

        # if self.args['--imit'] == '1':
        #     actionProb = K.sum(actionFilter * actionProbs, axis=1, keepdims=True)
        #     imit = -K.log(actionProb)


        # self.updatesDqn = self.optimizer.get_updates(params=self.model.trainable_weights, loss=loss_dqn)
        # self.trainDqn = K.function(inputs=[S, A, G, M, TARGETS],
        #                         outputs=[loss_dqn, val, qval],
        #                         updates=self.updatesDqn)
        #
        # self.updatesAll = self.optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        # self.trainAll = K.function(inputs=[S, A, G, M, T, E, TARGETS],
        #                            outputs=[loss_dqn, loss_imit, good_exp, val, qval, adv0],
        #                            updates=self.updatesAll)

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
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
        actionFilter = Lambda(lambda x: K.squeeze(K.one_hot(x, self.num_actions), axis=1), (self.num_actions,))(A)

        actionProbs = Lambda(lambda x: K.softmax(x[0] / x[1]), (self.num_actions,))([qvals, T])
        self.actionProbsModel = Model([S, G, M, T], actionProbs)

        qval = Lambda(lambda x: K.sum(x[0] * x[1], axis=1, keepdims=True), (1,))([actionFilter, qvals])
        self.criticModel = Model([S, A, G, M], qval)

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

        val = K.max(qvals, axis=1, keepdims=True)
        adv0 = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
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

        self.updatesDqn = self.optimizer.get_updates(params=self.criticModel.trainable_weights, loss=loss_dqn)
        self.trainDqn = K.function(inputs=[S, A, G, M, TARGETS],
                                outputs=[loss_dqn, val, qval],
                                updates=self.updatesDqn)

        self.updatesAll = self.optimizer.get_updates(params=self.criticModel.trainable_weights, loss=loss)
        self.trainAll = K.function(inputs=[S, A, G, M, T, E, TARGETS],
                                   outputs=[loss_dqn, loss_imit, good_exp, val, qval],
                                   updates=self.updatesAll)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        targetQvals = self.create_critic_network(S, G, M)
        targetQval = Lambda(self.actionFilterFn, output_shape=(1,))([A, targetQvals])
        self.criticTmodel = Model([S, A, G, M], targetQval)
        self.target_train()

    def get_targets_dqn(self, r, t, s, g=None, m=None):
        temp = np.expand_dims([0.5], axis=0)
        a1Probs = self.actionProbsModel.predict_on_batch([s, g, m, temp])
        a1 = np.argmax(a1Probs, axis=1)
        q = self.criticTmodel.predict_on_batch([s, a1, g, m])
        targets_dqn = self.compute_targets(r, t, q)
        return np.expand_dims(targets_dqn, axis=1)

    def create_critic_network(self, S, G=None, M=None):
        l1 = concatenate([multiply([subtract([S, G]), M]), S])
        l2 = Dense(200, activation="relu")(l1)
        l3 = Dense(200, activation="relu")(l2)
        Q_values = Dense(self.num_actions)(l3)
        return Q_values
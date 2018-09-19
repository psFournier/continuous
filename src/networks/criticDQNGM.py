from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNG import CriticDQNG
import numpy as np

class CriticDQNGM(CriticDQNG):
    def __init__(self, args, env):
        super(CriticDQNGM, self).__init__(args, env)

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        T = Input(shape=(1,), dtype='float32')
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        qvals = self.create_critic_network(S, G, M)
        actionProbs = Lambda(lambda x: K.softmax(x[0] / x[1]))([qvals, T])
        self.actionProbsModel = Model([S, G, M, T], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,))([A, qvals])
        self.qvalModel = Model([S, A, G, M], qval)
        self.qvalModel.compile(loss='mse', optimizer=self.optimizer)
        self.qvalModel.metrics_tensors = [qval]

        if self.args['--imit'] == '1':
            E = Input(shape=(1,), dtype='float32')
            actionProb = Lambda(self.actionFilterFn, output_shape=(1,))([A, actionProbs])
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(lambda x: -K.log(x[0]) * x[1], name='imit')([actionProb, advantage])
            self.imitModel = Model([S, A, G, M, T, E], [qval, imit, advantage])
            self.imitModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[float(self.args['--w0']), float(self.args['--w1']), float(self.args['--w2'])],
                                   optimizer=self.optimizer)

        if self.args['--imit'] == '2':
            E = Input(shape=(1,), dtype='float32')
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(self.marginFn, output_shape=(1,), name='imit')([A, qvals, qval, advantage])
            self.imitModel = Model([S, A, G, M, E], [qval, imit, advantage])
            self.imitModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[float(self.args['--w0']), float(self.args['--w1']),
                                                 float(self.args['--w2'])],
                                   optimizer=self.optimizer)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        targetQvals = self.create_critic_network(S, G, M)
        targetQval = Lambda(self.actionFilterFn, output_shape=(1,))([A, targetQvals])
        self.qvalTModel = Model([S, A, G, M], targetQval)
        self.target_train()

    def get_targets_dqn(self, r, t, s, g=None, m=None):
        temp = np.expand_dims([0.5], axis=0)
        a1Probs = self.actionProbsModel.predict_on_batch([s, g, m, temp])
        a1 = np.argmax(a1Probs, axis=1)
        q = self.qvalTModel.predict_on_batch([s, a1, g, m])
        targets_dqn = self.compute_targets(r, t, q)
        return targets_dqn

    def create_critic_network(self, S, G=None, M=None):
        l1 = concatenate([multiply([subtract([S, G]), M]), S])
        l2 = Dense(200, activation="relu")(l1)
        l3 = Dense(200, activation="relu")(l2)
        Q_values = Dense(self.num_actions)(l3)
        return Q_values




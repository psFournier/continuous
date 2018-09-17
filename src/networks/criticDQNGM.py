from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNG import CriticDQNG

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

        if self.args['--imit'] == '0':
            self.qvalModel = Model([S, A, G, M], qval)
            self.qvalModel.compile(loss='mse', optimizer=self.optimizer)
            self.qvalModel.metrics_tensors = [qval]

        if self.args['--imit'] == '1':
            E = Input(shape=(1,), dtype='float32')
            actionProb = Lambda(self.actionFilterFn, output_shape=(1,))([A, actionProbs])
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(lambda x: -K.log(x[0]) * x[1], name='imit')([actionProb, advantage])
            self.qvalModel = Model([S, A, G, M, T, E], [qval, imit, advantage])
            self.qvalModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[1, float(self.args['--w1']), float(self.args['--w2'])],
                                   optimizer=self.optimizer)

        if self.args['--imit'] == '2':
            E = Input(shape=(1,), dtype='float32')
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(self.marginFn, output_shape=(1,), name='imit')([A, qvals, qval, advantage])
            self.qvalModel = Model([S, A, G, M, E], [qval, imit, advantage])
            self.qvalModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[1, float(self.args['--w1']), float(self.args['--w2'])],
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

    def create_critic_network(self, S, G=None, M=None):
        if self.args['--network'] == '1':
            l1 = concatenate([S, G, M])
            l2 = Dense(400, activation="relu")(l1)
            l3 = concatenate([l2, G, M])
            l4 = Dense(300, activation="relu")(l3)
            Q_values = Dense(self.num_actions)(l4)
        elif self.args['--network'] == '2':
            l1 = multiply([subtract([S, G]), M])
            l2 = concatenate([l1, S])
            l3 = Dense(400, activation="relu")(l2)
            l4 = Dense(300, activation="relu")(l3)
            Q_values = Dense(self.num_actions)(l4)
        elif self.args['--network'] == '3':
            shared_l = Dense(200, activation='relu')
            l1 = shared_l(S)
            l2 = shared_l(G)
            l3 = shared_l(M)
            l4 = multiply([subtract([l1, l2]), l3])
            l5 = Dense(200, activation="relu")(l4)
            l6 = Dense(300, activation="relu")(l5)
            Q_values = Dense(self.num_actions)(l6)
        else:
            l1 = concatenate([S, G, M])
            l2 = Dense(400, activation="relu")(l1)
            l3 = Dense(300, activation="relu")(l2)
            Q_values = Dense(self.num_actions)(l3)
        return Q_values




from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQN import  CriticDQN

class CriticDQNG(CriticDQN):
    def __init__(self, args, env):
        self.g_dim = env.goal_dim
        super(CriticDQNG, self).__init__(args, env)

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.g_dim)
        qvals = self.create_critic_network(S, G)
        actionProbs = Lambda(lambda x: K.softmax(x))(qvals)
        self.actionProbsModel = Model([S, G], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,))([A, qvals])

        if self.args['--imit'] == '0':
            self.qvalModel = Model([S, A, G], qval)
            self.qvalModel.compile(loss='mse', optimizer=self.optimizer)

        if self.args['--imit'] == '1':
            E = Input(shape=(1,), dtype='float32')
            actionProb = Lambda(self.actionFilterFn, output_shape=(1,))([A, actionProbs])
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(lambda x: -K.log(x[0]) * x[1], name='imit')([actionProb, advantage])
            self.qvalModel = Model([S, A, G, E], [qval, imit, advantage])
            self.qvalModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[1, float(self.args['--w1']), float(self.args['--w2'])],
                                   optimizer=self.optimizer)

        if self.args['--imit'] == '2':
            E = Input(shape=(1,), dtype='float32')
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(self.marginFn, output_shape=(1,), name='imit')([A, qvals, qval, advantage])
            self.qvalModel = Model([S, A, G, E], [qval, imit, advantage])
            self.qvalModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[1, float(self.args['--w1']), float(self.args['--w2'])],
                                   optimizer=self.optimizer)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.g_dim)
        targetQvals = self.create_critic_network(S, G)
        targetQval = Lambda(self.actionFilterFn, output_shape=(1,))([A, targetQvals])
        self.qvalTModel = Model([S, A, G], targetQval)
        self.target_train()

    def create_critic_network(self, S, G=None):
        c1 = concatenate(inputs=[S, G])
        l1 = Dense(400, activation="relu")(c1)
        c2 = concatenate([l1, G])
        l2 = Dense(300, activation="relu")(c2)
        Q_values = Dense(self.num_actions)(l2)
        return Q_values




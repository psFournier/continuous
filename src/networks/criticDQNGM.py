from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNG import CriticDQNG

class CriticDQNGM(CriticDQNG):
    def __init__(self, s_dim, g_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        super(CriticDQNGM, self).__init__(s_dim, g_dim, num_a, gamma, tau, learning_rate)

    def initModels(self):
        S = Input(shape=self.s_dim)
        G = Input(shape=self.g_dim)
        A = Input(shape=(1,), dtype='uint8')
        M = Input(shape=self.g_dim)
        qvals = self.create_critic_network(S, G, M)
        actionProbs = Lambda(lambda x: K.softmax(x))(qvals)
        self.actionProbsModel = Model([S, G, M], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,))([A, qvals])
        self.qvalModel = Model([S, A, G, M], qval)
        self.qvalModel.compile(loss='mse', optimizer=self.optimizer)
        self.qvalModel.metrics_tensors += [qval]

    def create_critic_network(self, S, G=None, M=None):
        c1 = concatenate(inputs=[S, G, M])
        l1 = Dense(400, activation="relu")(c1)
        c2 = concatenate([l1, G, M])
        l2 = Dense(300, activation="relu")(c2)
        Q_values = Dense(self.num_actions)(l2)
        return Q_values




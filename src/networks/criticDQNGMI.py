from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNGM import CriticDQNGM

class CriticDQNGMI(CriticDQNGM):
    def __init__(self, s_dim, g_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001, weight1=1, weight2=1, margin=0.3):
        self.weights = [1, weight1, weight2]
        self.margin = margin
        super(CriticDQNGMI, self).__init__(s_dim, g_dim, num_a, gamma, tau, learning_rate)

    def marginFn(self, inputs):
        a = inputs[0]
        v = inputs[1]
        q = inputs[2]
        margin = self.margin * (1 - K.one_hot(a, self.num_actions))
        vmax = K.max(v)
        vmin = K.min(v)
        vnorm = (v - vmin) / (vmax - vmin)
        qnorm = (q - vmin) / (vmax - vmin)
        return K.max(vnorm + margin) - qnorm

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        E = Input(shape=(1,), dtype='float32')
        qvals = self.create_critic_network(S, G, M)
        actionProbs = Lambda(lambda x: K.softmax(x))(qvals)
        self.actionProbsModel = Model([S, G, M], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,))([A, qvals])
        val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
        advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0))([E, val])
        margin = Lambda(self.marginFn, output_shape=(1,))([A, qvals, qval])
        imitLossPolicy = multiply([advantage, margin])
        self.qvalModel = Model([S, A, G, M, E], [qval, imitLossPolicy, advantage])
        self.qvalModel.compile(loss=['mse', 'mae', 'mse'],
                               loss_weights=self.weights,
                               optimizer=self.optimizer)
        self.qvalModel.metrics_tensors += [qval]




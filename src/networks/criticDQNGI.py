from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNG import  CriticDQNG

class CriticDQNGI(CriticDQNG):
    def __init__(self, s_dim, g_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001, weight1=1, weight2=1):
        self.weights = [1, weight1, weight2]
        super(CriticDQNGI, self).__init__(s_dim, g_dim, num_a, gamma, tau, learning_rate)

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        G = Input(shape=self.g_dim)
        E = Input(shape=(1,), dtype='float32')
        qvals = self.create_critic_network(S, G)
        actionProbs = Lambda(lambda x: K.softmax(x))(qvals)
        self.actionProbsModel = Model([S, G], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,))([A, qvals])
        actionProb = Lambda(self.actionFilterFn, output_shape=(1,))([A, actionProbs])
        val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
        advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0))([E, val])
        imitLossPolicy = Lambda(lambda x: -K.log(x[0]) * x[1])([actionProb, advantage])
        self.qvalModel = Model([S, A, G, E], [qval, imitLossPolicy, advantage])
        self.qvalModel.compile(loss=['mse', 'mae', 'mse'],
                               loss_weights=self.weights,
                               optimizer=self.optimizer)
        self.qvalModel.metrics_tensors += [qval]
from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
import keras.backend as K

class CriticDQN(object):
    def __init__(self, s_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        self.tau = tau
        self.s_dim = s_dim
        self.a_dim = (1,)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_actions = num_a
        self.optimizer = Adam(lr=self.learning_rate)
        self.initModels()
        self.initTargetModels()

    def actionFilterFn(self, inputs):
        filter = K.squeeze(K.one_hot(inputs[0], self.num_actions), axis=1)
        return K.sum(filter * inputs[1], axis=1, keepdims=True)

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        qvals = self.create_critic_network(S)
        actionProbs = Lambda(lambda x: K.softmax(x))(qvals)
        self.actionProbsModel = Model([S], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,))([A, qvals])
        self.qvalModel = Model([S, A], qval)
        self.qvalModel.compile(loss='mse', optimizer=self.optimizer)
        self.qvalModel.metrics_tensors += [qval]

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        targetQvals = self.create_critic_network(S)
        targetQval = Lambda(self.actionFilterFn, output_shape=(1,))([A, targetQvals])
        self.qvalTModel = Model([S, A], targetQval)
        self.target_train()

    def create_critic_network(self, S):
        l1 = Dense(400, activation="relu")(S)
        l2 = Dense(300, activation="relu")(l1)
        Q_values = Dense(self.num_actions)(l2)
        return Q_values

    def target_train(self):
        weights = self.qvalModel.get_weights()
        target_weights = self.qvalTModel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.qvalTModel.set_weights(target_weights)

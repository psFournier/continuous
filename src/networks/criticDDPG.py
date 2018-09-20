from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

class CriticDDPG(object):
    def __init__(self, args, env):
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = (1,)
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.optimizer = Adam(lr=self.learning_rate)
        self.args = args
        self.initModels()
        self.initTargetModels()

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,))
        qval = self.create_critic_network(S, A)
        grads = K.gradients(qval, A)
        self.gradsModel = Model([S,A], grads)
        self.model = Model([S,A], qval)
        self.model.compile(loss='mse', optimizer=self.optimizer)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        Tqval = self.create_critic_network(S, A)
        self.Tmodel = Model([S, A], Tqval)
        self.target_train()

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.Tmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.Tmodel.set_weights(target_weights)

    def create_critic_network(self, S, A):
        w = Dense(400, activation="relu")(S)
        h = concatenate([w, A])
        h3 = Dense(300, activation="relu")(h)
        V = Dense(1, activation='linear')(h3)
        return V




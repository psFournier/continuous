from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

class ActorCriticDDPG(object):
    def __init__(self, args, env):
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = env.action_dim
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.optimizer = Adam(lr=self.learning_rate)
        self.args = args
        self.initModels()
        self.initTargetModels()

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=self.a_dim)

        L1 = Dense(400, activation="relu")
        L1out = L1(S)
        L1out = concatenate([L1out, A])
        L2 = Dense(300, activation="relu")
        L2out = L2(L1out)
        L3 = Dense(1, activation='linear')
        qval = L3(L2out)
        self.criticModel = Model([S, A], qval)
        self.criticModel.compile(loss='mse', optimizer=self.optimizer)
        self.criticModel.metrics_tensors = [qval]

        act = self.create_actor_network(S)
        self.actorModel = Model(S, act)

        L1outTrain = concatenate([L1(S), act])
        L2outTrain = L2(L1outTrain)
        qvalTrain = L3(L2outTrain)
        loss = -K.mean(qvalTrain)
        updates = self.optimizer.get_updates(params=self.actorModel.trainable_weights, loss=loss)
        self.trainActor = K.function(inputs=[S], outputs=[], updates=updates)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=self.a_dim)
        Tqval = self.create_critic_network(S, A)
        self.criticTmodel = Model([S, A], Tqval)

        S = Input(shape=self.s_dim)
        Tact = self.create_actor_network(S)
        self.actorTmodel = Model(S, Tact)

        self.target_train()

    def target_train(self):
        weights = self.criticModel.get_weights()
        target_weights = self.criticTmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.criticTmodel.set_weights(target_weights)

        weights = self.actorModel.get_weights()
        target_weights = self.actorTmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.actorTmodel.set_weights(target_weights)

    def create_critic_network(self, S, A):
        w = Dense(400, activation="relu")(S)
        h = concatenate([w, A])
        h3 = Dense(300, activation="relu")(h)
        V = Dense(1, activation='linear')(h3)
        return V

    def create_actor_network(self, S):
        h0 = Dense(400, activation="relu")(S)
        h1 = Dense(300, activation="relu")(h0)
        V = Dense(self.a_dim[0], activation="tanh")(h1)
        return V




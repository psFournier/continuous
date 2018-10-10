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
        TARGETS = Input(shape=(1,))

        layers, qval = self.create_critic_network(S, A)
        self.qvalModel = Model([S, A], qval)
        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        self.updatesQval = self.optimizer.get_updates(params=self.qvalModel.trainable_weights, loss=loss_dqn)
        self.trainQval = K.function(inputs=[S, A, TARGETS], outputs=[loss_dqn], updates=self.updatesQval)

        action = self.create_actor_network(S)
        self.actionModel = Model(S, action)
        self.action = K.function(inputs=[S], outputs=[action], updates=None)
        L1, L2, L3 = layers
        qvalTrain = concatenate([L1(S), action])
        qvalTrain = L2(qvalTrain)
        qvalTrain = L3(qvalTrain)
        loss_actor = -K.mean(qvalTrain)
        self.updatesActor = self.optimizer.get_updates(params=self.actionModel.trainable_weights, loss=loss_actor)
        self.trainActor = K.function(inputs=[S], outputs=[loss_actor], updates=self.updatesActor)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=self.a_dim)
        Tlayers, Tqval = self.create_critic_network(S, A)
        self.TqvalModel = Model([S, A], Tqval)
        self.Tqval = K.function(inputs=[S, A], outputs=[Tqval], updates=None)

        S = Input(shape=self.s_dim)
        Taction = self.create_actor_network(S)
        self.TactionModel = Model(S, Taction)
        self.Taction = K.function(inputs=[S], outputs=[Taction], updates=None)

        self.target_train()

    def get_targets_dqn(self, r, t, s):

        a = self.Taction([s])[0]
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        q = self.Tqval([s, a])[0]
        targets = self.compute_targets(r, t, q)
        return np.expand_dims(targets, axis=1)

    def compute_targets(self, r, t, q):
        targets = r + (1 - t) * self.gamma * np.squeeze(q)
        targets = np.clip(targets, self.env.minQ, self.env.maxQ)
        return targets

    def target_train(self):
        weights = self.qvalModel.get_weights()
        target_weights = self.TqvalModel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.TqvalModel.set_weights(target_weights)

        weights = self.actionModel.get_weights()
        target_weights = self.TactionModel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.TactionModel.set_weights(target_weights)

    def create_critic_network(self, S, A):

        L1 = Dense(400, activation="relu")
        L1out = L1(S)
        L1out = concatenate([L1out, A])
        L2 = Dense(300, activation="relu")
        L2out = L2(L1out)
        L3 = Dense(1, activation='linear')
        qval = L3(L2out)
        return [L1, L2, L3], qval

    def create_actor_network(self, S):
        h0 = Dense(400, activation="relu")(S)
        h1 = Dense(300, activation="relu")(h0)
        V = Dense(self.a_dim[0], activation="tanh")(h1)
        return V




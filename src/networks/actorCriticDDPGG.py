from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from .actorCriticDDPG import ActorCriticDDPG

class ActorCriticDDPGG(ActorCriticDDPG):
    def __init__(self, args, env):
        self.g_dim = env.goal_dim
        super(ActorCriticDDPGG, self).__init__(args, env)

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=self.a_dim)
        G = Input(shape=self.g_dim)
        TARGETS = Input(shape=(1,))

        layers, qval = self.create_critic_network(S, A, G)
        self.qvalModel = Model([S, A, G], qval)
        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        self.updatesQval = self.optimizer.get_updates(params=self.qvalModel.trainable_weights, loss=loss_dqn)
        self.trainQval = K.function(inputs=[S, A, G, TARGETS], outputs=[loss_dqn], updates=self.updatesQval)

        action = self.create_actor_network(S, G)
        self.actionModel = Model([S, G], action)
        self.action = K.function(inputs=[S, G], outputs=[action], updates=None)
        L1, L2, L3 = layers
        qvalTrain = L1(concatenate([S, G]))
        qvalTrain = concatenate([qvalTrain, action])
        qvalTrain = L2(qvalTrain)
        qvalTrain = L3(qvalTrain)
        loss_actor = -K.mean(qvalTrain)
        self.updatesActor = self.optimizer.get_updates(params=self.actionModel.trainable_weights, loss=loss_actor)
        self.trainActor = K.function(inputs=[S, G], outputs=[loss_actor], updates=self.updatesActor)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=self.a_dim)
        G = Input(shape=self.g_dim)
        Tlayers, Tqval = self.create_critic_network(S, A, G)
        self.TqvalModel = Model([S, A, G], Tqval)
        self.Tqval = K.function(inputs=[S, A, G], outputs=[Tqval], updates=None)

        S = Input(shape=self.s_dim)
        G = Input(shape=self.g_dim)
        Taction = self.create_actor_network(S, G)
        self.TactionModel = Model([S, G], Taction)
        self.Taction = K.function(inputs=[S, G], outputs=[Taction], updates=None)

        self.target_train()

    def get_targets_dqn(self, r, t, s, g=None):

        a = self.Taction([s, g])[0]
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        q = self.Tqval([s, a, g])[0]
        targets = self.compute_targets(r, t, q)
        return np.expand_dims(targets, axis=1)

    def create_critic_network(self, S, A, G=None):

        L1 = Dense(400, activation="relu")
        L1out = L1(concatenate([S, G]))
        L1out = concatenate([L1out, A])
        L2 = Dense(300, activation="relu")
        L2out = L2(L1out)
        L3 = Dense(1, activation='linear')
        qval = L3(L2out)
        return [L1, L2, L3], qval

    def create_actor_network(self, S, G=None):
        h0 = Dense(400, activation="relu")(concatenate([S, G]))
        h1 = Dense(300, activation="relu")(h0)
        V = Dense(self.a_dim[0], activation="tanh")(h1)
        return V




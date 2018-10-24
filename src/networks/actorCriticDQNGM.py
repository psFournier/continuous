from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from keras.layers.merge import concatenate, multiply, add, subtract, maximum

class ActorCriticDQNGM(object):
    def __init__(self, args, env):
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = env.action_dim
        self.g_dim = env.goal_dim
        self.gamma = 0.99
        self.args = args
        self.initModels()
        self.initTargetModels()

    def initModels(self):

        S_c = Input(shape=self.s_dim)
        A_c = Input(shape=(1,), dtype='uint8')
        G_c = Input(shape=self.g_dim)
        M_c = Input(shape=self.g_dim)
        TARGETS = Input(shape=(1,))

        layers, qvals = self.create_critic_network(S_c, G_c, M_c)
        self.qvalsModel = Model([S_c, G_c, M_c], qvals)
        self.qvals = K.function(inputs=[S_c, G_c, M_c], outputs=[qvals], updates=None)

        actionFilter = K.squeeze(K.one_hot(A_c, self.a_dim), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        self.qval = K.function(inputs=[S_c, G_c, M_c, A_c], outputs=[qval], updates=None)

        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        inputs = [S_c, A_c, G_c, M_c, TARGETS]
        outputs = [loss_dqn, qval]
        self.updatesQval = Adam(lr=0.001).get_updates(params=self.qvalsModel.trainable_weights, loss=loss_dqn)
        self.trainCritic = K.function(inputs=inputs, outputs=outputs, updates=self.updatesQval)

        S_a = Input(shape=self.s_dim)
        G_a = Input(shape=self.g_dim)
        M_a = Input(shape=self.g_dim)
        probs = self.create_actor_network(S_a, G_a, M_a)
        self.probsModel = Model([S_a, G_a, M_a], probs)
        self.probs = K.function(inputs=[S_a, G_a, M_a], outputs=[probs], updates=None)

        L1, L2, L3 = layers
        input = concatenate([multiply([subtract([S_a, G_a]), M_a]), S_a])
        qvalTrain = L1(input)
        qvalTrain = L2(qvalTrain)
        qvalTrain = L3(qvalTrain)
        val = K.sum(qvalTrain * probs)
        inputs = [S_a, G_a, M_a]
        outputs = [val]
        self.updatesActor = Adam(lr=0.001).get_updates(params=self.probsModel.trainable_weights, loss=-val)
        self.trainActor = K.function(inputs=inputs, outputs=outputs, updates=self.updatesActor)

    def initTargetModels(self):
        S_c = Input(shape=self.s_dim)
        G_c = Input(shape=self.g_dim)
        M_c = Input(shape=self.g_dim)
        _, Tqvals = self.create_critic_network(S_c, G_c, M_c)
        self.TqvalsModel = Model([S_c, G_c, M_c], Tqvals)
        self.Tqvals = K.function(inputs=[S_c, G_c, M_c], outputs=[Tqvals])

        S_a = Input(shape=self.s_dim)
        G_a = Input(shape=self.g_dim)
        M_a = Input(shape=self.g_dim)
        Tprobs = self.create_actor_network(S_a, G_a, M_a)
        self.TprobsModel = Model([S_a, G_a, M_a], Tprobs)
        self.Tprobs = K.function(inputs=[S_a, G_a, M_a], outputs=[Tprobs])

        self.target_train()

    def get_targets_dqn(self, r, t, s, g, m):
        probs = self.Tprobs([s, g, m])[0]
        qvals = self.Tqvals([s, g, m])[0]
        q = qvals[np.argmax(probs, axis=1)]
        targets = self.compute_targets(r, t, q)
        return np.expand_dims(targets, axis=1)

    def compute_targets(self, r, t, q):
        targets = r + (1 - t) * self.gamma * np.squeeze(q)
        targets = np.clip(targets, self.env.minQ, self.env.maxQ)
        return targets

    def target_train(self):
        weights = self.qvalsModel.get_weights()
        target_weights = self.TqvalsModel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.TqvalsModel.set_weights(target_weights)

        weights = self.probsModel.get_weights()
        target_weights = self.TprobsModel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.TprobsModel.set_weights(target_weights)

    def create_critic_network(self, S, G=None, M=None):
        input = concatenate([multiply([subtract([S, G]), M]), S])
        L1 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))
        L1out = L1(input)
        L2 = Dense(300, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))
        L2out = L2(L1out)
        L3 = Dense(self.env.action_dim,
                         activation='linear',
                         kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                         kernel_regularizer=l2(0.01),
                         bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))
        Q_values = L3(L2out)
        return [L1, L2, L3], Q_values

    def create_actor_network(self, S, G, M):
        input = concatenate([multiply([subtract([S, G]), M]), S])
        h0 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform())(input)
        h1 = Dense(300, activation="relu",
                   kernel_initializer=lecun_uniform())(h0)
        probs = Dense(self.a_dim,
                  activation='softmax',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                  bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))(h1)
        return probs
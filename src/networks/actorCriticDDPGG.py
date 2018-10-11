from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from keras.optimizers import Adam
from .actorCriticDDPG import DDPGAdam
import keras.backend as K
import numpy as np
import tensorflow as tf
from .actorCriticDDPG import ActorCriticDDPG

class ActorCriticDDPGG(ActorCriticDDPG):
    def __init__(self, args, env):
        self.g_dim = env.goal_dim
        super(ActorCriticDDPGG, self).__init__(args, env)

    def initModels(self):

        S_c = Input(shape=self.s_dim)
        A_c = Input(shape=self.a_dim)
        G_c = Input(shape=self.g_dim)
        TARGETS = Input(shape=(1,))
        layers, qval = self.create_critic_network(S_c, A_c, G_c)
        self.qvalModel = Model([S_c, A_c, G_c], qval)
        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        self.updatesQval = Adam(lr=0.001).get_updates(params=self.qvalModel.trainable_weights, loss=loss_dqn)
        self.trainQval = K.function(inputs=[S_c, A_c, G_c, TARGETS], outputs=[loss_dqn], updates=self.updatesQval)

        S_a = Input(shape=self.s_dim)
        G_a = Input(shape=self.g_dim)
        action = self.create_actor_network(S_a, G_a)
        self.actionModel = Model([S_a, G_a], action)
        self.action = K.function(inputs=[S_a, G_a], outputs=[action], updates=None)

        L1, L2, L3 = layers
        qvalTrain = L1(concatenate([S_a, G_a]))
        qvalTrain = concatenate([qvalTrain, action])
        qvalTrain = L2(qvalTrain)
        qvalTrain = L3(qvalTrain)
        self.criticActionGrads = K.gradients(qvalTrain, action)[0]

        low = tf.convert_to_tensor(self.env.action_space.low)
        high = tf.convert_to_tensor(self.env.action_space.high)
        width = high - low
        pos = K.cast(K.greater_equal(self.criticActionGrads, 0), dtype='float32')
        pos *= high - action
        neg = K.cast(K.less(self.criticActionGrads, 0), dtype='float32')
        neg *= action - low
        inversion = (pos + neg) / width
        self.invertedCriticActionGrads = self.criticActionGrads * inversion

        # self.actorGrads = tf.gradients(action, self.actionModel.trainable_weights, grad_ys=-self.criticActionGrads)
        self.actorGrads = tf.gradients(action, self.actionModel.trainable_weights, grad_ys=-self.invertedCriticActionGrads)
        self.updatesActor = DDPGAdam(lr=0.0001).get_updates(params=self.actionModel.trainable_weights,
                                                        loss=None,
                                                        grads=self.actorGrads)
        self.trainActor = K.function(inputs=[S_a, G_a],
                                     outputs=[action, self.criticActionGrads, self.invertedCriticActionGrads],
                                     updates=self.updatesActor)

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

        L1 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))
        L1out = L1(concatenate([S, G]))
        L1out = concatenate([L1out, A])
        L2 = Dense(300, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))
        L2out = L2(L1out)
        L3 = Dense(1, activation='linear',
                   kernel_initializer=RandomUniform(minval=-3e-4, maxval=3e-4),
                   kernel_regularizer=l2(0.01),
                   bias_initializer=RandomUniform(minval=-3e-4, maxval=3e-4))
        qval = L3(L2out)
        return [L1, L2, L3], qval

    def create_actor_network(self, S, G=None):
        h0 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform())(concatenate([S, G]))
        h1 = Dense(300, activation="relu",
                   kernel_initializer=lecun_uniform())(h0)
        V = Dense(self.a_dim[0],
                  activation="tanh",
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                  bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))(h1)
        return V




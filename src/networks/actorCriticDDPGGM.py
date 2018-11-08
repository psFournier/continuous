from keras.initializers import RandomUniform, lecun_uniform
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from keras.optimizers import Adam
from .actorCriticDDPG import DDPGAdam
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers.merge import concatenate, multiply, add, subtract, maximum


class ActorCriticDDPGGM(object):
    def __init__(self, args, env):
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = env.action_dim
        self.g_dim = env.goal_dim
        self.gamma = 0.99
        self.w_i = float(args['--wimit'])
        self.margin = float(args['--margin'])
        self.inv_grads = args['--inv_grads']
        self.initModels()
        self.initTargetModels()

    def initModels(self):

        S_c = Input(shape=self.s_dim)
        A_c = Input(shape=self.a_dim)
        G_c = Input(shape=self.g_dim)
        M_c = Input(shape=self.g_dim)
        TARGETS = Input(shape=(1,))

        layers, qval = self.create_critic_network(S_c, A_c, G_c, M_c)
        self.qvalModel = Model([S_c, A_c, G_c, M_c], qval)
        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        inputs = [S_c, A_c, G_c, M_c, TARGETS]
        outputs = [loss_dqn, qval]
        self.updatesQval = Adam(lr=0.001).get_updates(params=self.qvalModel.trainable_weights, loss=loss_dqn)
        self.trainQval = K.function(inputs=inputs, outputs=outputs, updates=self.updatesQval)

        S_a = Input(shape=self.s_dim)
        G_a = Input(shape=self.g_dim)
        M_a = Input(shape=self.g_dim)
        action = self.create_actor_network(S_a, G_a, M_a)
        self.actionModel = Model([S_a, G_a, M_a], action)
        self.action = K.function(inputs=[S_a, G_a, M_a], outputs=[action], updates=None)

        L1, L2, L3 = layers
        qvalTrain = L1(concatenate([multiply([subtract([S_a, G_a]), M_a]), S_a]))
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

        if self.inv_grads == '0':
            self.actorGrads = tf.gradients(action, self.actionModel.trainable_weights, grad_ys=-self.criticActionGrads)
        else:
            self.actorGrads = tf.gradients(action, self.actionModel.trainable_weights, grad_ys=-self.invertedCriticActionGrads)

        self.updatesActor = DDPGAdam(lr=0.0001).get_updates(params=self.actionModel.trainable_weights,
                                                        loss=None,
                                                        grads=self.actorGrads)
        inputs = [S_a, G_a, M_a]
        outputs = []
        self.trainActor = K.function(inputs=inputs, outputs=outputs, updates=self.updatesActor)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=self.a_dim)
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        Tlayers, Tqval = self.create_critic_network(S, A, G, M)
        self.TqvalModel = Model([S, A, G, M], Tqval)
        self.Tqval = K.function(inputs=[S, A, G, M], outputs=[Tqval], updates=None)

        S = Input(shape=self.s_dim)
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)
        Taction = self.create_actor_network(S, G, M)
        self.TactionModel = Model([S, G, M], Taction)
        self.Taction = K.function(inputs=[S, G, M], outputs=[Taction], updates=None)

        self.target_train()

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

    def get_targets_dqn(self, r, t, s, g=None, m=None):

        a = self.Taction([s, g, m])[0]
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        q = self.Tqval([s, a, g, m])[0]
        targets = self.compute_targets(r, t, q)
        return np.expand_dims(targets, axis=1)

    def compute_targets(self, r, t, q):
        targets = r + (1 - t) * self.gamma * np.squeeze(q)
        targets = np.clip(targets, self.env.minQ, self.env.maxQ)
        return targets

    def create_critic_network(self, S, A, G=None, M=None):
        input = concatenate([multiply([subtract([S, G]), M]), S])
        L1 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform(),
                   kernel_regularizer=l2(0.01))
        L1out = L1(input)
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

    def create_actor_network(self, S, G=None, M=None):
        input = concatenate([multiply([subtract([S, G]), M]), S])
        h0 = Dense(400, activation="relu",
                   kernel_initializer=lecun_uniform())(input)
        h1 = Dense(300, activation="relu",
                   kernel_initializer=lecun_uniform())(h0)
        V = Dense(self.a_dim[0],
                  activation="tanh",
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                  bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))(h1)
        return V

    def train(self, inputsCritic, inputsActor):
        metrics = self.trainQval(inputsCritic)
        self.trainActor(inputsActor)
        return metrics



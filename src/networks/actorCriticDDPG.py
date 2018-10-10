from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input, Lambda, concatenate
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
import tensorflow as tf

class ActorCriticDDPG(object):
    def __init__(self, args, env):
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = env.action_dim
        self.gamma = 0.99
        self.optimizerCritic = Adam(lr=0.001)
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
        self.updatesQval = self.optimizerCritic.get_updates(params=self.qvalModel.trainable_weights, loss=loss_dqn)
        self.trainQval = K.function(inputs=[S, A, TARGETS], outputs=[loss_dqn], updates=self.updatesQval)

        action = self.create_actor_network(S)
        self.action = K.function(inputs=[S], outputs=[action], updates=None)
        L1, L2, L3 = layers
        qvalTrain = concatenate([L1(S), action])
        qvalTrain = L2(qvalTrain)
        qvalTrain = L3(qvalTrain)
        self.actionModel = Model(S, action)

        # Method 1 without gradient inversion
        self.optimizerActor = Adam(lr=0.0001)
        loss_actor = -K.mean(qvalTrain)
        self.updatesActor = self.optimizerActor.get_updates(params=self.actionModel.trainable_weights, loss=loss_actor)

        # Method 2 with gradient inversion
        # self.optimizerActor = DDPGAdam(lr=0.0001)
        # loss_actor = K.mean(qvalTrain)
        # self.updatesActor = self.optimizerActor.get_updates(params=self.actionModel.trainable_weights, loss=loss_actor,
        #                                                     action=action, low=self.env.action_space.low, high=self.env.action_space.high)

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

class DDPGAdam(Adam):
    def __init__(self, lr=0.001):
        super(DDPGAdam, self).__init__(lr=lr)

    def get_updates(self, loss, params, action=None, high=None, low=None):
        gradsC = K.gradients(loss, action)[0]
        low = tf.convert_to_tensor(low)
        high = tf.convert_to_tensor(high)
        width = high - low
        pos = K.cast(K.greater_equal(gradsC, 0), dtype='float32')
        pos *= high - action
        neg = K.cast(K.less(gradsC, 0), dtype='float32')
        neg *= action - low
        gradsC *= (pos + neg) / width
        grads = tf.gradients(action, params, grad_ys=gradsC)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
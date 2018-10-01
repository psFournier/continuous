from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

class CriticDQN(object):
    def __init__(self, args, env):
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = (1,)
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.num_actions = env.action_dim
        self.optimizer = Adam(lr=self.learning_rate)
        self.args = args
        self.initModels()
        self.initTargetModels()

    def initModels(self):

        w0, w1, w2 = float(self.args['--w0']), float(self.args['--w1']), float(self.args['--w2'])
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        T = Input(shape=(1,), dtype='float32')
        TARGETS = Input(shape=(1,))

        qvals = self.create_critic_network(S)
        self.model = Model([S], qvals)
        self.qvals = K.function(inputs=[S], outputs=[qvals], updates=None)

        actionProbs = K.softmax(qvals / T)
        self.actionProbs = K.function(inputs=[S, T], outputs=[actionProbs], updates=None)

        actionFilter = K.squeeze(K.one_hot(A, self.num_actions), axis=1)
        qval = K.sum(actionFilter * qvals, axis=1, keepdims=True)
        self.qval = K.function(inputs=[S, A], outputs=[qval], updates=None)

        val = K.max(qvals, axis=1, keepdims=True)
        self.val = K.function(inputs=[S], outputs=[val], updates=None)

        loss = 0
        loss_dqn = K.mean(K.square(qval - TARGETS), axis=0)
        loss += w0 * loss_dqn
        self.updatesDqn = self.optimizer.get_updates(params=self.model.trainable_weights, loss=loss_dqn)
        self.trainDqn = K.function(inputs=[S, A, TARGETS],
                                   outputs=[loss_dqn, qval],
                                   updates=self.updatesDqn)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        Tqvals = self.create_critic_network(S)
        self.Tmodel = Model([S], Tqvals)
        self.target_train()

    def get_targets_dqn(self, r, t, s):
        qvals = self.model.predict_on_batch([s])
        a1 = np.argmax(qvals, axis=1)
        Tqvals = self.Tmodel.predict_on_batch([s])
        q = Tqvals[a1]
        targets_dqn = self.compute_targets(r, t, q)
        return targets_dqn

    def compute_targets(self, r, t, q):
        targets = r + (1 - t) * self.gamma * np.squeeze(q)
        targets = np.clip(targets, self.env.minR / (1 - self.gamma), self.env.maxR / (1 - self.gamma))
        return targets

    def create_critic_network(self, S):
        l1 = Dense(400, activation="relu")(S)
        l2 = Dense(300, activation="relu")(l1)
        Q_values = Dense(self.num_actions)(l2)
        return Q_values

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.Tmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.Tmodel.set_weights(target_weights)

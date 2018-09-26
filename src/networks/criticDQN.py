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

    def actionFilterFn(self, inputs):
        filter = K.squeeze(K.one_hot(inputs[0], self.num_actions), axis=1)
        return K.sum(filter * inputs[1], axis=1, keepdims=True)

    def marginFn(self, inputs):
        a = inputs[0]
        v = inputs[1]
        q = inputs[2]
        adv = inputs[3]
        width = float(self.args['--margin']) * (K.max(v, axis=1, keepdims=True) - K.min(v, axis=1, keepdims=True))
        one_hot = 1 - K.squeeze(K.one_hot(a, self.num_actions), axis=1)
        width =  K.repeat_elements(width, self.num_actions, axis=1)
        margin = width * one_hot
        res = (K.max(v + margin, axis=1, keepdims=True) - q) * adv
        return res

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        T = Input(shape=(1,), dtype='float32')
        qvals = self.create_critic_network(S)
        actionProbs = Lambda(lambda x: K.softmax(x[0]/x[1]))([qvals, T])
        self.actionProbsModel = Model([S, T], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,), name='qval')([A, qvals])
        self.criticModel = Model([S, A], qval)
        self.criticModel.compile(loss='mse', optimizer=self.optimizer)
        self.criticModel.metrics_tensors = [qval]

        if self.args['--imit'] == '1':
            E = Input(shape=(1,), dtype='float32')
            actionProb = Lambda(self.actionFilterFn, output_shape=(1,))([A, actionProbs])
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(lambda x: -K.log(x[0]) * x[1], name='imit')([actionProb, advantage])
            self.imitModel = Model([S, A, E], [qval, imit, advantage])
            self.imitModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[float(self.args['--w0']), float(self.args['--w1']), float(self.args['--w2'])],
                                   optimizer=self.optimizer)

        if self.args['--imit'] == '2':
            E = Input(shape=(1,), dtype='float32')
            val = Lambda(lambda x: K.max(x, axis=1, keepdims=True))(qvals)
            advantage = Lambda(lambda x: K.maximum(x[0] - x[1], 0), name='advantage')([E, val])
            imit = Lambda(self.marginFn, output_shape=(1,), name='imit')([A, qvals, qval, advantage])
            self.imitModel = Model([S, A, E], [qval, imit, advantage])
            self.imitModel.compile(loss=['mse', 'mae', 'mse'],
                                   loss_weights=[float(self.args['--w0']), float(self.args['--w1']), float(self.args['--w2'])],
                                   optimizer=self.optimizer)

    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        targetQvals = self.create_critic_network(S)
        targetQval = Lambda(self.actionFilterFn, output_shape=(1,))([A, targetQvals])
        self.criticTmodel = Model([S, A], targetQval)
        self.target_train()

    def get_targets_dqn(self, r, t, s):
        temp = np.expand_dims([1], axis=0)
        a1Probs = self.actionProbsModel.predict_on_batch([s, temp])
        a1 = np.argmax(a1Probs, axis=1)
        q = self.criticTmodel.predict_on_batch([s, a1])
        targets_dqn = self.compute_targets(r, t, q)
        return targets_dqn

    def compute_targets(self, r, t, q):
        targets = r + (1 - t) * self.gamma * np.squeeze(q)
        targets = np.clip(targets, self.env.minR / (1 - self.gamma), self.env.maxR)
        return targets

    def create_critic_network(self, S):
        l1 = Dense(400, activation="relu")(S)
        l2 = Dense(300, activation="relu")(l1)
        Q_values = Dense(self.num_actions)(l2)
        return Q_values

    def target_train(self):
        weights = self.criticModel.get_weights()
        target_weights = self.criticTmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.criticTmodel.set_weights(target_weights)

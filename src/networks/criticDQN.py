from keras.models import Model
from keras.layers import Dense, Input, Lambda
from keras.optimizers import Adam
import keras.backend as K

class CriticDQN(object):
    def __init__(self, args, env):
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
        margin = self.args['--margin'] * 0.14 * (1 - K.one_hot(a, self.num_actions))
        return (K.max(v + margin) - q) * adv

    def initModels(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        T = Input(shape=(1,), dtype='float32')
        qvals = self.create_critic_network(S)
        actionProbs = Lambda(lambda x: K.softmax(x[0]/x[1]))([qvals, T])
        self.actionProbsModel = Model([S, T], actionProbs)
        qval = Lambda(self.actionFilterFn, output_shape=(1,), name='qval')([A, qvals])
        self.qvalModel = Model([S, A], qval)
        self.qvalModel.compile(loss='mse', optimizer=self.optimizer)
        self.qvalModel.metrics_tensors = [qval]

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
        self.qvalTModel = Model([S, A], targetQval)
        self.target_train()

    def create_critic_network(self, S):
        l1 = Dense(400, activation="relu")(S)
        l2 = Dense(300, activation="relu")(l1)
        Q_values = Dense(self.num_actions)(l2)
        return Q_values

    def target_train(self):
        weights = self.qvalModel.get_weights()
        target_weights = self.qvalTModel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.qvalTModel.set_weights(target_weights)

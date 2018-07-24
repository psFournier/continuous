from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.merge import concatenate, multiply
from keras.losses import mse

class CriticDQN(object):
    def __init__(self, sess, s_dim, g_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        self.sess = sess
        self.tau = tau
        self.s_dim = s_dim
        self.g_dim = g_dim
        self.a_dim = (1,)
        self.learning_rate = learning_rate
        self.stat_ops = []
        self.stat_names = []
        self.model = None
        self.target_model = None
        self.td_errors = None
        self.gamma = gamma
        self.num_actions = num_a

        K.set_session(sess)

        self.qValue, self.bestAction, self.states = self.create_critic_network()
        self.target_qValue, self.target_bestAction, self.target_state = self.create_critic_network()

    def target_train(self):
        weights = self.qValue.get_weights()
        target_weights = self.target_qValue.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_qValue.set_weights(target_weights)

    def create_critic_network(self):

        S = Input(shape=self.s_dim)
        G = Input(shape=self.g_dim)
        A = Input(shape=(1,), dtype='uint8')

        h = concatenate(inputs=[S,G])
        h = Dense(50, activation="relu")(h)
        V = Dense(self.num_actions,
                  activation='linear',
                  use_bias=False,
                  kernel_initializer='random_uniform',
                  name='dense_0')(h)
        V = Reshape((1, self.num_actions))(V)
        mask = Lambda(K.one_hot,
                      arguments={'num_classes': self.num_actions},
                      output_shape=(self.num_actions,))(A)
        filteredV = multiply([V, mask])
        out1 = Lambda(K.sum,
                      arguments={'axis': 2})(filteredV)
        out2 = Lambda(K.argmax,
                      arguments={'axis': 2})(V)
        qValue = Model(inputs=[S, A, G], outputs=out1)
        bestAction = Model(inputs=[S, G], outputs=out2)
        optimizer = Adam(lr=self.learning_rate)
        qValue.compile(loss='mse', optimizer=optimizer)
        qValue.metrics_tensors = [qValue.targets[0] - qValue.outputs[0]]
        return qValue, bestAction, S




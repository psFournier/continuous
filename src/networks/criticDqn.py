from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.merge import concatenate, multiply
from keras.losses import mse

class CriticDQN(object):
    def __init__(self, sess, s_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        self.sess = sess
        self.tau = tau
        self.s_dim = s_dim
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

        self.model1, self.model2, self.states = self.create_critic_network()
        self.target_model1, self.target_model2, self.target_state = self.create_critic_network()
        # input_tensors = [self.model.inputs[0],  # input data
        #                  self.model.sample_weights[0],  # how much to weight each sample by
        #                  self.model.targets[0],  # labels
        #                  K.learning_phase(),  # train or test mode
        #                  ]
        # self.get_gradients = K.function(inputs=input_tensors, outputs=self.grads)

    def target_train(self):
        weights = self.model1.get_weights()
        target_weights = self.target_model1.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model1.set_weights(target_weights)

    def create_critic_network(self):
        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        h = Dense(50, activation="relu")(S)
        V = Dense(self.num_actions,
                  activation='linear',
                  use_bias=False,
                  kernel_initializer='random_uniform',
                  name='dense_0')(h)
        V = Reshape((1, self.num_actions))(V)
        mask = Lambda(K.one_hot,
                      arguments={'num_classes': self.num_actions},
                      output_shape=(self.num_actions,))(A)
        filteredV = multiply([V, mask], )
        out1 = Lambda(K.sum,
                      arguments={'axis': 2})(filteredV)
        out2 = Lambda(K.argmax,
                      arguments={'axis': 2})(V)
        model1 = Model(inputs=[S,A], outputs=out1)
        model2 = Model(inputs=[S], outputs=out2)
        optimizer = Adam(lr=self.learning_rate)
        model1.compile(loss='mse', optimizer=optimizer)
        return model1, model2, S



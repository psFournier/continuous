from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract
from keras.losses import mse
from keras.utils import plot_model

def margin_fn(indices, num_classes):
    return 0.8 * (1 - K.one_hot(indices, num_classes))

class CriticDQNfD(object):
    def __init__(self, sess, s_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001, lambda1=1, lambda2=1):
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
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        K.set_session(sess)

        self.optimizer_ql = Adam(lr=self.learning_rate)
        self.optimizer_lm = Adam(lr=self.learning_rate)

        self.model_ql, self.model_lm, self.model_act, self.states = self.create_critic_network()
        plot_model(self.model_ql, to_file='model_ql.png')

        self.target_model_ql, self.target_model_lm, self.target_model_act, self.target_state = self.create_critic_network()

    def target_train(self):
        weights = self.model_ql.get_weights()
        target_weights = self.target_model_ql.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model_ql.set_weights(target_weights)

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

        out_act = Lambda(K.argmax,
                      arguments={'axis': 2})(V)

        mask = Lambda(K.one_hot,
                      arguments={'num_classes': self.num_actions},
                      output_shape=(self.num_actions,))(A)
        filteredV = multiply([V, mask])
        out_qLearning = Lambda(K.sum,
                      arguments={'axis': 2})(filteredV)

        margin = Lambda(margin_fn,
                        arguments={'num_classes': self.num_actions},
                        output_shape=(self.num_actions,))(A)
        Vsum = add([V, margin])
        Vmax = Lambda(K.max,
                     arguments={'axis': 2})(Vsum)
        out_largeMargin = subtract([Vmax, out_qLearning])

        model_ql = Model(inputs=[S,A], outputs=out_qLearning)
        model_ql.compile(loss='mse', optimizer=self.optimizer_ql)
        model_ql.metrics_tensors = [model_ql.targets[0] - model_ql.outputs[0]]

        model_lm = Model(inputs=[S,A], outputs=out_largeMargin)
        model_lm.compile(loss='mae', optimizer=self.optimizer_lm)

        model_act = Model(inputs=[S], outputs=out_act)

        return model_ql, model_lm, model_act, S




from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.merge import concatenate
from keras.losses import mse

class CriticDQN(object):
    def __init__(self, sess, s_dim, a_dim, gamma=0.99, tau=0.001, learning_rate=0.001):
        self.sess = sess
        self.tau = tau
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = learning_rate
        self.stat_ops = []
        self.stat_names = []
        self.model = None
        self.target_model = None
        self.td_errors = None
        self.gamma = gamma

        K.set_session(sess)

        self.model, self.states, self.grads = self.create_critic_network()
        self.t_model, self.t_states, t_grads = self.create_critic_network()
        input_tensors = [self.model.inputs[0],  # input data
                         self.model.sample_weights[0],  # how much to weight each sample by
                         self.model.targets[0],  # labels
                         K.learning_phase(),  # train or test mode
                         ]
        self.get_gradients = K.function(inputs=input_tensors, outputs=self.grads)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model.set_weights(target_weights)

    def create_critic_network(self):
        S = Input(shape=self.s_dim)
        # h = Dense(50, activation="relu")(S)
        V = Dense(self.a_dim[0],
                  activation='linear',
                  use_bias=False,
                  kernel_initializer='random_uniform',
                  name='dense_0')(S)
        model = Model(inputs=[S], outputs=V)
        optimizer = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer)
        gradients = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
        return model, S, gradients



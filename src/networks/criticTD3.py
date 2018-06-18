from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf
from keras.layers.merge import concatenate
import keras.backend as K

class CriticTD3(object):
    def __init__(self, sess, s_dim, a_dim, gamma=0.99, tau=0.005, learning_rate=0.001):
        self.sess = sess
        self.tau = tau
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = learning_rate
        self.stat_ops = []
        self.stat_names = []
        self.model1 = None
        self.target_model1 = None
        self.model2 = None
        self.target_model2 = None

        self.gamma = gamma

        # Now create the model
        self.model1, self.action1, self.state1 = self.create_critic_network(self.s_dim, self.a_dim)
        self.target_model1, self.target_action1, self.target_state1 = self.create_critic_network(self.s_dim, self.a_dim)
        self.model2, self.action2, self.state2 = self.create_critic_network(self.s_dim, self.a_dim)
        self.target_model2, self.target_action2, self.target_state2 = self.create_critic_network(self.s_dim, self.a_dim)

        self.out1 = self.model1.output
        self.action_grads1 = tf.gradients(self.out1, self.action1)

        self.targets = K.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")
        self.imp_weights = K.placeholder(dtype=tf.float32, shape=(None, 1), name="weights")
        self.td_errors1 = self.out1 - self.targets
        self.weighted_error = K.mean(K.square(self.td_errors1), axis=-1)
        self.optimizer = Adam()
        self.updates = Adam().get_updates(self.model1.trainable_weights, [], self.weighted_error)
        self.train = K.function([self.state1, self.action1, self.targets, self.imp_weights],
                                [self.td_errors1, self.action_grads1],
                                updates=self.updates)

    def target_train(self):

        weights1 = self.model1.get_weights()
        target_weights1 = self.target_model1.get_weights()
        for i in range(len(weights1)):
            target_weights1[i] = self.tau * weights1[i] + (1 - self.tau)* target_weights1[i]
        self.target_model1.set_weights(target_weights1)

        weights2 = self.model2.get_weights()
        target_weights2 = self.target_model2.get_weights()
        for i in range(len(weights2)):
            target_weights2[i] = self.tau * weights2[i] + (1 - self.tau) * target_weights2[i]
        self.target_model2.set_weights(target_weights2)

    def create_critic_network(self, s_dim, a_dim):
        S = Input(shape=s_dim)
        A = Input(shape=a_dim, name='action2')
        input = concatenate([S,A])
        l1 = Dense(400, activation="relu", kernel_initializer="he_uniform")(input)
        l2 = concatenate([l1, A])
        l3 = Dense(300, activation="relu", kernel_initializer="he_uniform")(l2)
        V = Dense(1, activation='linear',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(l3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S


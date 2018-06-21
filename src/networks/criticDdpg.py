from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.merge import concatenate

class CriticDDPG(object):
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

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(self.s_dim, self.a_dim)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(self.s_dim, self.a_dim)
        self.out = self.model.output

        self.action_grads = tf.gradients(self.out, self.action)
        self.targets = K.placeholder(dtype=tf.float32, shape=(None,1), name="targets")
        self.imp_weights = K.placeholder(dtype=tf.float32, shape=(None,1), name="weights")
        self.td_errors = self.out - self.targets
        self.weighted_error = K.mean(K.square(self.td_errors), axis=-1)
        self.updates = Adam(lr=self.learning_rate).get_updates(self.model.trainable_weights, [], self.weighted_error)
        self.train = K.function([self.state, self.action, self.targets, self.imp_weights], [self.td_errors], updates=self.updates)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model.set_weights(target_weights)

    def hard_target_train(self):
        self.target_model.set_weights(self.model.get_weights())

    def gradients(self, states, actions):
        out, grads =  self.sess.run([self.out, self.action_grads], feed_dict={
            self.state: states,
            self.action: actions
        })
        return out, grads[0]

    def create_critic_network(self, state_size, action_dim):
        S = Input(shape=state_size)
        A = Input(shape=action_dim, name='action2')
        w = Dense(400, activation="relu", kernel_initializer="he_uniform")(S)
        h = concatenate([w, A])
        h3 = Dense(300, activation="relu", kernel_initializer="he_uniform")(h)
        V = Dense(1, activation='linear',
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(h3)
        model = Model(inputs=[S, A], outputs=V)
        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        return model, A, S



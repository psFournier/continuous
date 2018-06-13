from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.merge import concatenate
from ddpg.util import reduce_std

class CriticDDPG(object):
    def __init__(self, sess, state_size, action_size, gamma=0.99, tau=0.005, learning_rate=0.001):
        self.sess = sess
        self.tau = tau
        self.s_dim = state_size
        self.a_dim = action_size
        self.learning_rate = learning_rate
        self.stat_ops = []
        self.stat_names = []
        K.set_session(sess)
        self.model = None
        self.target_model = None

        self.gamma = gamma

        # Now create the model
        self.model, self.action, self.state = self.create_critic_network(self.s_dim, self.a_dim)
        self.target_model, self.target_action, self.target_state = self.create_critic_network(self.s_dim, self.a_dim)
        self.out = self.model.output
        self.action_grads = tf.gradients(self.out, self.action)

        # Setting up stats
        self.stat_ops += [tf.reduce_mean(self.out)]
        self.stat_names += ['mean_Q_values']
        self.stat_ops += [tf.reduce_mean(self.action_grads)]
        self.stat_names += ['mean_action_grads']

        self.stat_ops += [reduce_std(self.out)]
        self.stat_names += ['std_Q_values']
        self.stat_ops += [reduce_std(self.action_grads)]
        self.stat_names += ['std_action_grads']

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model.set_weights(target_weights)

    def gradients(self, states, actions):
        out, grads =  self.sess.run([self.out, self.action_grads], feed_dict={
            self.state: states,
            self.action: actions
        })
        return out, grads[0]

    def predict_target(self, states, actions):
        return self.target_model.predict_on_batch([states, actions])

    def predict(self, states, actions):
        return self.model.predict_on_batch([states, actions])

    def train(self, states, actions, targets):
        self.model.train_on_batch([states, actions], targets)
        critic_stats = self.sess.run(self.stat_ops, feed_dict={
            self.state: states,
            self.action: actions})
        return critic_stats

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

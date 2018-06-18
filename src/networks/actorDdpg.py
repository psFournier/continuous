from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input
import tensorflow as tf
import keras.backend as K

class ActorDDPG(object):
    def __init__(self, sess, s_dim, a_dim, tau=0.001, learning_rate=0.0001):
        self.sess = sess
        self.tau = tau
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learning_rate = learning_rate
        self.stat_ops = []
        self.stat_names = []
        K.set_session(sess)
        self.model = None
        self.target_model = None

        self.model, self.weights, self.state = self.create_actor_network(self.s_dim, self.a_dim)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(self.s_dim, self.a_dim)
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim[0]])
        self.out = self.model.output
        self.params_grad = tf.gradients(self.out, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.target_model.set_weights(target_weights)

    def create_actor_network(self, state_size, action_dim):
        S = Input(shape=state_size)
        h0 = Dense(400, activation="relu", kernel_initializer="he_uniform")(S)
        h1 = Dense(300, activation="relu", kernel_initializer="he_uniform")(h0)
        V = Dense(action_dim[0], activation="tanh",
                  kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3, seed=None))(h1)
        model = Model(inputs=S,outputs=V)
        return model, model.trainable_weights, S

    def train(self, states, action_grads):
        res = self.sess.run([self.optimize]+self.stat_ops, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

        return res[1:]



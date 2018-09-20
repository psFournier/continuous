from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input
import tensorflow as tf
import keras.backend as K
from keras.optimizers import Adam

class ActorDDPG(object):
    def __init__(self, args, env):
        self.env = env
        self.tau = 0.001
        self.s_dim = env.state_dim
        self.a_dim = (1,)
        self.learning_rate = 0.0001
        self.optimizer = Adam(lr=self.learning_rate)
        self.args = args
        self.initModels()
        self.initTargetModels()

    def initModels(self):
        S = Input(shape=self.s_dim)
        GRAD = Input(shape=self.a_dim)
        act = self.create_actor_network(S)
        self.model = Model(S, act)
        # loss =
        updates = self.optimizer.get_updates(params=self.model.trainable_weights,
                                   constraints=[],
                                   loss=loss)
        # self.train = K.function(inputs=[S,
        #                                    action_onehot_placeholder,
        #                                    discount_reward_placeholder],
        #                            outputs=[],
        #                            updates=updates)


    def initTargetModels(self):
        S = Input(shape=self.s_dim)
        Tact = self.create_actor_network(S)
        self.Tmodel = Model(S, Tact)
        self.target_train()

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.Tmodel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau) * target_weights[i]
        self.Tmodel.set_weights(target_weights)

        self.model, self.weights, self.state = self.create_actor_network(self.s_dim, self.a_dim)
        self.target_model, self.target_weights, self.target_state = self.create_actor_network(self.s_dim, self.a_dim)
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim[0]])
        self.out = self.model.output
        self.params_grad = tf.gradients(self.out, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

    def create_actor_network(self, S):
        h0 = Dense(400, activation="relu")(S)
        h1 = Dense(300, activation="relu")(h0)
        V = Dense(self.a_dim, activation="tanh")(h1)
        return V

    def train(self, states, action_grads):
        res = self.sess.run([self.optimize]+self.stat_ops, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

        return res[1:]



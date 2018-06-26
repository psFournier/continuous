from keras.initializers import RandomUniform
from keras.models import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
import tensorflow as tf
import keras.backend as K
from keras.layers.merge import concatenate
from keras.losses import mse

class CriticDQN(object):
    def __init__(self, sess, s_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        self.sess = sess
        self.tau = tau
        self.s_dim = s_dim
        self.a_dim = (1,)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_actions = num_a

        K.set_session(sess)

        self.model, s, q = self.build_model()
        self.modelt, st, qt = self.build_model()

        a = Input(shape=self.a_dim, dtype='uint8')
        qselect = K.sum(q * K.one_hot(a, self.num_actions), axis=1)
        # qmax = K.max(q, axis=1)
        # self.qselect = K.function([s, a], [qselect], updates=None)
        target = K.placeholder(name='target', shape=(None, 1))
        td_errors = qselect - target
        loss = K.mean(K.square(td_errors), axis=-1)
        opt = Adam(lr=self.learning_rate)
        updates = opt.get_updates(self.model.trainable_weights, [], loss)
        self.train = K.function([s, a, target], [td_errors, loss], updates=updates)

        qtmax = K.max(qt, axis=1)
        self.qtmax = K.function([st], [qtmax], updates=None)

    def build_model(self):
        s = Input(shape=self.s_dim)
        h = Dense(self.num_actions,
                  activation='linear',
                  use_bias=False,
                  kernel_initializer='random_uniform',
                  name='dense_0')
        q = h(s)
        model = Model(inputs=[s], outputs=[q])
        return model, s, q

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.modelt.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.modelt.set_weights(target_weights)




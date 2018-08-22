from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQN import  CriticDQN

def margin_fn(indices, num_classes):
    return 0.8 * (1 - K.one_hot(indices, num_classes))

class CriticDQNG(CriticDQN):
    def __init__(self, s_dim, g_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        self.g_dim = g_dim
        super(CriticDQNG, self).__init__(s_dim, num_a, gamma, tau, learning_rate)

    def create_critic_network(self):

        self.optimizer = Adam(lr=self.learning_rate)

        S = Input(shape=self.s_dim)
        G = Input(shape=self.g_dim)
        A = Input(shape=(1,), dtype='uint8')
        R = Input(shape=(1,), dtype='float32')
        Z = Input(shape=(1,), dtype='float32')

        c1 = concatenate(inputs=[S,G])
        l1 = Dense(400, activation="relu")(c1)
        c2 = concatenate([l1,G])
        l2 = Dense(300, activation="relu")(c2)

        V = Dense(self.num_actions,
                  activation=None,
                  kernel_initializer='random_uniform')(l2)
        V = Reshape((1, self.num_actions))(V)

        mask = Lambda(K.one_hot,
                      arguments={'num_classes': self.num_actions},
                      output_shape=(self.num_actions,))(A)
        filteredV = multiply([V, mask])
        qValue = Lambda(K.sum,
                      arguments={'axis': 2})(filteredV)
        qValue_model = Model(inputs=[S, A, G], outputs=qValue)
        qValue_model.compile(loss='mse', optimizer=self.optimizer)
        qValue_model.metrics_tensors = [qValue_model.outputs[0]]

        bestAction = Lambda(K.argmax,
                      arguments={'axis': 2})(V)
        bestAction_model = Model(inputs=[S, G], outputs=bestAction)

        margin = Lambda(margin_fn,
                        arguments={'num_classes': self.num_actions},
                        output_shape=(self.num_actions,))(A)
        Vsum = add([V, margin])
        Vmax = Lambda(K.max,
                      arguments={'axis': 2})(Vsum)
        margin = subtract([Vmax, qValue])
        margin_model = Model(inputs=[S, A, G], outputs=margin)
        margin_model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))

        # advantage = subtract([R, qValue])
        # mask2 = maximum([advantage, Z])
        # margin_masked = multiply([margin, mask2])
        # margin_masked_model = Model(inputs=[S, A, G, R, Z], outputs=margin_masked)
        # margin_masked_model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))

        return qValue_model, margin_model, bestAction_model




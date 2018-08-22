from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum
from .criticDQNG import CriticDQNG

def margin_fn(indices, num_classes):
    return 0.8 * (1 - K.one_hot(indices, num_classes))

class CriticDQNGM(CriticDQNG):
    def __init__(self, s_dim, g_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        super(CriticDQNGM, self).__init__(s_dim, g_dim, num_a, gamma, tau, learning_rate)

    def create_critic_network(self):

        self.optimizer = Adam(lr=self.learning_rate)

        S = Input(shape=self.s_dim)
        A = Input(shape=self.a_dim, dtype='uint8')
        G = Input(shape=self.g_dim)
        M = Input(shape=self.g_dim)

        c1 = concatenate(inputs=[S,G,M])
        l1 = Dense(400, activation="relu")(c1)
        c2 = concatenate(inputs=[l1, G, M])
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
        qValue_model = Model(inputs=[S, A, G, M], outputs=qValue)
        qValue_model.compile(loss='mse', optimizer=self.optimizer)
        qValue_model.metrics_tensors = [qValue_model.outputs[0]]

        bestAction = Lambda(K.argmax,
                      arguments={'axis': 2})(V)
        bestAction_model = Model(inputs=[S, G, M], outputs=bestAction)

        margin = Lambda(margin_fn,
                        arguments={'num_classes': self.num_actions},
                        output_shape=(self.num_actions,))(A)
        Vsum = add([V, margin])
        Vmax = Lambda(K.max,
                      arguments={'axis': 2})(Vsum)
        margin = subtract([Vmax, qValue])
        margin_model = Model(inputs=[S, A, G, M], outputs=margin)
        margin_model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))

        return qValue_model, margin_model, bestAction_model




from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import multiply, add, subtract

def margin_fn(indices, num_classes):
    return 0.8 * (1 - K.one_hot(indices, num_classes))

class CriticDQN(object):
    def __init__(self, s_dim, num_a, gamma=0.99, tau=0.001, learning_rate=0.001):
        self.tau = tau
        self.s_dim = s_dim
        self.a_dim = (1,)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.num_actions = num_a
        self.qvalModel, self.marginModel, self.actModel = self.create_critic_network()
        self.qvalTModel, _, _ = self.create_critic_network()
        self.target_train()

    def target_train(self):
        weights = self.qvalModel.get_weights()
        target_weights = self.qvalTModel.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau * weights[i] + (1 - self.tau)* target_weights[i]
        self.qvalTModel.set_weights(target_weights)

    def create_critic_network(self):

        self.optimizer = Adam(lr=self.learning_rate)

        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')

        l1 = Dense(400, activation="relu")(S)
        l2 = Dense(300, activation="relu")(l1)
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
        qValue_model = Model(inputs=[S, A], outputs=qValue)
        qValue_model.compile(loss='mse', optimizer=self.optimizer)
        qValue_model.metrics_tensors = [qValue_model.targets[0] - qValue_model.outputs[0]]

        bestAction = Lambda(K.argmax,
                      arguments={'axis': 2})(V)
        bestAction_model = Model(inputs=[S], outputs=bestAction)

        margin = Lambda(margin_fn,
                        arguments={'num_classes': self.num_actions},
                        output_shape=(self.num_actions,))(A)
        Vsum = add([V, margin])
        Vmax = Lambda(K.max,
                      arguments={'axis': 2})(Vsum)
        margin = subtract([Vmax, qValue])
        margin_model = Model(inputs=[S, A], outputs=margin)
        margin_model.compile(loss='mae', optimizer=Adam(lr=self.learning_rate))

        return qValue_model, margin_model, bestAction_model



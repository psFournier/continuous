from keras.models import Model
from keras.layers import Dense, Input, Lambda, Reshape
from keras.optimizers import Adam
import keras.backend as K
from keras.layers.merge import concatenate, multiply, add, subtract, maximum

def margin_fn(indices, num_classes):
    return 0.8 * (1 - K.one_hot(indices, num_classes))

def maxplus(L):
    return K.maximum(L, 0)

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

    def marginFn(self, inputs):
        a = inputs[0]
        v = inputs[1]
        q = inputs[2]
        margin = 0.8 * (1 - K.one_hot(a, self.num_actions))
        return K.max(v + margin) - q

    def qvalFn(self, inputs):
        a = inputs[0]
        v = inputs[1]
        return (K.sum(K.one_hot(a, self.num_actions) * v, axis=2))

    def imitFn(self, inputs):
        e = inputs[0]
        q = inputs[1]
        m = inputs[2]
        return m * K.maximum(e - q, 0)

    def create_critic_network(self):

        self.optimizer = Adam(lr=self.learning_rate)

        S = Input(shape=self.s_dim)
        A = Input(shape=(1,), dtype='uint8')
        E = Input(shape=(1,), dtype='float32')

        l1 = Dense(400, activation="relu")(S)
        l2 = Dense(300, activation="relu")(l1)
        V = Dense(self.num_actions,
                  activation=None,
                  kernel_initializer='random_uniform')(l2)
        V = Reshape((1, self.num_actions))(V)

        qValue = Lambda(self.qvalFn, output_shape=(1,))([A, V])
        bestAction = Lambda(lambda x: K.argmax(x, axis=2))(V)
        margin = Lambda(self.marginFn, output_shape=(1,))([A, V, qValue])
        imit = Lambda(self.imitFn, output_shape=(1,))([E, qValue, margin])

        imitation_model = Model(inputs=[S, A, E], outputs=imit)
        imitation_model.compile(loss='mae', optimizer=self.optimizer)
        imitation_model.metrics_tensors = [margin]

        qValue_model = Model(inputs=[S, A], outputs=qValue)
        qValue_model.compile(loss='mse', optimizer=self.optimizer)
        qValue_model.metrics_tensors = [qValue]

        bestAction_model = Model(inputs=[S], outputs=bestAction)

        return qValue_model, imitation_model, bestAction_model


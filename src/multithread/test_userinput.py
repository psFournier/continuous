from threading import Thread
import time
from keras.models import Model
from keras.layers import Dense, Input, Reshape, Lambda, multiply
from keras.optimizers import Adam
from buffers import ReplayBuffer
import tensorflow as tf
import numpy as np
from collections import deque
import keras.backend as K

class Agent():
    def __init__(self, sess, s_dim, num_actions, lr):
        self.sess = sess
        self.step = 0
        self.epStep = 0
        self.run_thread = Thread(target=self.run)
        self.tutor_thread = Thread(target=self.input)
        self.sDim = s_dim
        self.num_actions = num_actions
        self.learning_rate = lr
        self.names = ['state0', 'action', 'feedback', 'weight']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.batchSize = 64
        self.episode = deque(maxlen=400)

    def create_model(self):
        K.set_session(self.sess)
        state = Input(shape=self.sDim)
        action = Input(shape=(1,), dtype='uint8')
        l1 = Dense(400, activation="relu")(state)
        feedback = Dense(self.num_actions, activation=None, kernel_initializer='random_uniform')(l1)
        feedback = Reshape((1, self.num_actions))(feedback)
        mask = Lambda(K.one_hot, arguments={'num_classes': self.num_actions},
                      output_shape=(self.num_actions,))(action)
        feedback = multiply([feedback, mask])
        feedback = Lambda(K.sum, arguments={'axis': 2})(feedback)
        feedbackModel = Model(inputs=[state, action], outputs=feedback)
        feedbackModel.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return feedbackModel

    def train(self):
        if self.buffer.nb_entries > self.batchSize:
            samples = self.buffer.sample(self.batchSize)
            s, a, targets, weights = [np.array(samples[name]) for name in self.names]
            self.loss = self.model.train_on_batch(x=[s,a], y=targets, sample_weight=weights)

    def start(self):
        self.run_thread.start()
        self.tutor_thread.start()

    # def init_variables(self):
    #     variables = tf.global_variables()
    #     uninitialized_variables = []
    #     for v in variables:
    #         if not hasattr(v, '_keras_initialized') or not v._keras_initialized:
    #             uninitialized_variables.append(v)
    #             v._keras_initialized = True
    #     self.sess.run(tf.variables_initializer(uninitialized_variables))

    def run(self):
        self.model = self.create_model()
        # self.init_variables()
        state0 = np.random.randint(0, 4, size=(5,))
        while self.step < 100000:
            action = np.random.randint(self.num_actions)
            state1 = np.random.randint(0, 4, size=(5,))
            self.step += 1
            self.epStep += 1
            experience = {'state0': state0, 'action': action, 'weight': 0, 'feedback': np.random.choice([1,-1])}
            self.episode.append(experience)
            self.train()
            state0 = state1
            if self.epStep >= 200:
                for s in range(self.epStep):
                    exp = self.episode.popleft()
                    if exp['weight'] != 0:
                        self.buffer.append(exp)
                self.epStep = 0
                state0 = np.random.randint(0, 4, size=(5,))
            if self.step % 200 == 0:
                print(self.step)

    def input(self):
        while True:
            if input() == '+':
                inputStep = self.step
                time.sleep(2)
                print('input +1, step = ', inputStep)
            elif input() == '-':
                inputStep = self.step
                time.sleep(2)
                print('input -1, step = ', inputStep)
            else:
                print('wrong input')

if __name__ == '__main__':
    s_dim = (5,)
    num_actions = 5
    lr = 0.01
    with tf.Session() as sess:
        agent = Agent(sess, s_dim, num_actions, lr)
        agent.start()
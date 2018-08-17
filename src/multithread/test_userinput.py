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
import sys

class Agent():
    def __init__(self, s_dim, num_actions, lr):
        self.step = 0
        self.epStep = 0
        self.ep = 0
        self.tutorListened = True
        self.tutorInput = ''
        self.sDim = s_dim
        self.num_actions = num_actions
        self.learning_rate = lr
        self.names = ['state0', 'action', 'feedback', 'fWeight']
        self.buffer = ReplayBuffer(limit=int(1e6), names=self.names)
        self.batchSize = 64
        self.episode = deque(maxlen=400)
        self.model = self.create_model()

    def create_model(self):
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
        loss = 0
        if self.buffer.nb_entries > self.batchSize:
            samples = self.buffer.sample(self.batchSize)
            s, a, targets, weights = [np.array(samples[name]) for name in self.names]
            loss = self.model.train_on_batch(x=[s,a], y=targets, sample_weight=weights)
        return loss

    def tutorListener(self):
        self.tutorInput = input("> ")
        print("maybe updating...the kbdInput variable is: {}".format(self.tutorInput))
        self.tutorListened = True

    def run(self):
        state0 = np.random.randint(0, 4, size=(5,))
        while self.step < 100000:

            if self.tutorInput != '':
                print("Received new keyboard Input. Setting playing ID to keyboard input value")
                for i in range(1,10):
                    self.episode[-i]['fWeight'] = 1
                    self.episode[-i]['feedback'] = self.tutorInput
                self.tutorInput = ''
            else:
                action = np.random.randint(self.num_actions)
                state1 = np.random.randint(0, 4, size=(5,))
                self.step += 1
                self.epStep += 1
                experience = {'state0': state0, 'action': action, 'fWeight': 0}
                self.episode.append(experience)
                self.loss = self.train()
                state0 = state1
                time.sleep(0.001)

            if self.tutorListened:
                self.tutorListened = False
                self.listener = Thread(target=self.tutorListener)
                self.listener.start()

            if self.epStep >= 200:
                if self.ep > 0:
                    for s in range(self.epStep):
                        exp = self.episode.popleft()
                        if exp['fWeight'] != 0:
                            self.buffer.append(exp)
                self.epStep = 0
                self.ep += 1
                state0 = np.random.randint(0, 4, size=(5,))
            if self.step % 1000 == 0:
                print(self.step, self.loss)

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
        agent = Agent(s_dim, num_actions, lr)
        agent.run()
from numpy.core.numeric import ndarray
from scipy.misc.pilutil import imresize

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist

import gym
import scipy.misc

C = 10000
N = 1000000

def initNet():
    model = Sequential()

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', input_shape=(20, 20, 32)))
    model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(9, 9, 64)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(18, activation='linear', input_shape=(512,)))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    # model.fit(X_train, Y_train, batch_size=32, verbose=1)
    return model

def QFunc(model, phi, action):
    
    pass

def preprocess(prev, curr):
    def step1(prev, curr):
        maxObservation = ndarray(prev.shape)
        for i in range(prev.shape[0]):
            for j in range(prev.shape[1]):
                rgb = prev[i][j]
                maxObservation[i][j] = (
                max(prev[i][j][0], curr[i][j][0]), max(prev[i][j][1], curr[i][j][1]),
                max(prev[i][j][2], curr[i][j][2]))
        return maxObservation

    def getYChannel(observation):
        yData = ndarray((210, 160))
        for i in range(observation.shape[0]):
            for j in range(observation.shape[1]):
                rgb = observation[i][j]
                yData[i][j] = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return yData

    def step2(img):
        return imresize(img, (84, 84))

    maxObservation = step1(prev, curr)
    yChannel = getYChannel(maxObservation)
    return step2(yChannel)

def DQN():
    Ccounter = 0
    env = gym.make('Riverraid-v0')
    Q = initNet()
    QHat = initNet()
    average=0
    num_episodes=1
    D = []
    for i_episode in range(num_episodes):
        total_reward = 0
        observation = env.reset()
        prevObservation = observation
        for t in range(1000000):
            env.render()
            # TODO: Perform epsilon greedy selection of actions
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            # Preprocess state to phi
            # TODO: preprocess needs to take a list of 8 frames but can be assumed for now
            phi = preprocess(prevObservation, observation)
            D.append(phi)
            if len(D) == N:
                D.pop(0)

            total_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                #print observation, reward, done, info
                average += total_reward
                print "episode reward ={}".format(total_reward)
                print "observation = {}".format(observation.shape)
                break

            # Minibatch gradient descent
                # 9. Fit model on training data
                # model.fit(X_train, Y_train, batch_size=32, nb_epoch=10, verbose=1)
                # model.fit(X_train, Y_train, batch_size=32, verbose=1)

            # updating QHat to Q after C steps
                Ccounter += 1
                if Ccounter == C:
                    weights = Q.get_weights()
                    QHat.set_weights(weights)
    print "average reward={}".format(average/num_episodes)

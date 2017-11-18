import numpy
from numpy import random
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
k = 4
epsilon = 0.1  # TODO: linearly variable with t


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


def preprocess(recentObservations):

    def getMaxBetweenTwo(ob1, ob2):
        maxObservation = ndarray(ob1.shape)
        for i in range(ob1.shape[0]):
            for j in range(ob1.shape[1]):
                rgb1 = ob1[i][j]
                rgb2 = ob2[i][j]
                maxObservation[i][j] = (
                    max(rgb1[0], rgb2[0]),
                    max(rgb1[1], rgb2[1]),
                    max(rgb1[2], rgb2[2]))
        return maxObservation

    def step1():
        maxObservations = []
        for i in range(k):
            maxObservations.append(getMaxBetweenTwo(recentObservations[i], recentObservations[i+1]))
        return maxObservations


    def getYChannelForOneObservation(ob):
        yData = ndarray((210, 160))
        for i in range(ob.shape[0]):
            for j in range(ob.shape[1]):
                rgb = ob[i][j]
                yData[i][j] = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return yData

    def getYChannelsForAllObservations(maxObservations):
        yChannels = []
        for ob in maxObservations:
            yChannels.append(ob)
        return yChannels

    def step2(yChannels):
        preprocessedImage = ndarray((84,84,4))
        for imgCounter in range(len(yChannels)):
            # TODO: look into bilinear reduction
            img = imresize(yChannels[imgCounter], (84, 84))
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    preprocessedImage[i][j][imgCounter] = img[i][j]
        return preprocessedImage

    return step2(getYChannelsForAllObservations(step1(recentObservations)))

def executeKActions(env, action):
    recentKObservations = []
    rewardTotal = 0
    done = False
    for i in range(2*k):
        observation, reward, done, info = env.step(action)
        recentKObservations.append(observation)
        rewardTotal += reward
        if done:
            recentKObservations = []
            recentKObservations = [observation] * (2*k)
            break
    return recentKObservations, rewardTotal, done

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
        action = env.action_space.sample()
        recentKObservations, rewardFromKSteps, done = executeKActions(env, action)
        prevPhi = preprocess(recentKObservations)
        for t in range(1000000):
            env.render()
            # perform epsilon greedy approach in choosing action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # exploration
            else:
                action = numpy.argmax(Q.predict(prevPhi))

            # RUN the selected action for 2K times for better results
            recentKObservations, rewardFromKSteps, done = executeKActions(env, action)
            # get preprocessed image
            phi = preprocess(recentKObservations)
            # add it to the replay memory
            D.append((prevPhi, action, rewardFromKSteps, phi))
            prevPhi = phi
            # Ensure the size of the D is not going above the limit
            if len(D) == N:
                D.pop(0)

            total_reward += rewardFromKSteps
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
                    Ccounter = 0
    print "average reward={}".format(average/num_episodes)

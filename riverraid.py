import numpy
from numpy.core.numeric import ndarray
from scipy.misc.pilutil import imresize
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from collections import deque
import random
import pydot
import gym
import scipy.misc

# hyperparameters
C = 10000
N = 1000000
NUM_EPISODES = 200
NUM_ITERATIONS = 1
EPSILON_MIN = 0.1
ESPILON_DECAY = 0.995
LEARNING_RATE = 0.00025
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1000000
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = 2

def initNet():
    model = Sequential()

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', input_shape=(20, 20, 32)))
    model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(9, 9, 64)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(18, activation='linear', input_shape=(512,)))
    # model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    # model.fit(X_train, Y_train, batch_size=32, verbose=1)
    return model

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


if __name__ == '__main__':
    env = gym.make('Riverraid-v0')
    memory = deque([], REPLAY_MEMORY_SIZE)
    Q = initNet()
    Q.summary()
    plot_model(Q, to_file='model.png')
    # TODO: figure out if cnn creation is deterministic
    QHat = initNet()
    epsilon = 1.0
    done = False
    c = 0
    average = 0
    for i_episode in range(NUM_EPISODES):
        total_reward = 0
        observation = env.reset()
        # TODO: maybe just need to do step2 here
        selfPhi = phi = preprocess(observation, observation)
        for t in range(NUM_ITERATIONS):
            action = None
            env.render()
            # choose random action with probability epsilon:
            val = random.uniform(0, 1)
            if val <= epsilon:
                action = env.action_space.sample()
            else:
                action = numpy.argmax(Q.predict(selfPhi)[0])

            # generate and save new action
            nextObservation, reward, done, info = env.step(action)
            # TODO: figure out whether to set reward
            # Preprocess state to phi
            # TODO: preprocess needs to take a list of 8 frames but can be assumed for now
            nextPhi = preprocess(observation, nextObservation)
            memory.append((selfPhi, action, reward, nextPhi, done))
            phi = nextPhi
            total_reward += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                #print observation, reward, done, info
                average += total_reward
                print "episode reward ={}".format(total_reward)
                print "observation = {}".format(observation.shape)
                break

            # update and do gradient descent
            if len(memory) > MINIBATCH_SIZE:
                minibatch = random.sample(memory, MINIBATCH_SIZE)
                for selfPhi, action, reward, nextPhi, done in minibatch:
                    target = reward
                    # update target if not in end state
                    if not done:
                        prediction = numpy.amax(QHat.predict(nextPhi)[0])
                        target = (reward + DISCOUNT_FACTOR * prediction)
                    actual = Q.predict(selfPhi)
                    target[0][action] = target
                    Q.fit(selfPhi, target, epochs=1, verbose=0)
                if epsilon > 0:
                    epsilon *= EPSILON_DECAY

            # update Qhat
            if c == UPDATE_FREQUENCY:
                QHat = Q
                c = 0
            else:
                c += 1

    print "average reward={}".format(average/NUM_EPISODES)

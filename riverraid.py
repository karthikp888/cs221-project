import numpy
from numpy import random
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
NUM_ITERATIONS = 10000
EPSILON_MIN = 0.1
ESPILON_DECAY = 0.995
LEARNING_RATE = 0.00025
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 1000000
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = 2
K_OPERATION_COUNT = 4

def initNet():
    model = Sequential()

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4)))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', input_shape=(20, 20, 32)))
    model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(9, 9, 64)))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(18, activation='linear', input_shape=(512,)))
    model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
    return model

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
        for i in range(K_OPERATION_COUNT):
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
            yChannels.append(getYChannelForOneObservation(ob))
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

    return step2(getYChannelsForAllObservations(step1()))

# def executeKActions(env, action):
#     recentKObservations = []
#     rewardTotal = 0
#     done = False
#     for i in range(2*K_OPERATION_COUNT):
#         env.render()
#         observation, reward, done, info = env.step(action)
#         recentKObservations.append(observation)
#         rewardTotal += reward
#         if done:
#             recentKObservations = []
#             recentKObservations = [observation] * (2*K_OPERATION_COUNT)
#             break
#     return env, recentKObservations, rewardTotal, done

if __name__ == '__main__':
    env = gym.make('Riverraid-v0')
    memory = deque([], REPLAY_MEMORY_SIZE)
    Q = initNet()
    Q.summary()
    #plot_model(Q, to_file='model.png')
    # TODO: figure out if cnn creation is deterministic
    QHat = initNet()
    weights = Q.get_weights()
    QHat.set_weights(weights)
    epsilon = 1.0
    done = False
    c = 0
    average = 0
    for i_episode in range(NUM_EPISODES):
        def executeKActions(action):
            recentKObservations = []
            rewardTotal = 0
            done = False
            for i in range(2 * K_OPERATION_COUNT):
                env.render()
                observation, reward, done, info = env.step(action)
                recentKObservations.append(observation)
                rewardTotal += reward
                if done:
                    recentKObservations = []
                    recentKObservations = [observation] * (2 * K_OPERATION_COUNT)
                    break
            return recentKObservations, rewardTotal, done


        total_reward = 0
        observation = env.reset()
        # TODO: maybe just need to do step2 here
        action = env.action_space.sample()
        recentKObservations, rewardFromKSteps, done = executeKActions(action)
        currentPhi = preprocess(recentKObservations)

        for t in range(NUM_ITERATIONS):
            action = None
            # choose random action with probability epsilon:
            val = random.uniform(0, 1)
            if val <= epsilon:
                action = env.action_space.sample()
            else:
                action = numpy.argmax(Q.predict(selfPhi[numpy.newaxis,:,:,:])[0])

            # RUN the selected action for 2K times for better results
            recentKObservations, rewardFromKSteps, done = executeKActions(action)
            # get preprocessed image
            nextPhi = preprocess(recentKObservations)
            # add it to the replay memory
            memory.append((currentPhi, action, rewardFromKSteps, nextPhi, done))
            currentPhi = nextPhi
            total_reward += rewardFromKSteps

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                average += total_reward
                print "episode reward ={}".format(total_reward)
                break

            # update and do gradient descent
            if len(memory) > MINIBATCH_SIZE:
                minibatch = random.sample(memory, MINIBATCH_SIZE)
                for selfPhi, action, reward, nextPhi, done in minibatch:
                    target = reward
                    # update target if not in end state
                    if not done:
                        prediction = numpy.amax(QHat.predict(nextPhi[numpy.newaxis,:,:,:], batch_size=1)[0])
                        target = (reward + DISCOUNT_FACTOR * prediction)
                    actual = Q.predict(selfPhi[numpy.newaxis,:,:,:])
                    actual[0][action] = target
                    Q.fit(selfPhi[numpy.newaxis,:,:,:], actual, epochs=1, verbose=0)
                if epsilon > EPSILON_MIN:
                    epsilon *= ESPILON_DECAY

            # update Qhat
            if c == UPDATE_FREQUENCY:
                weights = Q.get_weights()
                QHat.set_weights(weights)
                c = 0
            else:
                c += 1
    print "average reward={}".format(average/NUM_EPISODES)

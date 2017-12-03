import numpy
from numpy import random
from numpy.core.numeric import ndarray
from scipy.misc.pilutil import imresize
from scipy.misc.pilutil import imshow
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.utils import np_utils, plot_model
from keras.datasets import mnist
from collections import deque
import random
import pydot
import gym
import scipy.misc
import os
import pickle
import time

# hyperparameters
NUM_EPISODES = 100000
NUM_ITERATIONS = 10000
EPSILON_MIN = 0.1
ESPILON_DECAY = (0.9/1000000)
LEARNING_RATE = 0.00025
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 40000
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = 10000
K_OPERATION_COUNT = 4
REPLAY_START_SIZE = 10000
NO_OP_MAX = 45
SHOOT_ONLY_ACTION = 1
ACTION_SPACE = [0, 1, 2, 3, 4]

def initNet():
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4), kernel_initializer='glorot_uniform'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', input_shape=(20, 20, 32), kernel_initializer='glorot_uniform'))
    model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(9, 9, 64), kernel_initializer='glorot_uniform'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(5, activation='linear', input_shape=(512,), kernel_initializer='glorot_uniform'))
    model.compile(loss='mse', optimizer=RMSprop(lr=LEARNING_RATE, epsilon=0.01, decay=0.95, rho=0.95))
    return model

def preprocess(recentObservations):
    def getMaxBetweenTwo(ob1, ob2):
        return numpy.maximum(ob1,ob2)

    def step1():
        maxObservations = []
        for i in xrange(K_OPERATION_COUNT):
            maxObservations.append(getMaxBetweenTwo(recentObservations[i], recentObservations[i+1]))
        return maxObservations

    def rgb2gray(rgb):
        r,g,b = rgb[:,:,0],rgb[:,:,1],rgb[:,:,2]
        gray = 0.299 * r + 0.587 * g + 0.114 * b
        return gray

    def getYChannelForOneObservation(ob):
        yData = rgb2gray(ob)
        return yData

    def getYChannelsForAllObservations(maxObservations):
        yChannels = []
        for ob in maxObservations:
            yChannels.append(getYChannelForOneObservation(ob))
        return yChannels

    def step2(yChannels):
        preprocessedImage = ndarray((84,84,4))
        for imgCounter in xrange(len(yChannels)):
            # TODO: look into bilinear reduction
            preprocessedImage[:,:, imgCounter] = imresize(yChannels[imgCounter], (84, 84))
        return preprocessedImage

    return step2(getYChannelsForAllObservations(step1()))

def executeKActions(action, prevObservation):
    recentKObservations = []
    recentKObservations.append(prevObservation)
    rewardTotal = 0
    done = False
    for i in xrange(K_OPERATION_COUNT):
        # env.render()
        observation, reward, done, info = env.step(action+1)
        recentKObservations.append(observation)
        rewardTotal += reward
        if done:
            recentKObservations = []
            recentKObservations = [observation] * ((K_OPERATION_COUNT) + 1)
            break
    return recentKObservations, rewardTotal, done

def getNextModelName():
    max = 0
    for filename in os.listdir('.'):
        if filename.startswith('model') and filename.endswith('.h5'):
            i = filename.split('.')[0].split('model')[1]
            if max < i:
                max = i
    return "model{}.h5".format(max+1)

def getMostRecentModel():
    max = 0
    for filename in os.listdir('.'):
        if filename.startswith('model') and filename.endswith('.h5'):
            i = filename.split('.')[0].split('model')[1]
            if max < i:
                max = i
    return "model{}.h5".format(max)


if __name__ == '__main__':
    env = gym.make('Riverraid-v0')
    memory = deque([], REPLAY_MEMORY_SIZE)
    Q = initNet()
    Q.summary()
    #plot_model(Q, to_file='model.png')
    recentModel = getMostRecentModel()
    if os.path.exists(recentModel):
        print "load weights from previous run"
        Q.load_weights(recentModel)
    else :
        exit
    # TODO: figure out if cnn creation is deterministic
    QHat = initNet()
    weights = Q.get_weights()
    QHat.set_weights(weights)
    epsilon = 1.0
    done = False
    c = 0
    average = 0
    #load replay_start_size observations. generate if needed. We initially
    #load this many obeservatins into memory before we start training the model
    prevObservation = []
    if os.path.exists("memory.txt"):
        pass
        print "Loading initial set of observations"
        memory = pickle.load(open("memory.txt", "rb"))
        print "Initial observations loaded"
    else:
        prevObservation = env.reset()
        action = random.choice(ACTION_SPACE)
        recentKObservations, rewardFromKSteps, done = executeKActions(action, prevObservation)
        prevObservation = recentKObservations[K_OPERATION_COUNT]
        currentPhi = preprocess(recentKObservations)
        for j in xrange(REPLAY_START_SIZE):
            # if (j%100) == 0:
            #     print j
            action = random.choice(ACTION_SPACE)
            recentKObservations, rewardFromKSteps, done = executeKActions(action, prevObservation)
            prevObservation = recentKObservations[K_OPERATION_COUNT]
            nextPhi = preprocess(recentKObservations)
            # add it to the replay memory
            memory.append((currentPhi, action, rewardFromKSteps, nextPhi, done))
            currentPhi = nextPhi
            if done:
                prevObservation = env.reset()
                action = random.choice(ACTION_SPACE)
                recentKObservations, rewardFromKSteps, done = executeKActions(action, prevObservation)
                prevObservation = recentKObservations[K_OPERATION_COUNT]
                currentPhi = preprocess(recentKObservations)
        print "generated initial set of observations...writing to file"
        pickle.dump(memory, open("memory.txt", "wb"))
        print "initial observations written to file"
    for i_episode in xrange(NUM_EPISODES):
        sgd_skip = 0
        num_target_updates=0
        episodeStart = time.time()
        total_reward = 0
        prevObservation = env.reset()
        # TODO: maybe just need to do step2 here

        # Do SHOOT_ONLY_ACTION operation for NO_OP_MAX times at the beginning of each episode
        action = SHOOT_ONLY_ACTION
        for i in xrange(NO_OP_MAX):
            env.step(SHOOT_ONLY_ACTION)
        recentKObservations, rewardFromKSteps, done = executeKActions(action, prevObservation)
        prevObservation = recentKObservations[K_OPERATION_COUNT]
        currentPhi = preprocess(recentKObservations)

        non_random=0
        my_random=0
        for t in xrange(NUM_ITERATIONS):
            action = None
            # choose random action with probability epsilon:
            val = random.uniform(0, 1)
            if val <= epsilon:
                action = random.choice(ACTION_SPACE)
                my_random+=1
            else:
                non_random+=1
                action = numpy.argmax(Q.predict(currentPhi[numpy.newaxis,:,:,:], batch_size=1)[0])

            recentKObservations, rewardFromKSteps, done = executeKActions(action, prevObservation)
            prevObservation = recentKObservations[K_OPERATION_COUNT]
            # get preprocessed image
            nextPhi = preprocess(recentKObservations)
            # add it to the replay memory
            memory.append((currentPhi, action, rewardFromKSteps, nextPhi, done))
            currentPhi = nextPhi
            total_reward += rewardFromKSteps

            if done:
                average += total_reward
                print("Episode={} reward={} steps={} secs={} epsilon={} non_rand={} my_rand={}".format(i_episode, total_reward, t+1, time.time() - episodeStart, epsilon, non_random, my_random))
                break

            # update and do gradient descent
            if len(memory) > MINIBATCH_SIZE and sgd_skip == 4:
                sgd_skip = 0
                minibatch = random.sample(memory, MINIBATCH_SIZE)
                index = 0
                selfPhiList = numpy.empty((MINIBATCH_SIZE,84,84,4))
                actualList = numpy.empty((MINIBATCH_SIZE,len(ACTION_SPACE)))
                for selfPhi, action, reward, nextPhi, done in minibatch:
                    target = reward
                    # update target if not in end state
                    if not done:
                        prediction = numpy.amax(QHat.predict(nextPhi[numpy.newaxis,:,:,:], batch_size=1)[0])
                        target = (reward + DISCOUNT_FACTOR * prediction)
                    actual = Q.predict(selfPhi[numpy.newaxis,:,:,:], batch_size=1)
                    actual[0][action] = target
                    actualList[index] = actual[0]
                    selfPhiList[index] = selfPhi
                    index += 1
                    #imshow(selfPhi[:,:, 0])
                    #imshow(nextPhi[:,:, 0])
                Q.fit(selfPhiList, actualList, epochs=1, verbose=0)
                c += 1
                # update Qhat
                if c == UPDATE_FREQUENCY:
                    weights = Q.get_weights()
                    QHat.set_weights(weights)
                    QHat.save_weights(getNextModelName())
                    c = 0
                    print "target NN update={}".format(num_target_updates)
            else:
                sgd_skip += 1

            if epsilon > EPSILON_MIN:
                epsilon -= ESPILON_DECAY


    print "average reward={}".format(average/NUM_EPISODES)
    #QHat.save_weights("model0.h5")

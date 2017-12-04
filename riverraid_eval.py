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
from keras import backend as K
from collections import deque
import random
import pydot
import gym
import scipy.misc
import os
import pickle
import time

# hyperparameters
NUM_EPISODES = 100
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


def huber_loss(target, prediction):
    # sqrt(1+error^2)-1
    error = prediction - target
    # return K.sum(K.sqrt(1+K.square(error))-1, axis=-1)
    return K.mean(K.sqrt(1+K.square(error))-1, axis=-1)

def initNet():
    model = Sequential()

    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4), kernel_initializer='glorot_uniform'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', input_shape=(20, 20, 32), kernel_initializer='glorot_uniform'))
    model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(9, 9, 64), kernel_initializer='glorot_uniform'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(5, activation='linear', input_shape=(512,), kernel_initializer='glorot_uniform'))
    model.compile(loss=huber_loss, optimizer=RMSprop(lr=LEARNING_RATE, epsilon=0.01, decay=0.95, rho=0.95))
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
        observation, reward, done, info = env.step(action+1)
        recentKObservations.append(observation)
        rewardTotal += reward
        if done:
            recentKObservations = []
            recentKObservations = [observation] * ((K_OPERATION_COUNT) + 1)
            break
    return recentKObservations, rewardTotal, done



if __name__ == '__main__':
    env = gym.make('Riverraid-v0')
    memory = deque([], REPLAY_MEMORY_SIZE)
    Q = initNet()
    #plot_model(Q, to_file='model.png')
    if os.path.exists("huber_loss_test_3_184.h5"):
        print "load weights from previous run"
        Q.load_weights("huber_loss_test_3_184.h5")
    else :
        exit
    # Q.summary()
    # print 'weights: {}'.format(Q.get_weights())

    # TODO: figure out if cnn creation is deterministic
    epsilon = 1.0
    done = False
    c = 0
    average = 0
    episode_rewards = []
    #load replay_start_size observations. generate if needed. We initially
    #load this many obeservatins into memory before we start training the model
    prevObservation = []
    for i_episode in xrange(NUM_EPISODES):
        episodeStart = time.time()
        total_reward = 0
        prevObservation = env.reset()
        # TODO: maybe just need to do step2 here
        action = random.choice([0,1,2,3,4])
        recentKObservations, rewardFromKSteps, done = executeKActions(action, prevObservation)
        prevObservation = recentKObservations[K_OPERATION_COUNT]
        currentPhi = preprocess(recentKObservations)

        non_random=0
        my_random=0
        for t in xrange(NUM_ITERATIONS):
            action = None
            # choose random action with probability epsilon:
            #os.system("clear")
            #print Q.predict(currentPhi[numpy.newaxis,:,:,:], batch_size=1)
            action = numpy.argmax(Q.predict(currentPhi[numpy.newaxis,:,:,:], batch_size=1)[0])

            recentKObservations, rewardFromKSteps, done = executeKActions(action, prevObservation)
            prevObservation = recentKObservations[K_OPERATION_COUNT]
            total_reward += rewardFromKSteps
            env.render()
            #print "action = {} iter={}".format(action,t)
            nextPhi = preprocess(recentKObservations)
            currentPhi=nextPhi
            if done:
                episode_rewards.append(total_reward)
                print("Episode={} reward={} steps={} secs={} epsilon={} non_rand={} my_rand={}".format(i_episode, total_reward, t+1, time.time() - episodeStart, epsilon, non_random, my_random))
                break


    print "average reward={}".format(numpy.mean(episode_rewards))
    print "std dev reward={}".format(numpy.std(episode_rewards))
    print "median reward={}".format(numpy.median(episode_rewards))
    #QHat.save_weights("model.h5")

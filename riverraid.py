import numpy
from numpy import random
from numpy.core.numeric import ndarray
from scipy.misc.pilutil import imresize
from scipy.misc.pilutil import imshow
from keras.models import Sequential
from keras import backend as K
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
REPLAY_MEMORY_SIZE = 250000
DISCOUNT_FACTOR = 0.99
UPDATE_FREQUENCY = 10000
K_OPERATION_COUNT = 4
REPLAY_START_SIZE = 50000
ACTION_SPACE = range(18)
NUM_ACTIONS = len(ACTION_SPACE)
ACTION_NOOP = 0


#manav's pseudo-huber
#def huber_loss(target, prediction):
    #error = prediction - target
    #return K.sum(K.sqrt(1+K.square(error))-1, axis=-1)

def huber_loss(y_true, y_pred, clip_value=1):
    clip_value = 1
    # Huber loss, see https://en.wikipedia.org/wiki/Huber_loss and
    # https://medium.com/@karpathy/yes-you-should-understand-backprop-e2f06eab496b
    # for details.
    assert clip_value > 0.

    x = y_true - y_pred
    if numpy.isinf(clip_value):
        # Spacial case for infinity since Tensorflow does have problems
        # if we compare `K.abs(x) < np.inf`.
        return .5 * K.square(x)

    condition = K.abs(x) < clip_value
    squared_loss = .5 * K.square(x)
    linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        if hasattr(tf, 'select'):
            return tf.select(condition, squared_loss, linear_loss)  # condition, true, false
        else:
            return tf.where(condition, squared_loss, linear_loss)  # condition, true, false
    elif K.backend() == 'theano':
        from theano import tensor as T
        return T.switch(condition, squared_loss, linear_loss)
    else:
        raise RuntimeError('Unknown backend "{}".'.format(K.backend()))

def initNet():
    model = Sequential()
    model.add(Convolution2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(84, 84, 4), kernel_initializer='glorot_uniform'))
    model.add(Convolution2D(64, (4, 4), strides=(2, 2), activation='relu', input_shape=(20, 20, 32), kernel_initializer='glorot_uniform'))
    model.add(Convolution2D(64, (3, 3), activation='relu', input_shape=(9, 9, 64), kernel_initializer='glorot_uniform'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_initializer='glorot_uniform'))
    model.add(Dense(NUM_ACTIONS, activation='linear', input_shape=(512,), kernel_initializer='glorot_uniform'))
    #model.compile(loss='mse', optimizer=RMSprop(lr=LEARNING_RATE, epsilon=0.01, decay=0.95, rho=0.95))
    model.compile(loss=huber_loss, optimizer=RMSprop(lr=LEARNING_RATE, epsilon=0.01, decay=0.95, rho=0.95))
    return model

def preprocess(recentObservations):
    def getMaxBetweenTwo(ob1, ob2):
        return numpy.maximum(ob1,ob2)

    def step1():
        maxObservations = []
        for i in xrange(0, K_OPERATION_COUNT * 2, 2):
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

    return (numpy.round(step2(getYChannelsForAllObservations(step1())))).astype(numpy.uint8)

#handle loss of life similar to end of game
lives = 4

#this function works only for a noFrameSkip environment 
#when an action A is provided it is run as follows:
#t0,t4,t8,t12 is A
#t1,t2,t3,t5,t6,t7,t9,t10,t11,t13,t14,t15 is NOOP
#observations obtained by running the following are recorded 
#hence we return K_OPERATION_COUNT *2 observations
#(t0,t1), (t4,t5), (t8,t9) (t12,t13) 
#the above tuples are then preprocessed later to obtain a frame
def executeKActions(action):
    recentKObservations = []
    rewardTotal = 0
    done = False
    global lives
    for i in xrange(K_OPERATION_COUNT * K_OPERATION_COUNT):
        # env.render()
        observation = []
        reward = 0
        done = 0
        info = dict()

        #when an action A is provided it is run as follows:
        #t0,t4,t8,t12 is A
        #t1,t2,t3,t5,t6,t7,t9,t10,t11,t13,t14,t15 is NOOP
        if (i % K_OPERATION_COUNT) == 0:
            observation, reward, done, info = env.step(action)
        else:
            observation, reward, done, info = env.step(ACTION_NOOP)

        #observations obtained by running the following are recorded
        #(t0,t1), (t4,t5), (t8,t9) (t12,t13) 
        if ((i % K_OPERATION_COUNT) == 0) or ((i % K_OPERATION_COUNT) == 1):
            recentKObservations.append(observation)

        rewardTotal += reward
        if done or (info["ale.lives"] == (lives -1)):
            recentKObservations = []
            recentKObservations = [observation] * ((K_OPERATION_COUNT * 2) )
            lives = lives - 1
            rewardTotal = -1
            break
        else:
            lives = info["ale.lives"]
    if rewardTotal > 0:
        rewardTotal = 1
    elif rewardTotal < 0:
        rewardTotal = -1
    return recentKObservations, rewardTotal, done


if __name__ == '__main__':

    #stochastic and frame skip(provided action is only run 75 percent of time
    #25% of time the previous action is run
    #env = gym.make('Riverraid-v0')

    #non-stochastic and frame skip
    #env = gym.make('Riverraid-v4')

    #stochastic and no frame skip (for real final results)
    env = gym.make('RiverraidNoFrameskip-v0')

    #non-stochastic and no frame skip (best for checking if you algo is learing)
    #env = gym.make('RiverraidNoFrameskip-v4')

    memory = deque([], REPLAY_MEMORY_SIZE)
    Q = initNet()
    Q.summary()
    if os.path.exists("model.h5"):
        print "load weights from previous run"
        Q.load_weights("model.h5")
    else :
        exit
    QHat = initNet()
    weights = Q.get_weights()
    QHat.set_weights(weights)

    #saving the initialization makes results more reproducable
    #also note that this is purposely different from model.h5 which is for load only
    model_num = 0
    QHat.save_weights("model_{}.h5".format(model_num))
    model_num += 1

    epsilon = 1.0
    done = False
    c = 0
    average = 0
    #load replay_start_size observations. generate if needed. We initially
    #load this many obeservatins into memory before we start training the model
    if os.path.exists("memory.txt"):
        pass
        print "Loading initial set of observations"
        memory = pickle.load(open("memory.txt", "rb"))
        print "Initial observations loaded from memory.txt"
    else:
        env.reset()
        action = random.choice(ACTION_SPACE)
        recentKObservations, rewardFromKSteps, done = executeKActions(action)
        currentPhi = preprocess(recentKObservations)
        print "Replay memory loading"
        for j in xrange(REPLAY_START_SIZE):
            if (j%1000)==0:
                print j,
            action = random.choice(ACTION_SPACE)
            recentKObservations, rewardFromKSteps, done = executeKActions(action)
            nextPhi = preprocess(recentKObservations)
            # add it to the replay memory
            memory.append((currentPhi, action, rewardFromKSteps, nextPhi, done))
            currentPhi = nextPhi
            if done:
                env.reset()
                action = random.choice(ACTION_SPACE)
                recentKObservations, rewardFromKSteps, done = executeKActions(action)
                currentPhi = preprocess(recentKObservations)
        print "Writing replay memory to file"
        pickle.dump(memory, open("memory.txt", "wb"))
        print "Replay memory written to file"
    for i_episode in xrange(NUM_EPISODES):
        sgd_skip = 0
        num_target_updates=0
        episodeStart = time.time()
        total_reward = 0
        env.reset()
        # TODO: maybe just need to do step2 here

        action = random.choice(ACTION_SPACE)
        recentKObservations, rewardFromKSteps, done = executeKActions(action)
        currentPhi = preprocess(recentKObservations)

        predicted_action=0
        random_action=0
        for t in xrange(NUM_ITERATIONS):
            action = None
            # choose random action with probability epsilon:
            val = random.uniform(0, 1)
            if val <= epsilon:
                action = random.choice(ACTION_SPACE)
                random_action+=1
            else:
                predicted_action+=1
                action = numpy.argmax(Q.predict(currentPhi[numpy.newaxis,:,:,:], batch_size=1)[0])

            recentKObservations, rewardFromKSteps, done = executeKActions(action)
            # get preprocessed image
            nextPhi = preprocess(recentKObservations)
            # add it to the replay memory
            memory.append((currentPhi, action, rewardFromKSteps, nextPhi, done))
            currentPhi = nextPhi
            total_reward += rewardFromKSteps

            if done:
                average += total_reward
                print("Episode={} reward={} steps={} secs={} epsilon={} predicted_action={} random_action={}".format(i_episode, total_reward, t+1, time.time() - episodeStart, epsilon, predicted_action, random_action))
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
                    #imshow(selfPhi[:,:, 3])
                    #imshow(nextPhi[:,:, 0])
                Q.fit(selfPhiList, actualList, epochs=1, verbose=0)
                c += 1
                # update Qhat
                if c == UPDATE_FREQUENCY:
                    weights = Q.get_weights()
                    QHat.set_weights(weights)
                    QHat.save_weights("model_{}.h5".format(model_num))
                    model_num += 1
                    c = 0
                    print "target NN update={}".format(num_target_updates)
            else:
                sgd_skip += 1

            if epsilon > EPSILON_MIN:
                epsilon -= ESPILON_DECAY


    print "average reward={}".format(average/NUM_EPISODES)

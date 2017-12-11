from fileinput import filename

import numpy
from numpy.core.numeric import ndarray
from scipy.misc.pilutil import imresize
from keras.models import Sequential
from keras import backend as K
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import RMSprop
import random
import gym
import time
import sys
# import matplotlib.pyplot as plt

# hyperparameters
# NUM_EPISODES = 1
# NUM_ITERATIONS = 10000
# LEARNING_RATE = 0.00025
# K_OPERATION_COUNT = 4
# ACTION_SPACE = range(18)
# NUM_ACTIONS = len(ACTION_SPACE)
# ACTION_FIRE = 1
# ACTION_NOOP = 0
NUM_EPISODES = 10
NUM_ITERATIONS = 10000
LEARNING_RATE = 0.00025
K_OPERATION_COUNT = 4
ACTION_SPACE = range(18)
NUM_ACTIONS = len(ACTION_SPACE)
ACTION_FIRE = 1
ACTION_NOOP = 0

#only needed for model.compile. Never used here as we don't train the model
#i think this code has mit license so we should be good to use it.
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

    return step2(getYChannelsForAllObservations(step1()))

lives = 4
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
            break
        else:
            lives = info["ale.lives"]
    return recentKObservations, rewardTotal, done


if __name__ == '__main__':
    #env = gym.make('Riverraid-v0')
    # env = gym.make('RiverraidNoFrameskip-v4')
    env = gym.make(sys.argv[2])
    # env.frameskip = 1
    Q = initNet()
    #Q.summary()
    Q.load_weights(sys.argv[1])
    epsilon = 1.0
    done = False
    c = 0
    average = 0
    for i_episode in xrange(NUM_EPISODES):
        sgd_skip = 0
        num_target_updates=0
        episodeStart = time.time()
        total_reward = 0
        env.reset()
        # TODO: maybe just need to do step2 here

        #for introducing stochasticity te same way google does.
        #if i_episode == 0:
            #for i in xrange(random.randint(0,4)):
                #action = 0
                #recentKObservations, rewardFromKSteps, done = executeKActions(action)
        action = ACTION_NOOP
        recentKObservations, rewardFromKSteps, done = executeKActions(action)
        currentPhi = preprocess(recentKObservations)

        predicted_action=0
        for t in xrange(NUM_ITERATIONS):
            action = None
            # choose random action with probability epsilon:
            val = random.uniform(0, 1)
            predicted_action+=1
            action = numpy.argmax(Q.predict(currentPhi[numpy.newaxis,:,:,:], batch_size=1)[0])

            recentKObservations, rewardFromKSteps, done = executeKActions(action)
            # get preprocessed image
            nextPhi = preprocess(recentKObservations)
            currentPhi = nextPhi
            total_reward += rewardFromKSteps

            if done:
                average += total_reward
                print("Episode={} reward={} steps={} secs={} epsilon={} predicted_action={}".format(i_episode, total_reward, t+1, time.time() - episodeStart, epsilon, predicted_action))
                break
    print "writing avg rewards started"
    with open('avg_reward.tsv', 'a+') as f:
        f.write(sys.argv[1].split('.')[0].split('model_')[1])
        f.write("\t")
        f.write("%d\r\n" % (average/NUM_EPISODES))
        f.close
    plotX = []
    plotY = []
    print "writing avg rewards ends"
    print "plotting graph start"
    # with open('avg_reward.tsv') as f:
    #     line = f.readline()
    #     while line:
    #         plotX.append(line.strip().split('\t')[0])
    #         plotY.append(line.strip().split('\t')[1])
    #         line = f.readline()
    # # plt.plot(plotX, plotY)
    # # plt.ylabel('Avg. Rewards')
    # # plt.xlabel('Training Epoch')
    # # plt.savefig('reward.png')
    # print "plotting graph end"
    #f.close()
    print "average reward={}".format(average/NUM_EPISODES)

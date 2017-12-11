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
import argparse
import json
STATS = {}

def flush_stats():
    with open('stats.json', 'w') as f:
        print >>f, json.dumps(STATS)
# hyperparameters
ACTION_NOOP = 0

argParser = argparse.ArgumentParser()
argParser.add_argument('--num-episodes', type=int, default=1000000)
argParser.add_argument('--num-iterations', type=int, default=100000)
argParser.add_argument('--epsilon-min', type=float, default=0.1)
argParser.add_argument('--epsilon-decay', type=float, default=(0.9/1000000))
argParser.add_argument('--learning-rate', type=float, default=0.00025)
argParser.add_argument('--minibatch-size', type=int, default=32)
argParser.add_argument('--replay-memory-size', type=int, default=125000)
argParser.add_argument('--discount-factor', type=float, default=0.99)
argParser.add_argument('--update-frequency', type=int, default=10000)
argParser.add_argument('--replay-start-size', type=int, default=50000)
argParser.add_argument('--k-operation-count', type=int, default=4)
argParser.add_argument('--action-space', type=int, default=18)
argParser.add_argument('--action-fire', type=int, default=1)
argParser.add_argument('--action-noop', type=int, default=0)
argParser.add_argument('--loss-function', default=0)
argParser.add_argument('--gym-environment', default='RiverraidNoFrameskip-v0')
args = argParser.parse_args()
print args
NUM_EPISODES = args.num_episodes
NUM_ITERATIONS = args.num_iterations
EPSILON_MIN = args.epsilon_min
EPSILON_DECAY = args.epsilon_decay
LEARNING_RATE = args.learning_rate
MINIBATCH_SIZE = args.minibatch_size
REPLAY_MEMORY_SIZE = args.replay_memory_size
DISCOUNT_FACTOR = args.discount_factor
UPDATE_FREQUENCY = args.update_frequency
REPLAY_START_SIZE = args.replay_start_size
K_OPERATION_COUNT = args.k_operation_count
ACTION_SPACE = range(args.action_space)
NUM_ACTIONS = len(ACTION_SPACE)
ACTION_FIRE = args.action_fire
ACTION_NOOP = args.action_noop
LOSS_Function = args.loss_function
gym_environment = args.gym_environment
PRINT_FREQUENCY = 2
PRINT_COUNT = 0

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
    if LOSS_Function == 'MSE':
        model.compile(loss='MSE', optimizer=RMSprop(lr=LEARNING_RATE, epsilon=0.01, decay=0.95, rho=0.95))
    else:
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
    gym_environment = 'RiverraidNoFrameskip-v0'
    env = gym.make(gym_environment)

    #non-stochastic and no frame skip (best for checking if you algo is learing)
    #env = gym.make('RiverraidNoFrameskip-v4')

    memory = deque([], REPLAY_MEMORY_SIZE)
    Q = initNet()
    Q.summary()
    if os.path.exists("model.h5"):
        print "Found weights from previous run, loading it"
        Q.load_weights("model.h5")
        print "Found weights from previous run, loading complete!"
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
        print "initial set of observations found, loading it"
        memory = pickle.load(open("memory.txt", "rb"))
        print "initial set of observations found, loading complete!"
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
            STATS['total_episode'] = NUM_EPISODES
            if done:
                average += total_reward
                print("Episode={} reward={} steps={} secs={} epsilon={} predicted_action={} random_action={}".format(i_episode, total_reward, t+1, time.time() - episodeStart, epsilon, predicted_action, random_action))
                PRINT_COUNT += 1
                if PRINT_COUNT % PRINT_FREQUENCY == 0:
                    STATS['episode'] = i_episode
                    STATS['reward'] = total_reward
                    STATS['steps'] = t + 1
                    STATS['secs'] = time.time() - episodeStart
                    STATS['epsilon'] = epsilon
                    STATS['predicted_action'] = predicted_action
                    STATS['random_action'] = random_action
                    flush_stats()
                    PRINT_COUNT = 0

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
                    print "Evaluating the model:", "model_{}.h5".format(model_num)
                    os.system("python riverraid_eval.py model_{}.h5 {}".format(model_num, gym_environment))
                    # model_eval.evaluate("model_{}.h5".format(model_num))
                    print "Evaluation Done!"
                    model_num += 1
                    c = 0
                    print "target NN update={}".format(num_target_updates)
            else:
                sgd_skip += 1

            if epsilon > EPSILON_MIN:
                epsilon -= EPSILON_DECAY

    STATS['average_reward'] = average/NUM_EPISODES
    flush_stats()
    print "average reward={}".format(average/NUM_EPISODES)

from numpy.core.numeric import ndarray
from scipy.misc.pilutil import imresize

import gym
import scipy.misc


def preprocess(prev, curr):
    def step1(prev, curr):
        maxObservation = ndarray(prev.shape)
        for i in range(prev.shape[0]):
            for j in range(prev.shape[1]):
                rgb = observation[i][j]
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

env = gym.make('Riverraid-v0')
average=0
num_episodes=1
for i_episode in range(num_episodes):
    total_reward = 0
    observation = env.reset()
    for t in range(1000000):
        env.render()
        action = env.action_space.sample()
        prevObservation = observation
        observation, reward, done, info = env.step(action)
        # preprocess(prevObservation, observation)
        #print observation, reward, done, info
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            #print observation, reward, done, info
            average += total_reward
            print "episode reward ={}".format(total_reward)
            print "observation = {}".format(observation.shape)
            break
print "average reward={}".format(average/num_episodes)


import argparse
import gym
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('environment')
args = parser.parse_args()

env = gym.make(args.environment)
for i_episode in xrange(20):
    observation = env.reset()
    action = np.zeros((1,))
    for t in xrange(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break

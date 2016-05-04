import gym
import numpy as np
import time

env = gym.make('Pendulum-v0')
for i_episode in xrange(20):
    observation = env.reset()
    action = np.zeros((1,))
    for t in xrange(100):
        env.render()
        print observation, -env.max_torque, env.max_torque
        action[0] = -1/observation[1]
        observation, reward, done, info = env.step(action)
        if done:
            print "Episode finished after {} timesteps".format(t+1)
            break
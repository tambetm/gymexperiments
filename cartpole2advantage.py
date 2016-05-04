import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras import backend as K
import numpy as np

env = gym.make('CartPole-v0')
assert isinstance(env.observation_space, Box)
assert isinstance(env.action_space, Discrete)

x = Input(shape=env.observation_space.shape)
h = Dense(100, activation='tanh')(x)
y = Dense(env.action_space.n + 1)(h)
#z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.mean(a[:, 1:], keepdims=True), output_shape=(env.action_space.n,))(y)
#z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:] - K.max(a[:, 1:], keepdims=True), output_shape=(env.action_space.n,))(y)
z = Lambda(lambda a: K.expand_dims(a[:,0], dim=-1) + a[:,1:], output_shape=(env.action_space.n,))(y)

model = Model(input=x, output=z)
model.summary()
model.compile(optimizer='rmsprop', loss='mse')

prestates = []
actions = []
rewards = []
poststates = []
terminals = []

gamma = 1

for i_episode in xrange(20):
    observation = env.reset()
    for t in xrange(200):
        env.render()

        if np.random.random() < 0.1:
          action = env.action_space.sample()
        else:
          s = np.array([observation])
          q = model.predict(s, batch_size=1)
          #print "q:", q
          action = np.argmax(q[0])
        #print "action:", action

        prestates.append(observation)
        actions.append(action)

        observation, reward, done, info = env.step(action)
        #print "reward:", reward

        rewards.append(reward)
        poststates.append(observation)
        terminals.append(done)

        for k in xrange(10):
          qpre = model.predict(np.array(prestates))
          qpost = model.predict(np.array(poststates))
          for i in xrange(qpre.shape[0]):
            if terminals[i]:
              qpre[i, actions[i]] = rewards[i]
            else:
              qpre[i, actions[i]] = rewards[i] + gamma * np.amax(qpost[i])
          model.train_on_batch(np.array(prestates), qpre)

        if done:
            break
    print "Episode finished after {} timesteps".format(t+1)

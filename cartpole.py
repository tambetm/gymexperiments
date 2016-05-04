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
model.compile(optimizer='adam', loss='mse')

prestates = []
actions = []
rewards = []
poststates = []
terminals = []

BATCH_SIZE = 100
TRAIN_REPEAT = 10
GAMMA = 1

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

        if len(prestates) > 10:
          for k in xrange(TRAIN_REPEAT):
            if len(prestates) > BATCH_SIZE:
              indexes = np.random.choice(len(prestates), size=BATCH_SIZE)
            else:
              indexes = range(len(prestates))

            qpre = model.predict(np.array(prestates)[indexes])
            qpost = model.predict(np.array(poststates)[indexes])
            for i in xrange(qpre.shape[0]):
              if terminals[indexes[i]]:
                qpre[i, actions[indexes[i]]] = rewards[indexes[i]]
              else:
                qpre[i, actions[indexes[i]]] = rewards[indexes[i]] + GAMMA * np.amax(qpost[i])
            model.train_on_batch(np.array(prestates)[indexes], qpre)

        if done:
            break
    print "Episode finished after {} timesteps".format(t+1)

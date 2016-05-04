import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, merge
from keras import backend as K
import numpy as np

BATCH_SIZE = 100
HIDDEN_SIZE = 100
TRAIN_REPEAT = 10
GAMMA = 0.9
NUM_EPISODES = 20
MAX_TIMESTEPS = 500

env = gym.make('Pendulum-v0')
assert isinstance(env.observation_space, Box)
assert isinstance(env.action_space, Box)
assert len(env.action_space.shape) == 1
num_actuators = env.action_space.shape[0]
#env.monitor.start('Pendulum-v0')

def P(x):
  return K.exp(x)

def A(t):
  m, p, u = t
  return - (u - m)**2 * p / 2

def Q(t):
  v, a = t
  return v + a

x = Input(shape=env.observation_space.shape, name='x')
u = Input(shape=env.action_space.shape, name='u')
h1 = Dense(HIDDEN_SIZE, activation='tanh', name='h1')(x)
h = Dense(HIDDEN_SIZE, activation='tanh', name='h')(h1)
v = Dense(1, name='v')(h)
m = Dense(num_actuators, name='m')(h)
l = Dense(num_actuators, name='l')(h)
p = Lambda(P, output_shape=(num_actuators,), name='p')(l)
a = merge([m, p, u], mode=A, output_shape=(None, num_actuators), name="a")
q = merge([v, a], mode=Q, output_shape=(None, num_actuators), name="q")

model = Model(input=[x,u], output=q)
model.summary()
model.compile(optimizer='adam', loss='mse')

m_f = K.function([x], m)
v_f = K.function([x], v)
q_f = K.function([x, u], q)

prestates = []
actions = []
rewards = []
poststates = []
terminals = []

total_reward = 0
for i_episode in xrange(NUM_EPISODES):
    observation = env.reset()
    #print "initial state:", observation
    episode_reward = 0
    for t in xrange(MAX_TIMESTEPS):
        env.render()

        x = np.array([observation])
        u = m_f([x])
        action = u[0] + np.random.randn(num_actuators) / (i_episode  + 1)
        #print "action:", action

        prestates.append(observation)
        actions.append(action)
        #print "prestate:", observation

        observation, reward, done, info = env.step(action)
        episode_reward += reward
        #print "reward:", reward
        #print "poststate:", observation

        rewards.append(reward)
        poststates.append(observation)
        terminals.append(done)

        if len(prestates) > 10:
          for k in xrange(TRAIN_REPEAT):
            if len(prestates) > BATCH_SIZE:
              indexes = np.random.choice(len(prestates), size=BATCH_SIZE)
            else:
              indexes = range(len(prestates))

            v = v_f([np.array(poststates)[indexes]])
            y = np.array(rewards)[indexes] + GAMMA * np.squeeze(v)
            model.train_on_batch([np.array(prestates)[indexes], np.array(actions)[indexes]], y)

        if done:
            break

    print "Episode {} finished after {} timesteps, total reward {}".format(i_episode + 1, t + 1, episode_reward)
    total_reward += episode_reward

print "Average reward per episode {}".format(total_reward / NUM_EPISODES)
#env.monitor.close()

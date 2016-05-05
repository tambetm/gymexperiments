import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense, Lambda, Reshape, merge
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
import theano.tensor as T
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--hidden_size', type=int, default=100)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--batch_norm', action="store_true", default=False)
parser.add_argument('--no_batch_norm', action="store_false", dest="batch_norm")
parser.add_argument('--min_train', type=int, default=10)
parser.add_argument('--train_repeat', type=int, default=10)
parser.add_argument('--gamma', type=float, default=0.9)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='relu')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--optimizer_lr', type=float, default=0.001)
parser.add_argument('--noise_decay', choices=['linear', 'exp', 'fixed'], default='exp')
parser.add_argument('--fixed_noise', type=float, default=0.1)
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_monitor')
parser.add_argument('environment')
args = parser.parse_args()

env = gym.make(args.environment)
assert isinstance(env.observation_space, Box)
assert isinstance(env.action_space, Box)
assert len(env.action_space.shape) == 1
num_actuators = env.action_space.shape[0]

if args.gym_monitor:
  env.monitor.start(args.gym_monitor)

if num_actuators == 1:
  def L(x):
    return K.exp(x)

  def P(x):
    return x*x

  def A(t):
    m, p, u = t
    return -(u - m)**2 * p

  def Q(t):
    v, a = t
    return v + a
else:
  def L(x):
    # TODO: batching
    #return T.nlinalg.alloc_diag(K.exp(T.nlinalg.ExtractDiag(view=True)(x))) + T.tril(x, k=-1)
    return K.exp(x)

  def P(x):
    return K.batch_dot(x, K.permute_dimensions(x, (0,2,1)))

  def A(t):
    m, p, u = t
    return -K.batch_dot(K.batch_dot(K.permute_dimensions(u - m, (0,2,1)), p), u - m)

  def Q(t):
    v, a = t
    return v + a

def createLayers():
  x = Input(shape=env.observation_space.shape, name='x')
  u = Input(shape=env.action_space.shape, name='u')
  if args.batch_norm:
    h = BatchNormalization()(x)
  else:
    h = x
  for i in xrange(args.layers):
    h = Dense(args.hidden_size, activation=args.activation, name='h'+str(i+1))(h)
    if args.batch_norm and i != args.layers - 1:
      h = BatchNormalization()(h)
  v = Dense(1, init='uniform', name='v')(h)
  m = Dense(num_actuators, init='uniform', name='m')(h)
  l = Dense(num_actuators**2, name='l0')(h)
  l = Reshape((num_actuators, num_actuators))(l)
  l = Lambda(L, output_shape=(num_actuators, num_actuators), name='l')(l)
  p = Lambda(P, output_shape=(num_actuators, num_actuators), name='p')(l)
  a = merge([m, p, u], mode=A, output_shape=(None, num_actuators,), name="a")
  q = merge([v, a], mode=Q, output_shape=(None, num_actuators,), name="q")
  return x, u, m, v, q

x, u, m, v, q = createLayers()

_mu = K.function([K.learning_phase(), x], m)
mu = lambda x: _mu([0] + [x])

model = Model(input=[x,u], output=q)
model.summary()

if args.optimizer == 'adam':
  optimizer = Adam(args.optimizer_lr)
elif args.optimizer == 'rmsprop':
  optimizer = RMSprop(args.optimizer_lr)
else:
  assert False
model.compile(optimizer=optimizer, loss='mse')

x, u, m, v, q = createLayers()

_V = K.function([K.learning_phase(), x], v)
V = lambda x: _V([0] + [x])

#q_f = K.function([x, u], q)

target_model = Model(input=[x,u], output=q)
target_model.set_weights(model.get_weights())

prestates = []
actions = []
rewards = []
poststates = []
terminals = []

total_reward = 0
for i_episode in xrange(args.episodes):
    observation = env.reset()
    #print "initial state:", observation
    episode_reward = 0
    for t in xrange(args.max_timesteps):
        if args.display:
          env.render()

        x = np.array([observation])
        u = mu(x)
        if args.noise_decay == 'linear':
          noise = 1. / (i_episode + 1)
        elif args.noise_decay == 'exp':
          noise = 10 ** -i_episode
        elif args.noise_decay == 'fixed':
          noise = args.fixed_noise
        else:
          assert False
        #print "noise:", noise
        action = u[0] + np.random.randn(num_actuators) * noise
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

        if len(prestates) > args.min_train:
          for k in xrange(args.train_repeat):
            if len(prestates) > args.batch_size:
              indexes = np.random.choice(len(prestates), size=args.batch_size)
            else:
              indexes = range(len(prestates))

            v = V(np.array(poststates)[indexes])
            y = np.array(rewards)[indexes] + args.gamma * np.squeeze(v)
            model.train_on_batch([np.array(prestates)[indexes], np.array(actions)[indexes]], y)

            weights = model.get_weights()
            target_weights = target_model.get_weights()
            for i in xrange(len(weights)):
              target_weights[i] = args.tau * weights[i] + (1 - args.tau) * target_weights[i]
            target_model.set_weights(target_weights)

        if done:
            break

    episode_reward = episode_reward / float(t + 1) 
    print "Episode {} finished after {} timesteps, average reward {}".format(i_episode + 1, t + 1, episode_reward)
    total_reward += episode_reward

print "Average reward per episode {}".format(total_reward / args.episodes)

if args.gym_monitor:
  env.monitor.close()

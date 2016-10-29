import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop
from keras.objectives import categorical_crossentropy, mse
from keras.utils import np_utils
import keras.backend as K
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.1)
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
parser.add_argument('--optimizer_lr', type=float)
#parser.add_argument('--batch_size', type=int, default=32)
#parser.add_argument('--repeat_train', type=int, default=1)
parser.add_argument('--average_over', type=int, default=1000)
parser.add_argument('--display', action='store_true', default=True)
parser.add_argument('--no_display', dest='display', action='store_false')
parser.add_argument('--gym_record')
parser.add_argument('environment')
args = parser.parse_args()

# create environment
env = gym.make(args.environment)
assert isinstance(env.observation_space, Box)
assert isinstance(env.action_space, Discrete)

# start recording for OpenAI Gym
if args.gym_record:
    env.monitor.start(args.gym_record)

# policy network
h = x = Input(shape=env.observation_space.shape)
for i in xrange(args.layers):
    h = Dense(args.hidden_size, activation=args.activation)(h)
y = Dense(env.action_space.n, activation='softmax')(h)

# baseline network
h = Dense(args.hidden_size, activation=args.activation)(h)
b = Dense(1)(h)

# advantage is an additional input to the network
R = Input(shape=(1,))
def policy_gradient_loss(l_sampled, l_predicted):
    return K.mean(K.stop_gradient(R - b) * categorical_crossentropy(l_sampled, l_predicted)[..., np.newaxis], axis=-1)

# create optimizer with optional learning rate parameter
if args.optimizer == 'adam':
    if args.optimizer_lr is None:
        optimizer = 'adam'
    else:
        optimizer = Adam(lr=args.optimizer_lr)
elif args.optimizer == 'rmsprop':
    if args.optimizer_lr is None:
        optimizer = 'rmsprop'
    else:
        optimizer = RMSprop(lr=args.optimizer_lr)
else:
    assert False

# inputs to the model are obesvation and advantage,
# outputs are action probabilities and baseline
model = Model(input=[x, R], output=[y, b])
model.summary()
# baseline is optimized with MSE
model.compile(optimizer=optimizer, loss=[policy_gradient_loss, 'mse'], loss_weights=[1, args.tau])

all_rewards = []
total_reward = 0
for i_episode in xrange(args.episodes):
    observations = []
    actions = []
    rewards = []
    #baselines = []

    observation = env.reset()
    episode_reward = 0
    for t in xrange(args.max_timesteps):
        if args.display:
            env.render()

        # create inputs for batch size 1
        x = np.array([observation])
        a = np.zeros((1, 1))
        # predict action probabilities (and baseline state value)
        y, b = model.predict([x, a], batch_size=1)
        y /= np.sum(y)  # ensure y-s sum up to 1
        #print "y:", y
        # sample action using those probabilities
        action = np.random.choice(env.action_space.n, p=y[0])
        #print "action:", action

        # record observation, action and baseline
        observations.append(observation)
        actions.append(action)
        #baselines.append(b[0,0])

        # make a step in environment
        observation, reward, done, info = env.step(int(action))
        episode_reward += reward
        #print "reward:", reward
        rewards.append(reward)

        if done:
            break

    # calculate discounted future rewards for this episode
    discounted_future_rewards = []
    g = 0
    for r in reversed(rewards):
        g = r + g * args.gamma
        discounted_future_rewards.insert(0, g)
    #print discounted_future_rewards
    all_rewards += discounted_future_rewards

    # form training data from observations, actions and rewards
    x = np.array(observations)
    y = np_utils.to_categorical(actions, env.action_space.n)
    r = np.array(discounted_future_rewards)
    #print x.shape, y.shape, r.shape, b.shape
    #print "x:", x
    #print "y:", y
    #print "r:", r
    #print "b:", b
    #print "a:", r - b
    # train the model, using discounted_future_rewards - baseline as advantage,
    # sampled actions as targets for actions and discounted_future_rewards as targets for baseline
    # the hope is the baseline is tracking average discounted_future_reward for this observation (state value)
    model.train_on_batch([x, r], [y, r])
 
    print "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward)
    total_reward += episode_reward

print "Average reward per episode {}".format(total_reward / args.episodes)

if args.gym_record:
    env.monitor.close()

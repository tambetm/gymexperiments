import argparse
import gym
from gym.spaces import Box, Discrete
from keras.models import Model
from keras.layers import Input, Dense
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_size', type=int, default=200)
parser.add_argument('--layers', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--episodes', type=int, default=200)
parser.add_argument('--max_timesteps', type=int, default=200)
parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
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
p = Dense(env.action_space.n, activation='softmax')(h)

# baseline network
h = Dense(args.hidden_size, activation=args.activation)(h)
b = Dense(1)(h)

# advantage is an additional input to the network
A = Input(shape=(1,))


def policy_gradient_loss(l_sampled, l_predicted):
    return A * categorical_crossentropy(l_sampled, l_predicted)[:, np.newaxis]

# inputs to the model are obesvation and advantage,
# outputs are action probabilities and baseline
model = Model(input=[x, A], output=[p, b])
model.summary()
# baseline is optimized with MSE
model.compile(optimizer='adam', loss=[policy_gradient_loss, 'mse'], loss_weights=[1, args.tau])

total_reward = 0
for i_episode in xrange(args.episodes):
    observations = []
    actions = []
    rewards = []
    baselines = []

    observation = env.reset()
    episode_reward = 0
    for t in xrange(args.max_timesteps):
        if args.display:
            env.render()

        # create inputs for batch size 1
        x = np.array([observation])
        A = np.zeros((1, 1))
        # predict action probabilities (and baseline state value)
        p, b = model.predict_on_batch([x, A])
        p /= np.sum(p)  # ensure y-s sum up to 1
        # sample action using those probabilities
        action = np.random.choice(env.action_space.n, p=p[0])

        # record observation, action and baseline
        observations.append(observation)
        actions.append(action)
        baselines.append(b[0, 0])

        # make a step in environment
        observation, reward, done, info = env.step(int(action))
        episode_reward += reward
        rewards.append(reward)

        if done:
            break

    # calculate discounted future rewards for this episode
    discounted_future_rewards = []
    g = 0
    for r in reversed(rewards):
        g = r + g * args.gamma
        discounted_future_rewards.insert(0, g)

    # form training data from observations, actions and rewards
    x = np.array(observations)
    y = np_utils.to_categorical(actions, env.action_space.n)
    R = np.array(discounted_future_rewards)
    b = np.array(baselines)
    A = (R - b)
    R = R[:, np.newaxis]
    A = A[:, np.newaxis]
    # train the model, using discounted_future_rewards - baseline as advantage,
    # sampled actions as targets for actions and discounted_future_rewards as targets for baseline
    # the hope is the baseline is tracking average discounted_future_reward for this observation (state value)
    model.train_on_batch([x, A], [y, R])

    print "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward)
    total_reward += episode_reward

print "Average reward per episode {}".format(total_reward / args.episodes)

if args.gym_record:
    env.monitor.close()

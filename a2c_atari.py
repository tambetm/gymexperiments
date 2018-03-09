import argparse
import os

import multiprocessing
from multiprocessing import Process, Queue, Array
import pickle

import gym
from gym.spaces import Box, Discrete

from keras.models import Model
from keras.layers import Input, TimeDistributed, Convolution2D, Flatten, LSTM, Dense
from keras.objectives import categorical_crossentropy
from keras.optimizers import Adam
from keras.utils import np_utils
import keras.backend as K
import numpy as np

from atari_utils import RandomizedResetEnv, AtariRescale42x42Env


def create_env(env_id):
    env = gym.make(env_id)
    env = RandomizedResetEnv(env)
    env = AtariRescale42x42Env(env)
    return env


def create_model(env, batch_size, num_steps):
    # network inputs are observations and advantages
    h = x = Input(batch_shape=(batch_size, num_steps) + env.observation_space.shape, name="x")
    A = Input(batch_shape=(batch_size, num_steps), name="A")

    # convolutional layers
    h = TimeDistributed(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same", activation='elu', dim_ordering='tf'), name='c1')(h)
    h = TimeDistributed(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same", activation='elu', dim_ordering='tf'), name='c2')(h)
    h = TimeDistributed(Convolution2D(32, 3, 3, subsample=(2, 2), border_mode="same", activation='elu', dim_ordering='tf'), name='c3')(h)
    h = TimeDistributed(Convolution2D(64, 3, 3, subsample=(2, 2), border_mode="same", activation='elu', dim_ordering='tf'), name='c4')(h)
    h = TimeDistributed(Flatten(), name="fl")(h)

    # recurrent layer
    h = LSTM(32, return_sequences=True, stateful=True, name="r1")(h)

    # policy network
    p = TimeDistributed(Dense(env.action_space.n, activation='softmax'), name="p")(h)

    # baseline network
    b = TimeDistributed(Dense(1), name="b")(h)

    # inputs to the model are observation and advantages,
    # outputs are action probabilities and baseline
    model = Model(input=[x, A], output=[p, b])

    # policy gradient loss and entropy bonus
    def policy_gradient_loss(l_sampled, l_predicted):
        return K.mean(A * categorical_crossentropy(l_sampled, l_predicted), axis=1) \
            - 0.01 * K.mean(categorical_crossentropy(l_predicted, l_predicted), axis=1)

    # baseline is optimized with MSE
    model.compile(optimizer='adam', loss=[policy_gradient_loss, 'mse'])

    return model


def predict(model, observation):
    # create inputs for batch (and timestep) of size 1
    x = np.array([[observation]])
    A = np.zeros((1, 1))  # dummy advantage
    # predict action probabilities (and baseline state value)
    p, b = model.predict_on_batch([x, A])
    # return action probabilities and baseline
    return p[0, 0], b[0, 0, 0]


def discount(rewards, terminals, v, gamma):
    # calculate discounted future rewards for this trajectory
    returns = []
    # start with the predicted value of the last state
    R = v
    for r, t in zip(reversed(rewards), reversed(terminals)):
        # if it was terminal state then restart from 0
        if t:
            R = 0
        R = r + R * gamma
        returns.insert(0, R)
    return returns


def runner(shared_buffer, fifo, num_timesteps, monitor, args):
    proc_name = multiprocessing.current_process().name
    print("Runner %s started" % proc_name)

    # local environment for runner
    env = create_env(args.env_id)

    # start monitor to record statistics and videos
    if monitor:
        env.monitor.start(args.env_id)

    # copy of model
    model = create_model(env, batch_size=1, num_steps=1)

    # record episode lengths and rewards for statistics
    episode_rewards = []
    episode_lengths = []
    episode_reward = 0
    episode_length = 0

    observation = env.reset()
    for i in range(num_timesteps // args.num_local_steps):
        # copy weights from main network at the beginning of iteration
        # the main network's weights are only read, never modified
        # but we create our own model instance, because Keras is not thread-safe
        model.set_weights(pickle.loads(shared_buffer.raw))

        observations = []
        actions = []
        rewards = []
        terminals = []
        baselines = []

        for t in range(args.num_local_steps):
            if args.display:
                env.render()

            # predict action probabilities (and baseline state value)
            p, b = predict(model, observation)

            # sample action using those probabilities
            p /= np.sum(p)  # ensure p-s sum up to 1
            action = np.random.choice(env.action_space.n, p=p)

            # log data
            observations.append(observation)
            actions.append(action)
            baselines.append(b)

            # step environment
            observation, reward, terminal, _ = env.step(int(action))
            rewards.append(reward)
            terminals.append(terminal)

            episode_reward += reward
            episode_length += 1

            # reset if terminal state
            if terminal:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                episode_reward = 0
                episode_length = 0
                observation = env.reset()

        # calculate discounted returns
        if terminal:
            # if the last was terminal state then start from 0
            returns = discount(rewards, terminals, 0, 0.99)
        else:
            # otherwise calculate the value of the last state
            _, v = predict(model, observation)
            returns = discount(rewards, terminals, v, 0.99)

        # convert to numpy arrays
        observations = np.array(observations)
        actions = np_utils.to_categorical(actions, env.action_space.n)
        baselines = np.array(baselines)
        returns = np.array(returns)
        advantages = returns - baselines

        # send observations, actions, rewards and returns. blocks if fifo is full.
        fifo.put((observations, actions, returns, advantages, episode_rewards, episode_lengths))
        episode_rewards = []
        episode_lengths = []

    if monitor:
        env.monitor.close()

    print("Runner %s finished" % proc_name)


def trainer(model, fifos, shared_buffer, args):
    proc_name = multiprocessing.current_process().name
    print("Trainer %s started" % proc_name)

    episode_rewards = []
    episode_lengths = []
    timestep = 0
    while len(multiprocessing.active_children()) > 0 and timestep < args.num_timesteps:
        batch_observations = []
        batch_actions = []
        batch_returns = []
        batch_advantages = []

        # loop over fifos from all runners
        for q, fifo in enumerate(fifos):
            # wait for a new trajectory and statistics
            observations, actions, returns, advantages, rewards, lengths = fifo.get()

            # add to batch
            batch_observations.append(observations)
            batch_actions.append(actions)
            batch_returns.append(returns)
            batch_advantages.append(advantages)

            # log statistics
            episode_rewards += rewards
            episode_lengths += lengths
            timestep += len(observations)

        # form training data from observations, actions and returns
        x = np.array(batch_observations)
        p = np.array(batch_actions)
        R = np.array(batch_returns)[:, :, np.newaxis]
        A = np.array(batch_advantages)

        # anneal learning rate
        model.optimizer.lr = max(0.001 * (args.num_timesteps - timestep) / args.num_timesteps, 0)

        # train the model
        total_loss, policy_loss, baseline_loss = model.train_on_batch([x, A], [p, R])

        # share model parameters
        shared_buffer.raw = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)

        if timestep % args.stats_interval == 0:
            print("Step %d/%d: episodes %d, mean episode reward %.2f, mean episode length %.2f." %
                (timestep, args.num_timesteps, len(episode_rewards), np.mean(episode_rewards), np.mean(episode_lengths)))
            episode_rewards = []
            episode_lengths = []

    print("Trainer %s finished" % proc_name)


def run(args):
    # create dummy environment to be able to create model
    env = create_env(args.env_id)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    print("Observation space: " + str(env.observation_space))
    print("Action space: " + str(env.action_space))

    # create main model
    model = create_model(env, batch_size=args.num_runners, num_steps=args.num_local_steps)
    model.summary()
    env.close()

    # for better compatibility with Theano and Tensorflow
    multiprocessing.set_start_method('spawn')

    # create shared buffer for sharing weights
    blob = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)
    shared_buffer = Array('c', len(blob))
    shared_buffer.raw = blob

    # force runner processes to use cpu, child processes inherit environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # create fifos and processes for all runners
    fifos = []
    for i in range(args.num_runners):
        fifo = Queue(args.queue_length)
        fifos.append(fifo)
        process = Process(target=runner,
            args=(shared_buffer, fifo, args.num_timesteps // args.num_runners, args.monitor and i == 0, args))
        process.start()

    # start trainer in main thread
    trainer(model, fifos, shared_buffer, args)

    print("All done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parallelization
    parser.add_argument('--num_runners', type=int, default=2)
    parser.add_argument('--queue_length', type=int, default=2)
    # how long
    parser.add_argument('--num_timesteps', type=int, default=5000000)
    parser.add_argument('--num_local_steps', type=int, default=20)
    parser.add_argument('--stats_interval', type=int, default=10000)
    # technical
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--monitor', action='store_true', default=False)
    # mandatory
    parser.add_argument('env_id')
    args = parser.parse_args()

    run(args)

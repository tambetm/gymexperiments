import argparse
import os
import gym
from gym.spaces import Box, Discrete

import multiprocessing
from multiprocessing import Process, Queue, Array
from queue import Empty
import pickle

from keras.models import Model
from keras.layers import Input, TimeDistributed, Dense
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
import keras.backend as K
import numpy as np


def create_model(env, args):
    h = x = Input(shape=(None,) + env.observation_space.shape, name="x")

    # policy network
    for i in range(args.layers):
        h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="h%d" % (i + 1))(h)
    p = TimeDistributed(Dense(env.action_space.n, activation='softmax'), name="p")(h)

    # baseline network
    h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="hb")(h)
    b = TimeDistributed(Dense(1), name="b")(h)

    # advantage is additional input
    A = Input(shape=(None,))

    # policy gradient loss and entropy bonus
    def policy_gradient_loss(l_sampled, l_predicted):
        return K.mean(A * categorical_crossentropy(l_sampled, l_predicted), axis=1) \
            - args.beta * K.mean(categorical_crossentropy(l_predicted, l_predicted), axis=1)

    # inputs to the model are observation and total reward,
    # outputs are action probabilities and baseline
    model = Model(input=[x, A], output=[p, b])

    # baseline is optimized with MSE
    model.compile(optimizer=args.optimizer, loss=[policy_gradient_loss, 'mse'])
    model.optimizer.lr = args.optimizer_lr

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


def runner(shared_buffer, fifo, args):
    # local environment for runner
    env = gym.make(args.environment)
    # copy of model
    model = create_model(env, args)

    # record episode lengths and rewards for statistics
    episode_rewards = []
    episode_lengths = []
    episode_reward = 0
    episode_length = 0

    observation = env.reset()
    for i in range(args.num_iterations):
        # copy weights from main network at the beginning of iteration
        # the main network's weights are only read, never modified
        # but we create our own model instance, because Keras is not thread-safe
        model.set_weights(pickle.loads(shared_buffer.raw))

        observations = []
        actions = []
        rewards = []
        terminals = []
        baselines = []

        for t in range(args.iteration_steps):
            if args.display:
                env.render()

            # predict action probabilities (and baseline state value)
            p, b = predict(model, observation)
            #print("b:", b)
            #print("p:", p)

            # sample action using those probabilities
            p /= np.sum(p)  # ensure p-s sum up to 1
            action = np.random.choice(env.action_space.n, p=p)
            #action = np.random.multinomial(1, p).argmax()
            #print("action:", action)

            # step environment and log data
            observations.append(observation)
            actions.append(action)
            baselines.append(b)
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
            returns = discount(rewards, terminals, 0, args.gamma)
        else:
            # otherwise calculate the value of the last state
            _, v = predict(model, observation)
            #print "v:", v
            returns = discount(rewards, terminals, v, args.gamma)

        # calculate advantages
        advantages = np.array(returns) - np.array(baselines)

        # send observations, actions, rewards and returns
        # block if fifo is full
        fifo.put((
            np.array(observations),
            np_utils.to_categorical(actions, env.action_space.n),
            np.array(returns),
            advantages,
            episode_rewards,
            episode_lengths
        ))
        episode_rewards = []
        episode_lengths = []


def trainer(model, fifos, shared_buffer, args):
    iteration = 0
    episode_rewards = []
    episode_lengths = []
    while len(multiprocessing.active_children()) > 0:
        batch_observations = []
        batch_actions = []
        batch_returns = []
        batch_advantages = []

        # loop over fifos from all runners
        for fifo in fifos:
            try:
                # wait for new trajectory
                observations, actions, returns, advantages, rewards, lengths = fifo.get(timeout=args.queue_timeout)

                # add to batch
                batch_observations.append(observations)
                batch_actions.append(actions)
                batch_returns.append(returns)
                batch_advantages.append(advantages)

                # log statistics
                episode_rewards += rewards
                episode_lengths += lengths

            except Empty:
                # just ignore empty fifos, batch will be smaller
                pass

        # if any of the runners produced trajectories
        if len(batch_observations) > 0:
            # form training data from observations, actions and returns
            x = np.array(batch_observations)
            p = np.array(batch_actions)
            R = np.array(batch_returns)
            A = np.array(batch_advantages)
            R = R[..., np.newaxis]

            # train the model
            total_loss, policy_loss, baseline_loss = model.train_on_batch([x, A], [p, R])

            # share model parameters
            shared_buffer.raw = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)

            iteration += 1

            if iteration % args.stats_interval == 0:
                print("Iter %d: episodes %d, mean episode reward %.2f, mean episode length %.2f." % (iteration, len(episode_rewards), np.mean(episode_rewards), np.mean(episode_lengths)))
                episode_rewards = []
                episode_lengths = []


def run(args):
    # create dummy environment to be able to create model
    env = gym.make(args.environment)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # create main model
    model = create_model(env, args)
    model.summary()
    env.close()

    # force runner processes to use cpu
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # for better compatibility with Theano and Tensorflow
    multiprocessing.set_start_method('spawn')

    # create shared buffer for sharing weights
    blob = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)
    shared_buffer = Array('c', len(blob))
    shared_buffer.raw = blob

    # create fifos and threads for all runners
    fifos = []
    for i in range(args.num_runners):
        fifo = Queue(args.queue_length)
        fifos.append(fifo)
        process = Process(target=runner, args=(shared_buffer, fifo, args))
        process.start()

    # start trainer in main thread
    trainer(model, fifos, shared_buffer, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
    # optimization
    parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
    parser.add_argument('--optimizer_lr', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--beta', type=float, default=0)
    # parallelization
    parser.add_argument('--num_runners', type=int, default=2)
    parser.add_argument('--queue_length', type=int, default=1)
    parser.add_argument('--queue_timeout', type=int, default=1)
    # how long
    parser.add_argument('--num_iterations', type=int, default=200)
    parser.add_argument('--iteration_steps', type=int, default=200)
    parser.add_argument('--stats_interval', type=int, default=10)
    # technical
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no_display', dest='display', action='store_false')
    parser.add_argument('--gym_record')
    parser.add_argument('environment')
    args = parser.parse_args()

    run(args)

import argparse
import gym
from gym.spaces import Box, Discrete

import multiprocessing
from multiprocessing import Process, Queue, Array
from queue import Empty
import cPickle as pickle

from keras.models import Model
from keras.layers import Input, Masking, TimeDistributed, Dense
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
import keras.backend as K
import numpy as np


def create_model(env, args):
    x = Input(shape=(None,) + env.observation_space.shape, name="x")
    # apply masking so that shorter episodes in batch can be padded
    # padded inputs must contain all zeros
    h = Masking()(x)

    # policy network
    for i in xrange(args.layers):
        h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="h%d" % (i + 1))(h)
    p = TimeDistributed(Dense(env.action_space.n, activation='softmax'), name="p")(h)

    # baseline network
    h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="hb")(h)
    b = TimeDistributed(Dense(1), name="b")(h)

    # total reward is additional input
    g = Input(shape=(None, 1))

    # policy gradient loss
    def policy_gradient_loss(l_sampled, l_predicted):
        return K.mean(K.stop_gradient(g - b) * categorical_crossentropy(l_sampled, l_predicted)[..., np.newaxis], axis=-1)

    # inputs to the model are observation and total reward,
    # outputs are action probabilities and baseline
    model = Model(input=[x, g], output=[p, b])

    # baseline is optimized with MSE
    model.compile(optimizer=args.optimizer, loss=[policy_gradient_loss, 'mse'], loss_weights=[1, args.tau])
    model.optimizer.lr = args.optimizer_lr

    return model


def predict(model, observation):
    # create inputs for batch (and timestep) of size 1
    x = np.array([[observation]])
    g = np.zeros((1, 1, 1))  # dummy return
    # predict action probabilities (and baseline state value)
    y, b = model.predict([x, g], batch_size=1)
    # return action probabilities and baseline
    return y[0, 0], b[0, 0, 0]


def discount(rewards, g, gamma):
    # calculate discounted future rewards for this episode
    returns = []
    for r in reversed(rewards):
        g = r + g * gamma
        returns.insert(0, g)
    #print returns
    return returns


def runner(shared_buffer, fifo, args):
    # local environment for runner
    env = gym.make(args.environment)
    # copy of model
    model = create_model(env, args)

    done = True
    for episode in range(args.max_episodes):
        # copy weights from main network at the beginning of episode
        # the main network's weights are only read, never modified
        # but we create our own model instance, because Keras is not thread-safe
        model.set_weights(pickle.loads(shared_buffer.raw))

        observations = []
        actions = []
        rewards = []

        # don't need this because of autoreset?
        if done:
            observation = env.reset()
        for t in xrange(args.max_timesteps):
            if args.display:
                env.render()

            # predict action probabilities (and baseline state value)
            p, b = predict(model, observation)
            #print "b:", b[0][0]
            #print "p:", p[0][0]

            # sample action using those probabilities
            p /= np.sum(p)  # ensure p-s sum up to 1
            action = np.random.choice(env.action_space.n, p=p)
            #print "action:", action

            # step environment and log data
            observations.append(observation)
            observation, reward, done, info = env.step(int(action))
            actions.append(action)
            rewards.append(reward)

            # stop if terminal state
            if done:
                break

        # calculate discounted returns
        if done:
            # if terminal state then start from 0
            returns = discount(rewards, 0, args.gamma)
        else:
            # otherwise calculate the value of the last state
            _, v = predict(model, observation)
            #print "v:", v
            returns = discount(rewards, v, args.gamma)

        # send observations, actions, rewards and returns
        # block if fifo is full
        fifo.put((observations, actions, rewards, returns))


def trainer(model, fifos, shared_buffer, args):
    step = 0
    while len(multiprocessing.active_children()) > 0:
        batch_observations = []
        batch_actions = []
        batch_returns = []
        episode_rewards = []
        episode_lengths = []
        maxlen = 0

        # loop over fifos from all runners
        for fifo in fifos:
            try:
                # wait for new episode
                observations, actions, rewards, returns = fifo.get(timeout=args.queue_timeout)

                # add to batch
                batch_observations.append(np.array(observations))
                batch_actions.append(np_utils.to_categorical(actions, env.action_space.n))
                batch_returns.append(np.array(returns))

                # log max episode length
                if len(observations) > maxlen:
                    maxlen = len(observations)

                # log statistics
                episode_rewards.append(sum(rewards))
                episode_lengths.append(len(rewards))

            except Empty:
                # just ignore empty fifos, batch will be smaller
                pass

        # if any of the runners produced episodes
        if maxlen > 0:
            print "Step %d: batch size %d, mean episode reward %.2f, mean episode length %.2f." % (step, len(episode_rewards), np.mean(episode_rewards), np.mean(episode_lengths))

            # pad all episodes to be of the same length
            for a in batch_observations:
                a.resize((maxlen,) + a.shape[1:], refcheck=False)
            for a in batch_actions:
                a.resize((maxlen,) + a.shape[1:], refcheck=False)
            for a in batch_returns:
                a.resize((maxlen,) + a.shape[1:], refcheck=False)

            # form training data from observations, actions and returns
            x = np.array(batch_observations)
            p = np.array(batch_actions)
            g = np.array(batch_returns)
            g = g[..., np.newaxis]
            #print x.shape, y.shape, r.shape, b.shape
            #print "x:", x
            #print "y:", y
            #print "g:", g

            # train the model
            total_loss, policy_loss, baseline_loss = model.train_on_batch([x, g], [p, g])
            #print "total_loss:", total_loss
            #print "policy_loss:", policy_loss
            #print "baseline_loss:", baseline_loss
            # share model parameters
            shared_buffer.raw = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)

            step += 1

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
    parser.add_argument('--tau', type=float, default=0.1)
    # parallelization
    parser.add_argument('--num_runners', type=int, default=2)
    parser.add_argument('--queue_length', type=int, default=2)
    parser.add_argument('--queue_timeout', type=int, default=10)
    # how long
    parser.add_argument('--max_episodes', type=int, default=200)
    parser.add_argument('--max_timesteps', type=int, default=200)
    # technical
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no_display', dest='display', action='store_false')
    parser.add_argument('--gym_record')
    parser.add_argument('environment')
    args = parser.parse_args()

    # create dummy environment to be able to create model
    env = gym.make(args.environment)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)

    # create main model
    model = create_model(env, args)
    model.summary()

    # create shared buffer for sharing weights
    blob = pickle.dumps(model.get_weights(), pickle.HIGHEST_PROTOCOL)
    shared_buffer = Array('c', len(blob))
    shared_buffer.raw = blob
    env.close()

    # create fifos and threads for all runners
    fifos = []
    for i in range(args.num_runners):
        fifo = Queue(args.queue_length)
        fifos.append(fifo)
        process = Process(target=runner, args=(shared_buffer, fifo, args))
        process.start()

    trainer(model, fifos, shared_buffer, args)

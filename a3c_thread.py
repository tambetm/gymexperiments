import argparse
import gym
from gym.spaces import Box, Discrete

import threading
from threading import Thread, Lock
from queue import Queue, Empty

from keras.models import Model
from keras.layers import Input, Dense, Masking, TimeDistributed
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
import keras.backend as K
import numpy as np


def create_model(env):
    x = Input(shape=(None,) + env.observation_space.shape, name="x")
    # apply masking so that shorter episodes in batch can be padded
    # padded inputs must contain all zeros
    h = Masking()(x)

    # policy network
    for i in xrange(args.layers):
        h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="h%d" % (i + 1))(h)
    y = TimeDistributed(Dense(env.action_space.n, activation='softmax'), name="y")(h)

    # baseline network
    h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="hb")(h)
    b = TimeDistributed(Dense(1), name="b")(h)

    # total reward is additional input
    R = Input(shape=(None, 1))

    # policy gradient loss
    def policy_gradient_loss(l_sampled, l_predicted):
        return K.mean(K.stop_gradient(R - b) * categorical_crossentropy(l_sampled, l_predicted)[..., np.newaxis], axis=-1)

    # inputs to the model are observation and total reward,
    # outputs are action probabilities and baseline
    model = Model(input=[x, R], output=[y, b])

    # baseline is optimized with MSE
    model.compile(optimizer=args.optimizer, loss=[policy_gradient_loss, 'mse'], loss_weights=[1, args.tau])
    model.optimizer.lr = args.optimizer_lr

    return model


def runner(main_model, weightlock, fifo):
    # local environment for runner
    env = gym.make(args.environment)
    # copy of model
    model = create_model(env)

    done = True
    for episode in range(args.max_episodes):
        # copy weights from main network at the beginning of episode
        # the main network's weights are only read, never modified
        # but we create our own model instance, because Keras is not thread-safe
        with weightlock:
            weights = main_model.get_weights()
        model.set_weights(weights)

        observations = []
        actions = []
        rewards = []

        # don't need this because of autoreset?
        if done:
            observation = env.reset()
        for t in xrange(args.max_timesteps):
            if args.display:
                env.render()

            # create inputs for batch (and timestep) of size 1
            x = np.array([[observation]])
            R = np.zeros((1, 1, 1))  # dummy return
            # predict action probabilities (and baseline state value)
            y, b = model.predict([x, R], batch_size=1)
            #print "b:", b[0][0]
            #print "y:", y[0][0]

            # sample action using those probabilities
            y /= np.sum(y)  # ensure y-s sum up to 1
            action = np.random.choice(env.action_space.n, p=y[0][0])
            #print "action:", action

            # step environment and log data
            observations.append(observation)
            observation, reward, done, info = env.step(int(action))
            actions.append(action)
            rewards.append(reward)

            # stop if terminal state
            if done:
                break

        # send observations, actions and rewards
        # block if fifo is full
        fifo.put((observations, actions, rewards, done, observation))


def discount(rewards, g=0):
    # calculate discounted future rewards for this episode
    returns = []
    for r in reversed(rewards):
        g = r + g * args.gamma
        returns.insert(0, g)
    #print returns
    return returns


def trainer(model, weightlock, fifos):
    step = 0
    while threading.active_count() > 1:
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
                observations, actions, rewards, done, last_obs = fifo.get(timeout=args.queue_timeout)
                # calculate discounted returns
                if done:
                    # if terminal state then start from 0
                    returns = discount(rewards, 0)
                else:
                    # otherwise calculate the value of the last state
                    x = np.array([[last_obs]])
                    R = np.zeros((1, 1, 1))  # dummy return
                    # predict action probabilities (and baseline state value)
                    _, b = model.predict([x, R], batch_size=1)
                    #print "v:", b[0][0, 0]
                    returns = discount(rewards, b[0, 0, 0])

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
            y = np.array(batch_actions)
            R = np.array(batch_returns)
            R = R[..., np.newaxis]
            #print x.shape, y.shape, r.shape, b.shape
            #print "x:", x
            #print "y:", y
            #print "R:", R

            # train the model, prevent reading weights while doing this
            with weightlock:
                total_loss, policy_loss, baseline_loss = model.train_on_batch([x, R], [y, R])
            #print "total_loss:", total_loss
            #print "policy_loss:", policy_loss
            #print "baseline_loss:", baseline_loss

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
    parser.add_argument('--queue_length', type=int, default=5)
    parser.add_argument('--queue_timeout', type=int, default=100)
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
    model = create_model(env)
    model.summary()
    weightlock = Lock()
    env.close()

    # create fifos and threads for all runners
    fifos = []
    for i in range(args.num_runners):
        fifo = Queue(args.queue_length)
        fifos.append(fifo)
        thread = Thread(target=runner, args=(model, weightlock, fifo))
        thread.start()

    trainer(model, weightlock, fifos)

import argparse
import gym
from gym.spaces import Box, Discrete

import threading
from threading import Thread
from queue import Queue

from keras.models import Model
from keras.layers import Input, Dense, Masking, TimeDistributed
from keras.optimizers import Adam, RMSprop
from keras.objectives import categorical_crossentropy
from keras.utils import np_utils
import keras.backend as K
import numpy as np

def create_model(env):
    x = Input(shape=(None,) + env.observation_space.shape, name="x")
    h = Masking()(x)
    for i in xrange(args.layers):
      h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="h%d" % (i + 1))(h)
    y = TimeDistributed(Dense(env.action_space.n, activation='softmax'), name="y")(h)

    # baseline network
    h = TimeDistributed(Dense(args.hidden_size, activation=args.activation), name="hb")(h)
    b = TimeDistributed(Dense(1), name="b")(h)

    # total reward is additional input
    R = Input(shape=(None, 1,))
    def policy_gradient_loss(l_sampled, l_predicted):
        return K.mean(K.stop_gradient(R - b) * categorical_crossentropy(l_sampled, l_predicted)[..., np.newaxis], axis=-1)

    # inputs to the model are observation and total reward,
    # outputs are action probabilities and baseline
    model = Model(input=[x, R], output=[y, b])
    # baseline is optimized with MSE
    model.compile(optimizer=args.optimizer, loss=[policy_gradient_loss, 'mse'], loss_weights=[1, args.tau])

    return model

def runner(main_model, fifo):
    env = gym.make(args.environment)
    model = create_model(env)

    for i_episode in xrange(args.episodes):
        weights = main_model.get_weights()
        model.set_weights(weights)

        episode_reward = 0
        observation = env.reset()
        for t in xrange(args.max_timesteps):
            if args.display:
              env.render()

            # create inputs for batch size 1
            x = np.array([[observation]])
            R = np.zeros((1,1,1))
            # predict action probabilities (and baseline state value)
            y, b = model.predict([x, R], batch_size=1)
            y = y[0]
            b = b[0]
            #print "b:", b
            y /= np.sum(y)  # ensure y-s sum up to 1
            #print "y:", y
            # sample action using those probabilities
            action = np.random.choice(env.action_space.n, p=y[0])
            #print "action:", action
            new_observation, reward, done, info = env.step(int(action))

            fifo.put((observation, action, reward, done))
            observation = new_observation
            episode_reward += reward

            if done:
                break

        print "Episode {} finished after {} timesteps, episode reward {}".format(i_episode + 1, t + 1, episode_reward)

def discount(rewards):
    # calculate discounted future rewards for this episode
    discounted_future_rewards = []
    g = 0
    for r in reversed(rewards):
        g = r + g * args.gamma
        discounted_future_rewards.insert(0, g)
    #print discounted_future_rewards
    return discounted_future_rewards

def trainer(model, fifos):
    while threading.active_count() > 1:
        print threading.active_count()
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        maxlen = 0
        for fifo in fifos:
            observations = []
            actions = []
            rewards = []
            
            done = False
            while not done:
                observation, action, reward, done = fifo.get()
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)
            rewards = discount(rewards)

            batch_observations.append(np.array(observations))
            batch_actions.append(np_utils.to_categorical(actions, env.action_space.n))
            batch_rewards.append(np.array(rewards))

            if len(observations) > maxlen:
                maxlen = len(observations)

        # pad all episodes to be the same length
        for a in batch_observations:
            a.resize((maxlen,) + a.shape[1:], refcheck=False)
        for a in batch_actions:
            a.resize((maxlen,) + a.shape[1:], refcheck=False)
        for a in batch_rewards:
            a.resize((maxlen,) + a.shape[1:], refcheck=False)

        # form training data from observations, actions and rewards
        x = np.array(batch_observations)
        y = np.array(batch_actions)
        R = np.array(batch_rewards)
        R = R[..., np.newaxis]
        #print x.shape, y.shape, r.shape, b.shape
        #print "x:", x
        #print "y:", y
        #print "r:", r
        # train the model, using discounted_future_rewards - baseline as advantage
        total_loss, policy_loss, baseline_loss = model.train_on_batch([x, R], [y, R])
        #print "total_loss:", total_loss
        #print "policy_loss:", policy_loss
        #print "baseline_loss:", baseline_loss
        #model.save_weights('temp.hdf5')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_size', type=int, default=200)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--max_timesteps', type=int, default=200)
    parser.add_argument('--activation', choices=['tanh', 'relu'], default='tanh')
    parser.add_argument('--optimizer', choices=['adam', 'rmsprop'], default='adam')
    #parser.add_argument('--optimizer_lr', type=float, default=0.01)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--no_display', dest='display', action='store_false')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--queue_length', type=int, default=200)
    parser.add_argument('--gym_record')
    parser.add_argument('environment')
    args = parser.parse_args()

    env = gym.make(args.environment)
    assert isinstance(env.observation_space, Box)
    assert isinstance(env.action_space, Discrete)
    model = create_model(env)
    model.summary()

    fifos = []
    for i in range(args.num_workers):
        fifo = Queue(args.queue_length)
        fifos.append(fifo)
        thread = Thread(target=runner, args=(model, fifo))
        thread.start()

    trainer(model, fifos)

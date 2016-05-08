import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression

class TimeBuffer:
  def __init__(self, max_timesteps, max_episodes, observation_shape, action_shape):
    self.max_timesteps = max_timesteps
    self.max_episodes = max_episodes
    self.observation_shape = observation_shape
    self.action_shape = action_shape

    self.preobs = np.empty((self.max_timesteps, self.max_episodes) + observation_shape)
    self.actions = np.empty((self.max_timesteps, self.max_episodes) + action_shape)
    self.rewards = np.empty((self.max_timesteps, self.max_episodes))
    self.postobs = np.empty((self.max_timesteps, self.max_episodes) + observation_shape)
    self.terminals = np.empty((self.max_timesteps, self.max_episodes), dtype = np.bool)
    self.lengths = np.zeros(self.max_episodes, np.uint)
    
    self.num_episodes = 0
    self.episode = 0
    self.timestep = 0

  def add(self, preobs, action, reward, postobs, terminal):
    assert preobs.shape == self.observation_shape
    assert action.shape == self.action_shape
    assert postobs.shape == self.observation_shape
    self.preobs[self.timestep, self.episode] = preobs
    self.actions[self.timestep, self.episode] = action
    self.rewards[self.timestep, self.episode] = reward
    self.postobs[self.timestep, self.episode] = postobs
    self.terminals[self.timestep, self.episode] = terminal
    self.timestep += 1
 
  def sample(self, batch_size, max_timestep):
    episodes = []
    timesteps = []
    for i in xrange(batch_size):
      episode = np.random.choice(self.num_episodes)
      timestep = np.random.choice(min(self.lengths[episode], max_timestep))
      episodes.append(episode)
      timesteps.append(timestep)
    #return self.preobs[indexes], self.actions[indexes], self.rewards[indexes], self.postobs[indexes], timesteps
    return self.postobs[timesteps, episodes], np.array(timesteps)

  def new_episode(self):
    self.lengths[self.episode] = self.timestep
    self.episode += 1
    self.timestep = 0
    self.num_episodes = self.episode

  def reset(self):
    self.num_episodes = 0
    self.episode = 0
    self.timestep = 0
    self.lengths *= 0

  def is_full(self):
    return self.num_episodes == self.max_episodes

class IRModel:
  def __init__(self, max_timesteps):
    self.max_timesteps = max_timesteps

    self.obsmodels = []
    self.obscovs = []
    self.rewmodels = []
    self.termmodels = []

  def fit(self, preobs, actions, rewards, postobs, terminals, lengths):
    self.obsmodels = []
    self.obscovs = []
    self.rewmodels = []
    self.termmodels = []
    for t in xrange(self.max_timesteps):
      episodes = lengths > t
      if sum(episodes) < 2:
        break
      
      # fit observation/state model
      X = np.concatenate([preobs[t, episodes], actions[t, episodes]], axis=1)
      Y = postobs[t, episodes]
      obsmodel = LinearRegression().fit(X, Y)
      self.obsmodels.append(obsmodel)
      Yhat = obsmodel.predict(X)
      obscov = np.cov(Y - Yhat, rowvar=0)
      self.obscovs.append(obscov)

      # fit reward model
      Y = rewards[t, episodes]
      rewmodel = LinearRegression().fit(X, Y)
      self.rewmodels.append(rewmodel)

      # fit terminal model
      #Y = terminals[t, episodes]
      #termmodel = LogisticRegression().fit(X, Y)
      #self.termmodels.append(termmodel)

  def predict(self, preobs, actions, timesteps):
    postobs = []
    rewards = []
    terminals = []
    for preob, action, timestep in zip(preobs, actions, timesteps):
      # predict next observation
      X = np.concatenate((preob, action), axis=0)
      obsmodel = self.obsmodels[timestep]
      obsmeans = obsmodel.predict(X)[0]
      obscov = self.obscovs[timestep]
      postob = np.random.multivariate_normal(obsmeans, obscov)
      postobs.append(postob)

      # predict reward
      rewmodel = self.rewmodels[timestep]
      reward = rewmodel.predict(X)[0]
      rewards.append(reward)

      # predict terminal
      #termmodel = self.termmodels[timestep]
      #terminal = termmodel.predict(X)[0]
      #terminals.append(terminal)
      terminals.append(False)

    return np.stack(postobs), np.stack(rewards), np.stack(terminals)

  def supported_timesteps(self):
    return len(self.obsmodels)

if __name__ == "__main__":
  obs_shape = (3,)
  act_shape = (2,)
  buf = TimeBuffer(10, 4, obs_shape, act_shape)
  for i in xrange(7):
    buf.add(i*np.ones(obs_shape), np.ones(act_shape), 1.0, (i+1)*np.ones(obs_shape), False)
  assert buf.timestep == 7
  buf.add(7*np.ones(obs_shape), np.ones(act_shape), 1.0, 8*np.ones(obs_shape), True)
  assert buf.num_episodes == 1
  assert buf.timestep == 0
  assert buf.lengths[0] == 8

  for i in xrange(5):
    buf.add((i+1)*np.ones(obs_shape), np.ones(act_shape), 1.0, (i+2)*np.ones(obs_shape), False)
  assert buf.timestep == 5
  buf.add(6*np.ones(obs_shape), np.ones(act_shape), 1.0, 7*np.ones(obs_shape), True)
  assert buf.num_episodes == 2
  assert buf.timestep == 0
  assert buf.lengths[1] == 6

  preobs, timesteps = buf.sample(3)
  assert len(preobs) == 3
  assert len(timesteps) == 3
  assert preobs[0].shape == obs_shape

  mdl = IRModel(10)
  mdl.fit(buf.preobs, buf.actions, buf.rewards, buf.postobs, buf.terminals, buf.lengths)
  assert len(mdl.obsmodels) == 6

  timesteps = [0,1,2]
  postobs, rewards, terminals = mdl.predict(preobs, np.ones((3,2)), timesteps)
  assert np.all(postobs - (preobs + 1) < 0.00001)

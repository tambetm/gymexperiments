from collections import deque
from gym import Wrapper, ObservationWrapper, ActionWrapper
from gym.spaces.box import Box
import numpy as np
import cv2


def _process_frame42(frame):
    reshaped_screen = np.reshape(frame, [210, 160, 3]).astype(np.float32).mean(2)
    resized_screen = cv2.resize(reshaped_screen, (84, 110))
    x_t = resized_screen[18:102, :]
    x_t = cv2.resize(x_t, (42, 42))
    x_t *= (1.0 / 255.0)
    x_t = np.reshape(x_t, [42, 42, 1])
    return x_t


class AtariRescale42x42Env(ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale42x42Env, self).__init__(env)
        self.observation_space = Box(0, 255, [42, 42, 1])

    def _observation(self, observation):
        return _process_frame42(observation)


def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    img = img[:, :, 0] * 0.2126 + img[:, :, 1] * 0.0722 + img[:, :, 2] * 0.7152
    resized_screen = cv2.resize(img, (84, 110),  interpolation=cv2.INTER_LINEAR)
    x_t = resized_screen[18:102, :]
    x_t /= 255.0
#    x_t -= 0.5
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t


class AtariRescale84x84Env(ObservationWrapper):
    def __init__(self, env=None):
        super(AtariRescale84x84Env, self).__init__(env)
        self.observation_space = Box(0, 255, [84, 84, 1])

    def _observation(self, observation):
        return _process_frame84(observation)


class RandomizedResetEnv(Wrapper):
    def __init__(self, env, no_op_max=7):
        super(RandomizedResetEnv, self).__init__(env)
        self._no_op_max = no_op_max

    def _reset(self):
        ob = self.env.reset()
        action = 0

        # randomize initial state
        if self._no_op_max > 0:
            no_op = np.random.randint(0, self._no_op_max + 1)
            for _ in range(no_op):
                ob, _, _, _ = self.env.step(action)
        return ob


class OneLiveResetEnv(Wrapper):
    def _step(self, action):
        lives = self.env.unwrapped.ale.lives()
        observation, reward, done, info = self.env.step(action)
        if lives != self.env.unwrapped.ale.lives():
            done = True
        return observation, reward, done, info


class UnstuckPolicyEnv(ActionWrapper):
    actions = deque(maxlen=30)

    def _action(self, action):
        if self.actions.count(action) == 30:
            action = 1
        self.actions.append(action)
        return action

    def _reverse_action(self, action):
        return action


class ObservationBuffer(Wrapper):
    def __init__(self, env, buffer_size=4):
        super(ObservationBuffer, self).__init__(env)
        self.buffer_size = buffer_size
        self.buffer = deque(maxlen=self.buffer_size)
        assert len(self.env.observation_space.shape) == 3
        self._shape = list(self.env.observation_space.shape)
        self._num_channels = self._shape[2]
        self._shape[2] *= self.buffer_size
        self.observation_space = Box(-0.5, 0.5, self._shape)

    def _step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.buffer.append(observation)
        return np.concatenate(self.buffer, axis=2), reward, done, info

    def _reset(self):
        obs = self.env.reset()
        for _ in range(self.buffer_size):
            self.buffer.append(obs)
        return np.concatenate(self.buffer, axis=2)

#    def _render(self, mode='human', close=False):
#        if mode == "rgb_array":
#            return self.buffer[-1]
#        return self.env.render(mode, close)

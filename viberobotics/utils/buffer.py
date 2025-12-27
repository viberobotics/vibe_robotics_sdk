import numpy as np

class Buffer:

    def __init__(self, obs_dim, horizon):
        self.obs_dim = obs_dim
        self.ptr = 0
        self.horizon = horizon

        self.obs = np.zeros((horizon, obs_dim))

    def add(self, obs):
        if self.ptr >= self.horizon:
            self.obs[:-1] = self.obs[1:]
            self.obs[-1] = obs
        else:
            self.obs[self.ptr] = obs
            self.ptr += 1

    def get(self):
        return np.reshape(self.obs, (self.obs_dim * self.horizon,)) # oldest obs first
        # return self.obs[::-1].reshape(self.obs_dim * self.horizon) # newest obs first

    def is_full(self):
        return self.ptr >= self.horizon
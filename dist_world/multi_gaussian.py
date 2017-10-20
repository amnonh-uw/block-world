import numpy as np
import gym

class MultiGaussian(gym.Space):
    def __init__(self, dims):
        self.dims = dims
        self.shape = tuple([self.dims] * 2)

    def to_continous(self, a):
        new_sample = np.zeros(self.dims)
        for i in range(self.dims):
            new_sample[i] = np.random.normal(a[i*2], a[i*2+1])

        return new_sample
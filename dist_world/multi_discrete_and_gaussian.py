import numpy as np
import gym

class MultiDiscreteAndGaussian(gym.Space):
    def __init__(self, array_of_param_array, dims):
        self.dims = dims
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n_discrete = np.prod(self.high - self.low + 1)
        self.shape = tuple(1 + [self.dims] * 2)

    def to_continous_action(self, a):
        new_sample = np.zeros(self.dims)
        for i in range(1,self.self.dims+1):
            new_sample[i] = np.random.normal(a[i*2], a[i*2+1])

        return new_sample

    def to_discrete_action(self, a):
        return a[0]
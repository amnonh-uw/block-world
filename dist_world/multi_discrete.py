import numpy as np
import tensorflow as tf

import gym
from gym.spaces import prng

class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)

    Note: A value of 0 always need to represent the NOOP action.

    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:

        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1

    - Can be initialized as

        MultiDiscrete([ [0,4], [0,1], [0,1] ])

    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = int(np.prod(self.high - self.low + 1))
        self.dtype = tf.int32

    def to_discrete(self, a):
        return a

    @property
    def shape(self):
        return self.num_discrete_space


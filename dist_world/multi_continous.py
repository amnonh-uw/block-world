import gym
import tensorflow as tf

class MultiContinuous(gym.Space):
    def __init__(self, dims):
        self.dims = dims
        self.shape = tuple([self.dims])
        self.dtype =  tf.float32
    

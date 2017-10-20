import gym

class MultiContinuous(gym.Space):
    def __init__(self, length):
        self.length = length
        self.shape = tuple([self.length])
    

import tensorflow as tf
from policies.base import DaggerPolicyBase

#
#
# Empty policy
# only used to generate expert dataset
#
#
#
class DaggerPolicy(DaggerPolicyBase):
    def __init__(self, dir_name):
        super().__init__(dir_name)

    def print_batch(self, batch):
        pass

    def policy_initializer(self):
        pass

    def get_output(self):
        return None
        return tf.constant(1)

    def get_loss(self):
        None


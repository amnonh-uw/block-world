import tensorflow as tf
import numpy as np
from lib.networks.network import Network
from policies.base import DaggerPolicyBase

#
#
# Learn to transform from image space coords (pixel x, pixel y, distance from cam)
# to world coordinates
#
#

class DaggerPolicy(DaggerPolicyBase):
    def __init__(self, dir_name):
        super().__init__(dir_name)
        self.positions = tf.placeholder(tf.float32, shape=[None, 6], name='screen_positions')
        self.action = tf.placeholder(tf.float32, name="action", shape=(None, 3))

        l1 = tf.contrib.layers.fully_connected(inputs=self.positions, num_outputs=256)
        l2 = tf.contrib.layers.fully_connected(inputs=l1, num_outputs=256)
        l3 = tf.contrib.layers.fully_connected(inputs=l2, num_outputs=256)
        l4 = tf.contrib.layers.fully_connected(inputs=l3, num_outputs=256, activation_fn=None)
        self.predicted_action = tf.contrib.layers.fully_connected(inputs=l4, num_outputs=3, activation_fn=None)

    @staticmethod
    def print_results(obs, action):
        pos1 = obs['finger_pos']
        pos2 = obs['target_pos']

        print('finger_pos {} target_pos {} action {}'.format(pos1, pos2, action))

    @staticmethod
    def train_sample_from_dict(sample_dict):
        #
        # This method must use tensorflow primitives
        #
        pos1 = sample_dict['finger_screen_pos']
        pos2 = sample_dict['target_screen_pos']

        positions = tf.concat((pos1, pos2), axis=0, name='concat_positions')
        action = sample_dict['action']

        return (positions, action)

    @staticmethod
    def eval_sample_from_dict(sample_dict):
        #
        # this method must use numpy primitives
        #

        pos1 = sample_dict['finger_screen_pos']
        pos2 = sample_dict['target_screen_pos']
        positions = np.concatenate((pos1, pos2), axis=0)
        positions = np.expand_dims(positions, axis=0)

        return (positions,)

    def loss_feed_dict(self, batch):
        return {
            self.positions: batch[0],
            self.action: batch[1]}

    def eval_feed_dict(self, obs_dict):
        sample = self.eval_sample_from_dict(obs_dict)
        return {
            self.positions: sample[0]
        }

    def print_batch(self, batch):
        pass

    def get_output(self):
        return self.predicted_action

    def get_loss(self):
        return tf.losses.mean_squared_error(self.action, self.predicted_action)

    def policy_initializer(self):
        pass

import os
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from dagger_policy_base import DaggerPolicyBase

class DaggerPolicy(DaggerPolicyBase):
    width = 224
    height = 224
    def __init__(self, dir_name):
        super().__init__(dir_name)
        self.img1 = tf.placeholder(tf.float32, shape=[None, width, height, 3], name='img1')
        self.positions = tf.placeholder(tf.float32, shape=[None, 2], name='position')

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):

            conv1 = slim.conv2d(self.img1, 64, [3, 3], padding='VALID', scope='conv1')
            conv2 = slim.conv2d(conv1, 64, [3, 3], padding='VALID', scope='conv2')
            flat1 = slim.flatten(conv1)
            fc1 = slim.fully_connected(flat1, 128, scope='fc1')
            self.predicted_positions = slim.fully_connected(fc1, 2, scope='fc')

    def train_sample_from_dict(self, sample_dict):
        #
        # This method must use tensorflow primitives
        #
        img1 = sample_dict['centercam']
        img1 = tf.slice(img1, [0,0,0], [-1,-1,3])
        img1 = tf.cast(img1, tf.float32)
        img1 = self.tf_resize(img1, DaggerPolicy.width, DaggerPolicy.height)

        pos1 = tf.slice(sample_dict['finger_screen_pos'], [0], [2])
        # pos2 = tf.slice(sample_dict['target_screen_pos'], [0], [2])

        positions = pos1

        return (img1, positions)

    def eval_sample_from_dict(self, sample_dict):
        #
        # this method must use numpy primitives
        #
        img1 = sample_dict['centercam']
        img1 = self.im_resize(img1, DaggerPolicy.width, DaggerPolicy.height)
        img1 = np.asarray(img1)
        img1 = img1[:,:,0:3]
        img1 = np.expand_dims(img1, axis=0)

        return (img1,)

    def loss_feed_dict(self, batch):
        return {
            self.img1: batch[0],
            self.positions: batch[1]}

    def eval_feed_dict(self, obs_dict):
        sample = self.eval_sample_from_dict(obs_dict)
        return {
            self.img1: sample[0]
        }

    def get_output(self):
        return self.predicted_positions

    def get_loss(self):
        return tf.losses.mean_squared_error(self.positions, self.predicted_positions)

    def print_batch(self, batch):
        imgs = batch[0]
        positions = batch[1]
        for i in range(imgs.shape[0]):
            pass

    def policy_initializer(self):
        pass

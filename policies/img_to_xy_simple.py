import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from policies.base import DaggerPolicyBase
from skimage import img_as_float

#
#
# Learn to transform from image to x,y position in image of finger
# using a simple convolutional network
#
#

class DaggerPolicy(DaggerPolicyBase):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def build_graph(self, dir_name):
        super().build_graph(dir_name)
        self.img1 = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='img1')
        self.positions = tf.placeholder(tf.float32, shape=[None, 2], name='position')

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=tf.nn.relu,
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer(0.0005)):

            conv1 = slim.conv2d(self.img1, 8, [3, 3], padding='VALID', scope='conv1')
            conv2 = slim.conv2d(conv1, 64, [3, 3], padding='VALID', scope='conv2')
            flat1 = slim.flatten(conv2)
            fc1 = slim.fully_connected(flat1, 128, scope='fc1')
            self.predicted_positions = slim.fully_connected(fc1, 2, scope='fc')

    def loss_feed_dict(self, batch):
        img1_batch = img_as_float(batch['centercam'])
        img1_batch = img1_batch[:,:,:,0:3]
        pos1_batch = batch['finger_screen_pos']
        pos1_batch = pos1_batch[:,0:2]

        return {
            self.img1: img1_batch,
            self.positions: pos1_batch}

    def eval_feed_dict(self, obs_dict):
        #
        # this method must use numpy primitives
        #
        img1 = img_as_float(obs_dict['centercam'])
        img1 = img1[:, :, 0:3]
        img1 = np.expand_dims(img1, axis=0)

        return { self.img1: img1 }

    def get_output(self):
        return self.predicted_positions

    def get_loss(self):
        return tf.losses.mean_squared_error(self.positions, self.predicted_positions)

    def print_batch(self, batch):
        # print("batch keys {}".format(list(batch.keys())))
        pass

    def policy_initializer(self):
        pass
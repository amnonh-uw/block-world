import os
import tensorflow as tf
import numpy as np
from lib.networks.network import Network
from policies.base import DaggerPolicyBase

class DaggerPolicy:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def build_graph(self, dir_name):
        super().build_graph(dir_name)
        self.img1 = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='img1')
        self.img2 = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='img2')
        self.positions = tf.placeholder(tf.float32, shape=[None, 4], name='positions')

        inputs = {'img1': self.img1, 'img2': self.img2 }
        self.base_network = vgg16_siamese(inputs)
        self.predicted_positions = tf.contrib.layers.fully_connected(inputs=self.base_network.get_output("final"), num_outputs=4,
                                                                     activation_fn=None, scope='predict_positions')

    def train_sample_from_dict(self, sample_dict):
        #
        # This method must use tensorflow primitives
        #
        img1 = sample_dict['centercam']
        img1 = tf.slice(img1, [0,0,0], [-1,-1,3])
        img1 = tf.cast(img1, tf.float32)
        img1 = img1 - vgg16_siamese.mean()
        # img1 = self.tf_resize(img1, self.width, self.height)
        pos1 = tf.slice(sample_dict['finger_screen_pos'],[0], [2])
        pos2 = tf.slice(sample_dict['target_screen_pos'], [0], [2])

        positions = tf.concat((pos1, pos2), axis=0, name='concat_positions')

        return (img1, positions)

    def eval_sample_from_dict(self, sample_dict):
        #
        # this method must use numpy primitives
        #
        img1 = sample_dict['centercam']
        # img1 = self.im_resize(img1, self.width, self.height)
        img1 = np.asarray(img1)
        img1 = img1[:,:,0:3]
        img1 = img1 - vgg16_siamese.mean()
        img1 = np.expand_dims(img1, axis=0)
        # img2 = sample_dict['multichanneldepthcam']
        # img2 = img2.resize([224, 224], PIL.Image.BILINEAR)
        # img2 = np.asarray(img2, dtype=np.float32) / (256.0 * 256.0)
        # img2 = np.stack((img2, img2, img2), axis=2)
        # img2 = np.expand_dims(img2, axis=0)

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
        # positions = batch[0]
        # img1s = batch[1]
        # img2s = batch[2]
        # for i in range(positions.shape[0]):

            # we really want to replace the third value by a value from the depth map

            # img1 = img1s[i, :, :]
            # img1 = np.squeeze(img1)
            # DaggerPolicy.find_pixels('target', (34, 34, 34), img1)
            # DaggerPolicy.find_pixels('finger', (255, 40, 47), img1)
            # img2 = img2s[i, :, :]
            # img2 = np.squeeze(img2)
            # pos1 = positions[i, 0:3]
            # pos2 = positions[i, 3:6]
            # DaggerPolicy.depth_map_lookup(img2, img1, pos1)
            # DaggerPolicy.depth_map_lookup(img2, img1, pos2)
        pass

    def policy_initializer(self):
        self.base_network.load('vgg16.npy', tf.get_default_session(), ignore_missing=True)
        pass

class vgg16_siamese(Network):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        # first tower
        (self.feed('img1')
         .conv(3, 3, 64, 1, 1, name='conv1_1', c_i=3)
         .conv(3, 3, 64, 1, 1, name='conv1_2', c_i=64)
         .max_pool(2, 2, 2, 2, name='pool1')
         .conv(3, 3, 128, 1, 1, name='conv2_1', c_i=64)
         .conv(3, 3, 128, 1, 1, name='conv2_2', c_i=128)
         .max_pool(2, 2, 2, 2, name='pool2')
         .conv(3, 3, 256, 1, 1, name='conv3_1', c_i=128)
         .conv(3, 3, 256, 1, 1, name='conv3_2', c_i=256)
         .conv(3, 3, 256, 1, 1, name='conv3_3', c_i=256)
         .max_pool(2, 2, 2, 2, name='pool3')
         .conv(3, 3, 512, 1, 1, name='conv4_1', c_i=256)
         .conv(3, 3, 512, 1, 1, name='conv4_2', c_i=512)
         .conv(3, 3, 512, 1, 1, name='conv4_3', c_i=512)
         .max_pool(2, 2, 2, 2, name='pool4')
         .conv(3, 3, 512, 1, 1, name='conv5_1', c_i=512)
         .conv(3, 3, 512, 1, 1, name='conv5_2', c_i=512)
         .conv(3, 3, 512, 1, 1, name='conv5_3', c_i=512)
         .max_pool(2, 2, 2, 2, name='pool5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7'))

        # second tower
        (self.feed('img2')
         .conv(3, 3, 64, 1, 1, name='conv1_1_p', c_i=3)
         .conv(3, 3, 64, 1, 1, name='conv1_2_p', c_i=64)
         .max_pool(2, 2, 2, 2, name='pool1_p')
         .conv(3, 3, 128, 1, 1, name='conv2_1_p', c_i=64)
         .conv(3, 3, 128, 1, 1, name='conv2_2_p', c_i=128)
         .max_pool(2, 2, 2, 2, name='pool2_p')
         .conv(3, 3, 256, 1, 1, name='conv3_1_p', c_i=128)
         .conv(3, 3, 256, 1, 1, name='conv3_2_p', c_i=256)
         .conv(3, 3, 256, 1, 1, name='conv3_3_p', c_i=256)
         .max_pool(2, 2, 2, 2, name='pool3_p')
         .conv(3, 3, 512, 1, 1, name='conv4_1_p', c_i=256)
         .conv(3, 3, 512, 1, 1, name='conv4_2_p', c_i=512)
         .conv(3, 3, 512, 1, 1, name='conv4_3_p', c_i=512)
         .max_pool(2, 2, 2, 2, name='pool4_p')
         .conv(3, 3, 512, 1, 1, name='conv5_1_p', c_i=512)
         .conv(3, 3, 512, 1, 1, name='conv5_2_p', c_i=512)
         .conv(3, 3, 512, 1, 1, name='conv5_3_p', c_i=512)
         .max_pool(2, 2, 2, 2, name='pool5_p')
         .fc(4096, name='fc6_p')
         .fc(4096, name='fc7_p'))

          # combine towers
        (self.feed('fc7', 'fc7_p')
         .concat(1, name='combined_fc7')
         .fc(1024, name="dagger_fc8")
         .fc(1024, name="dagger_fc9")
         .fc(1024, name="dagger_fc10")
         .fc(1024, name="final"))

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

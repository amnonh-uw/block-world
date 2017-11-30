import tensorflow as tf
import numpy as np
from lib.networks.network import Network
from policies.base import DaggerPolicyBase
from PIL import ImageDraw
from skimage import img_as_float


#
#
# Learn to transform from image to x,y position in image of finger and target
# using an alexnet convolutional network
#
#



class DaggerPolicy(DaggerPolicyBase):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def build_graph(self, dir_name):
        super().build_graph(dir_name)


        self.left = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='left')
        self.right = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='right')
        self.worldxyz = tf.placeholder(tf.float32, shape=[None, 3], name='worldxyz')

        inputs = {'left': self.left, 'right': self.right }
        self.base_network = siamese_alexnet(inputs)
        self.predicted_worldxyz = self.base_network.get_output("final")

    def loss_feed_dict(self, batch):
        leftcam = batch['leftcam']
        leftcam = leftcam[:,:,:,0:3]
        leftcam = img_as_float(leftcam)

        rightcam = batch['rightcam']
        rightcam = rightcam[:, :, :, 0:3]
        rightcam = img_as_float(rightcam)

        worldxyz = batch['finger_pos'] - batch['target_pos']

        return {
            self.left: leftcam,
            self.right: rightcam,
            self.worldxyz: worldxyz}

    def eval_feed_dict(self, obs):
        leftcam = obs['leftcam']
        leftcam = np.expand_dims(leftcam, axis=0)
        leftcam = leftcam[:, :, :, 0:3]
        leftcam = img_as_float(leftcam)

        rightcam = obs['rightcam']
        rightcam = np.expand_dims(rightcam, axis=0)
        rightcam = rightcam[:, :, :, 0:3]
        rightcam = img_as_float(rightcam)

        return {
            self.left: leftcam,
            self.right: rightcam}

    def get_output(self):
        return self.predicted_worldxyz

    def get_loss(self):
        return tf.losses.mean_squared_error(self.worldxyz, self.predicted_worldxyz)

    @staticmethod
    def print_results(obs, output, step=None):
        worldxyz = obs['finger_pos'] - obs['target_pos']
        print("preidcted worldxyz {} actual worldxyz {}".format(output, worldxyz))

    def print_batch(self, batch):
        pass

    def policy_initializer(self):
        # self.base_network.load('vgg16.npy', tf.get_default_session(), ignore_missing=True)
        pass


class siamese_alexnet(Network):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('left')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1', c_i=3)
             .lrn(2, 2e-05, 0.75, name='norm1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2', c_i=96)
             .lrn(2, 2e-05, 0.75, name='norm2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .conv(3, 3, 384, 1, 1, name='conv3', c_i=256)
             .conv(3, 3, 384, 1, 1, group=2, name='conv4', c_i=384)
             .conv(3, 3, 256, 1, 1, group=2, name='conv5', c_i=384)
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5'))

        (self.feed('right')
         .conv(11, 11, 96, 4, 4, padding='VALID', name='r_conv1', c_i=3)
         .lrn(2, 2e-05, 0.75, name='r_norm1')
         .max_pool(3, 3, 2, 2, padding='VALID', name='r_pool1')
         .conv(5, 5, 256, 1, 1, group=2, name='r_conv2', c_i=96)
         .lrn(2, 2e-05, 0.75, name='r_norm2')
         .max_pool(3, 3, 2, 2, padding='VALID', name='r_pool2')
         .conv(3, 3, 384, 1, 1, name='r_conv3', c_i=256)
         .conv(3, 3, 384, 1, 1, group=2, name='r_conv4', c_i=384)
         .conv(3, 3, 256, 1, 1, group=2, name='r_conv5', c_i=384)
         .max_pool(3, 3, 2, 2, padding='VALID', name='r_pool5'))

        (self.feed('pool5', 'r_pool5')
                .concat(1, name='combined_fc6')
                .fc(512, name='fc6')
                .fc(256, name='fc7')
                .fc(128, name='fc9')
                .fc(3, relu=False, name='final'))

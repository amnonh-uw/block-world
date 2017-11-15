import tensorflow as tf
import numpy as np
from lib.networks.network import Network
from policies.base import DaggerPolicyBase
from PIL import ImageDraw


#
#
# Learn to transform from image to x,y position in image of finger and target
# using a vgg16 convolutional network
#
#



class DaggerPolicy(DaggerPolicyBase):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def build_graph(self, dir_name):
        super().build_graph(dir_name)

        self.img1 = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='img1')
        self.positions = tf.placeholder(tf.float32, shape=[None, 4], name='positions')

        inputs = {'img1': self.img1 }
        self.base_network = vgg16(inputs)
        self.predicted_positions = tf.contrib.layers.fully_connected(inputs=self.base_network.get_output("final"), num_outputs=4,
                                                                     activation_fn=None, scope='predict_positions')

    def train_sample_from_dict(self, sample_dict):
        #
        # This method must use tensorflow primitives
        #
        img1 = sample_dict['centercam']
        img1 = tf.slice(img1, [0,0,0], [-1,-1,3])
        img1 = tf.cast(img1, tf.float32)
        img1 = img1 - vgg16.mean()

        # img1 = self.tf_resize(img1, self.width, self.height)

        pos1 = tf.slice(sample_dict['finger_screen_pos'], [0], [2])
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
        img1 = img1 - vgg16.mean()
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

    @staticmethod
    def print_results(obs, action, step=None):
        pos1 = obs['finger_screen_pos']
        pos1 = pos1[0:2]
        act1 = action[0:2]
        pos2 = obs['target_screen_pos']
        pos2 = pos2[0:2]
        act2 = action[2:4]

        print("finger screen pos: {} act {} delta {}".format(pos1, act1, pos1-act1))
        print("target screen pos: {} act {} delta {}".format(pos2, act2, pos2-act2))

        if step != None:
            im = obs['centercam']
            draw = ImageDraw.Draw(im)
            l1 = (int(pos1[0], int(pos1[1]))
            l2 = (int(pos2[0], intpos2[1]))
            draw.rectangle((l1, l2))
            del draw

            im.save("enjoy" + str(step) + ".png")

    def print_batch(self, batch):
        pass

    def policy_initializer(self):
        self.base_network.load('vgg16.npy', tf.get_default_session(), ignore_missing=True)
        pass


class vgg16(Network):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
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
         .max_pool(2, 2, 2, 2, name='pool5'))

        (self.feed('pool5')
         .fc(1024, name="dagger_fc8")
         .fc(512, name="dagger_fc9")
         .fc(256, name="dagger_fc10")
         .fc(128, name="final"))

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

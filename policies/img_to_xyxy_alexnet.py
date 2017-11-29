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

        self.img = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='img')
        self.positions = tf.placeholder(tf.float32, shape=[None, 4], name='positions')

        inputs = {'img': self.img }
        self.base_network = alexnet(inputs)
        self.predicted_positions = tf.contrib.layers.fully_connected(inputs=self.base_network.get_output("fc7"), num_outputs=4,
                                                                     activation_fn=None, scope='predict_positions')



    def loss_feed_dict(self, batch):
        centercam = batch['centercam']
        centercam = centercam[:,:,:,0:3]
        centercam = img_as_float(centercam)

        pos1 = batch['finger_screen_pos']
        pos1 = pos1[:,0:2]
        pos2 = batch['target_screen_pos']
        pos2 = pos2[:,0:2]

        positions = np.concatenate((pos1, pos2), axis=1)

        return {
            self.img: centercam,
            self.positions: positions}

    def eval_feed_dict(self, obs):
        centercam = obs['centercam']
        centercam = np.expand_dims(centercam, axis=0)
        centercam = centercam[:,:,:,0:3]
        centercam = img_as_float(centercam)

        return { self.img: centercam } 

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
            l1 = (int(pos1[0]), int(pos1[1]))
            l2 = (int(pos2[0]), int(pos2[1]))
            draw.rectangle((l1, l2))
            del draw

            im.save("enjoy" + str(step) + ".png")

    def print_batch(self, batch):
        pass

    def policy_initializer(self):
        # self.base_network.load('vgg16.npy', tf.get_default_session(), ignore_missing=True)
        pass


class alexnet(Network):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('img')
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

        (self.feed('pool5')
                .fc(4096, name='fc6')
                .fc(4096, name='fc7'))

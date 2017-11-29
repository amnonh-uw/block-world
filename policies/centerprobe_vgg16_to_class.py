import tensorflow as tf
from lib.networks.network import Network
import numpy as np
from policies.base import DaggerPolicyBase
from skimage import img_as_float
from record_io import record_io
from hitpred_data import get_type_dict

class DaggerPolicy(DaggerPolicyBase):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def build_graph(self, dir_name):
        self.io = record_io(dir_name, get_type_dict())

        self.centercam = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='centercam')
        self.probe = tf.placeholder(tf.float32, shape=[None, 3])
        self.class_onehot = tf.placeholder(tf.float32, shape=[None, 3])

        inputs = {'img1': self.centercam,
                  'probe': self.probe }
        self.base_network = vgg16_with_probe(inputs)
        self.predicted_class_logits = self.base_network.get_output("final")

    def loss_feed_dict(self, batch):
        centercam = batch['centercam']
        centercam = centercam[:,:,:,0:3]
        centercam = img_as_float(centercam)

        probe = batch['probe_direction']

        no_collision = np.squeeze(batch['no_collision'])
        target_collison = np.squeeze(batch['target_collision'])
        object_collision = np.squeeze(batch['object_collision'])
        class_onehot = np.stack((no_collision, target_collison, object_collision), axis=-1)

        return {
            self.centercam: centercam,
            self.depthcam: depthcam,
            self.probe: probe,
            self.class_onehot: class_onehot}

    def get_output(self):
        return self.predicted_class_logits

    def get_loss(self):
        return tf.losses.softmax_cross_entropy(self.class_onehot, self.predicted_class_logits)

    def print_batch(self, batch):
        # print("batch keys {}".format(list(batch.keys())))
        pass

    def policy_initializer(self):
        pass

class vgg16_with_probe(Network):
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
         .fc(4096, name='fc6'))

          # combine tower and probe
        (self.feed('fc6', 'probe')
         .concat(1, name='combined_fc6')
         .fc(1024, name="dagger_fc8")
         .fc(512, name="dagger_fc9")
         .fc(256, name="dagger_fc10")
         .fc(3, name="final", relu=False))

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

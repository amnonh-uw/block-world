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

        self.leftcam = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='leftcam')
        self.rightcam = tf.placeholder(tf.float32, shape=[None, self.width, self.height, 3], name='rightcam')
        self.probe = tf.placeholder(tf.float32, shape=[None, 3])
        self.class_onehot = tf.placeholder(tf.float32, shape=[None, 3])
        self.distance = tf.placeholder(tf.float32, shape=[None, 1])

        inputs = {'img1': self.leftcam,
                  'img2': self.rightcam,
                  'probe': self.probe }
        self.base_network = vgg16_siamese_with_probe(inputs)
        self.predicted_class_logits = self.base_network.get_output("final_logits")
        self.predicted_distance = self.base_network.get_output("final_distance")

    def loss_feed_dict(self, batch):
        leftcam = batch['leftcam']
        leftcam = leftcam[:,:,:,0:3]
        leftcam = img_as_float(leftcam)

        rightcam = batch['rightcam']
        rightcam = rightcam[:,:,:,0:3]
        rightcam = img_as_float(rightcam)

        probe = batch['probe_direction']

        no_collision = np.squeeze(batch['no_collision'])
        target_collison = np.squeeze(batch['target_collision'])
        object_collision = np.squeeze(batch['object_collision'])
        class_onehot = np.stack((no_collision, target_collison, object_collision), axis=-1)

        distance = batch['collision_distance']

        return {
            self.leftcam: leftcam,
            self.rightcam: rightcam,
            self.probe: probe,
            self.class_onehot: class_onehot,
            self.distance: distance}

    def get_output(self):
        return self.predicted_class_logits

    def get_loss(self):
        class_loss =  tf.losses.softmax_cross_entropy(self.class_onehot, self.predicted_class_logits)
        tf.summary.scalar('class_loss', class_loss)
        distance_loss = tf.losses.mean_squared_error(self.distance, self.predicted_distance)
        tf.summary.scalar('distance_loss', distance_loss)

        return class_loss + distance_loss

    @staticmethod
    def print_results(obs, output, step=None, iteration=None):
        output = np.squeeze(output)
        predicted_class = np.argmax(output)
        real_class = None
        if obs['no_collision']:
            real_class = 0
        if obs['target_collision']:
            if real_class is not None:
                print("results: class already set!")
            real_class = 1
        if obs['object_collision']:
            if real_class is not None:
                print("results: class already set!")
            real_class = 2

        print("results: real_class {} precited_class {}".format(real_class, predicted_class))

    def print_batch(self, batch):
        # print("batch keys {}".format(list(batch.keys())))
        pass

    def policy_initializer(self):
        pass

class vgg16_siamese_with_probe(Network):
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
         .fc(4096, name='fc6_p'))

          # combine towers abnd probe
        (self.feed('fc6', 'fc6_p', 'probe')
         .concat(1, name='combined_fc6')
         .fc(1024, name="dagger_fc8")
         .fc(512, name="dagger_fc9")
         .fc(256, name="dagger_fc10")
         .fc(3, name="final_logits", relu=False))

        (self.feed('dagger_fc9')
         .fc(512, name="dagger_fc11")
         .fc(256, name="dagger_fc12")
         .fc(1, name="final_distance", relu=False))

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

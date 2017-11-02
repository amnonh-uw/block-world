import os
import tensorflow as tf
import numpy as np
from lib.networks.network import Network

class DaggerPolicy:
    width = 224
    height = 224
    def __init__(self, images, positions, action, num_actions, dir_name):
        self.images = images
        self.action = action
        self.positions = positions
        self.num_actions = num_actions

        # observations: c = np.array(self.block_env.centercam)
        #                   c = c[:, :, :-1]
        #                   d = np.array(self.block_env.multichanneldepthcam)

        centercam, depthcam = tf.split(images, (3, 1), axis=-1)
        img1 = centercam - vgg16_siamese.mean()
        depthcam = depthcam  / (256.0 * 256.0)
        img2 = tf.concat((depthcam, depthcam, depthcam), axis=3)

        inputs = {'img1': img1, 'img2': img2, 'positions' : positions}
        self.base_network = vgg16_siamese(inputs)
        self.logits = tf.layers.dense(inputs=self.base_network.get_output("dagger_fc9"), units=num_actions, activation=None, name='logits')

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            self.path = dir_name + "/"
            self.sample_counter = 0
        else:
            self.path = None

    def get_output(self):
        return tf.argmax(self.logits, axis=1)

    def get_loss(self):
        onehot_labels = tf.one_hot(self.action, self.num_actions)
        return tf.contrib.losses.softmax_cross_entropy(self.logits, onehot_labels)

    def policy_initializer(self):
        self.base_network.load('vgg16.npy', tf.get_default_session(), ignore_missing=True)
        pass

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save(self, obs, positions, action, phase = None, rollout=None, step=None):
        if self.path == None:
            return (obs, positions, action)
        else:
            sample_file = self.path + "s"
            if phase is not None:
                sample_file += "p" + str(phase)
            if rollout is not None:
                sample_file += "r" + str(rollout)
            if step is not None:
                sample_file += "s" + str(step)

            if step is None or rollout is None or phase is None:
                sample_file += "c" +  str(self.sample_counter)

            sample_file += ".tfrecord"
            self.sample_counter += 1
            writer = tf.python_io.TFRecordWriter(sample_file)

            positions_shape = np.array(positions.shape, np.uint32).tobytes()
            obs_shape = np.array(obs.shape, np.int32).tobytes()
            positions = positions.tobytes()
            obs = obs.astype(np.uint8)
            obs = obs.tobytes()
            # write label, shape, and image content to the TFRecord file
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'action': self.int64_feature(action),
                    'positions_shape' : self.bytes_feature(positions_shape),
                    'positions' : self.bytes_feature(positions),
                    'obs_shape': self.bytes_feature(obs_shape),
                    'obs': self.bytes_feature(obs)
                }))

            writer.write(example.SerializeToString())
            writer.close()
            return sample_file

    def get_dataset(self, samples):
        if self.path == None:
            return tf.contrib.data.Dataset.from_tensor_slices(samples)
        else:
            def tfrecord_map(serialized_example):
                features = tf.parse_single_example(
                    serialized_example,
                    features={
                        'action': tf.FixedLenFeature([], tf.int64),
                        'positions_shape' : tf.FixedLenFeature([], tf.string),
                        'positions' : tf.FixedLenFeature([], tf.string),
                        'obs_shape': tf.FixedLenFeature([], tf.string),
                        'obs': tf.FixedLenFeature([], tf.string)
                    })

                positions = tf.decode_raw(features['positions'], tf.float32)
                positions_shape = tf.decode_raw(features['positions_shape'], tf.int32)
                positions = tf.reshape(positions, positions_shape)
                obs = tf.decode_raw(features['obs'], tf.uint8)
                obs_shape = tf.decode_raw(features['obs_shape'], tf.int32)
                obs = tf.reshape(obs, obs_shape)
                action = features['action']

                return (obs, positions, action)

            dataset = tf.contrib.data.TFRecordDataset(samples)
            return dataset.map(tfrecord_map)


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

          # combine towers and positions
        (self.feed('fc7', 'fc7_p', 'positions')
         .concat(1, name='combined_fc7')
         .fc(256, name="dagger_fc8")
         .fc(256, name="dagger_fc9"))

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

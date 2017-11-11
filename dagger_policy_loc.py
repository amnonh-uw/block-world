import os
import tensorflow as tf
import numpy as np
from PIL import Image
from lib.networks.network import Network
import PIL

class DaggerPolicy:
    width = 224
    height = 224
    def __init__(self, dir_name):
        self.img1 = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='img1')
        self.position = tf.placeholder(tf.float32, shape=[None, 2], name='position')

        inputs = {'img1': self.img1 }
        self.base_network = vgg16(inputs)
        self.predicted_position = tf.contrib.layers.fully_connected(inputs=self.base_network.get_output("final"), num_outputs=2,
                                                                     activation_fn=None, scope='predict_positions')
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            self.path = dir_name + "/"
            self.sample_counter = 0
        else:
            self.path = None

    @staticmethod
    def train_sample_from_dict(sample_dict):
        #
        # This method must use tensorflow primitives
        #
        img1 = sample_dict['centercam']
        img1 = tf.slice(img1, [0,0,0], [-1,-1,3])
        img1 = tf.cast(img1, tf.float32)
        img1 = img1 - vgg16.mean()
        # img1 = tf.image.resize_images(img1, [224, 224])

        position = tf.slice(sample_dict['finger_screen_pos'],[0], [2])

        return (img1, position)

    @staticmethod
    def eval_sample_from_dict(sample_dict):
        #
        # this method must use numpy primitives
        #
        img1 = sample_dict['centercam']
        #img1 = img1.resize([224, 224], PIL.Image.BILINEAR)
        img1 = np.asarray(img1)
        img1 = img1[:,:,0:3]
        img1 = img1 - vgg16.mean()
        img1 = np.expand_dims(img1, axis=0)

        return (img1,)

    def loss_feed_dict(self, batch):
        return {
            self.img1: batch[0],
            self.position: batch[1]}

    def eval_feed_dict(self, obs_dict):
        sample = self.eval_sample_from_dict(obs_dict)
        return {
            self.img1: sample[0]
        }


    def get_output(self):
        return self.predicted_position

    def get_loss(self):
        return tf.losses.mean_squared_error(self.position, self.predicted_position)

    def print_batch(self, batch):

        pass

    def policy_initializer(self):
        self.base_network.load('vgg16.npy', tf.get_default_session(), ignore_missing=True)
        pass

    def invalid_sample(self, sample_dict):
        def invalid_pos(x, y, z, width, height):
            if x < 0.0 or x >= width:
                return True
            if y < 0.0 or y >= height:
                return True

            if abs(z) < 0.1:
                return True

            return False

        pos1 = sample_dict['finger_screen_pos']
        pos2 = sample_dict['target_screen_pos']
        width, height = sample_dict['centercam'].size

        if invalid_pos(pos1[0], pos1[1], pos1[2], width, height):
            print("rejecting sample finger {}".format(pos1))
            return True

        if invalid_pos(pos2[0], pos2[1], pos2[2], width, height):
            print("rejecting sample target {}".format(pos2))
            return True

        return False

    def save_sample(self, sample_dict, phase = None, rollout=None, step=None):
        if self.invalid_sample(sample_dict):
            return None
        def int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

        def bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def encode_image(key, obs, feature, dtype=np.uint8):
            img_array = np.asarray(obs[key], dtype=dtype)
            feature[key] = bytes_feature(img_array.tobytes())
            img_array_shape = np.array(img_array.shape, np.int32)
            feature[key + '_shape'] = bytes_feature(img_array_shape.tobytes())

        def encode_vector(key, obs, feature, dtype=np.float32):
            v = obs[key].astype(dtype)
            feature[key] = bytes_feature(v.tobytes())
            v_shape = np.array(v.shape, np.int32)
            feature[key + '_shape'] = bytes_feature(v_shape.tobytes())

        def encode_int64(key, obs, feature):
            feature[key] = int64_feature(obs[key])

        def encode_example(obs):
            feature = dict()
            encode_image('leftcam', obs, feature)
            encode_image('rightcam', obs, feature)
            encode_image('centercam', obs, feature)
            encode_image('multichanneldepthcam', obs, feature, dtype=np.int32)
            encode_image('normalcam', obs, feature)
            encode_vector('target_pos', obs, feature)
            encode_vector('target_rot', obs, feature)
            encode_vector('target_screen_pos', obs, feature)
            encode_vector('finger_pos', obs, feature)
            encode_vector('finger_rot', obs, feature)
            encode_vector('finger_screen_pos', obs, feature)
            encode_vector('action', obs, feature)

            return tf.train.Example(features=tf.train.Features(feature=feature))

        if self.path == None:
            return sample_dict
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
            example = encode_example(sample_dict)
            writer.write(example.SerializeToString())
            writer.close()
            return sample_file

    def get_dataset(self, samples):
        def tfrecord_map(serialized_example):
            def array_feature(key, feat):
                feat[key] = tf.FixedLenFeature([], tf.string)
                feat[key + '_shape'] = tf.FixedLenFeature([], tf.string)

            def int64_feature(key, feat):
                feat[key] = tf.FixedLenFeature([], tf.int64)

            def decode_array(key, sample_dict, features, dtype=tf.uint8):
                a = tf.decode_raw(features[key], dtype)
                a_shape = tf.decode_raw(features[key + '_shape'], tf.int32)
                sample_dict[key] = tf.reshape(a, a_shape, name='reshape_array_' + key)

            example_features = dict()

            array_feature("leftcam", example_features)
            array_feature('rightcam', example_features)
            array_feature('centercam', example_features)
            array_feature('multichanneldepthcam', example_features)
            array_feature('normalcam', example_features)
            array_feature('target_pos', example_features)
            array_feature('target_rot', example_features)
            array_feature('target_screen_pos', example_features)
            array_feature('finger_pos', example_features)
            array_feature('finger_rot', example_features)
            array_feature('finger_screen_pos', example_features)
            array_feature('action', example_features)

            features = tf.parse_single_example(serialized_example, features=example_features)

            sample_dict = dict()

            decode_array('leftcam', sample_dict, features)
            decode_array('rightcam', sample_dict, features)
            decode_array('centercam', sample_dict, features)
            decode_array('multichanneldepthcam', sample_dict, features, dtype=tf.int32)
            decode_array('normalcam', sample_dict, features)
            decode_array('target_pos', sample_dict, features, dtype=tf.float32)
            decode_array('target_rot', sample_dict, features, dtype=tf.float32)
            decode_array('target_screen_pos', sample_dict, features, dtype=tf.float32)
            decode_array('finger_pos', sample_dict, features, dtype=tf.float32)
            decode_array('finger_rot', sample_dict, features, dtype=tf.float32)
            decode_array('finger_screen_pos', sample_dict, features, dtype=tf.float32)
            decode_array('action', sample_dict, features, dtype=tf.float32)

            return self.train_sample_from_dict(sample_dict)

        if self.path == None:
            return tf.contrib.data.Dataset.from_tensor_slices(samples)
        else:
            dataset = tf.contrib.data.TFRecordDataset(samples)
            return dataset.map(tfrecord_map)

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
         .max_pool(2, 2, 2, 2, name='pool5')
         .fc(4096, name='fc6')
         .fc(4096, name='fc7'))

        (self.feed('fc7')
         .fc(1024, name="dagger_fc8")
         .fc(512, name="dagger_fc9")
         .fc(256, name="dagger_fc10")
         .fc(128, name="final"))

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

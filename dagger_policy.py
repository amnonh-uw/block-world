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
        self.positions = tf.placeholder(tf.float32, shape=[None, 6], name='screen_positions')
        self.action = tf.placeholder(tf.float32, name="action", shape=(None, 3))

        l1 = tf.contrib.layers.fully_connected(inputs=self.positions, num_outputs=256)
        l2 = tf.contrib.layers.fully_connected(inputs=l1, num_outputs=256)
        l3 = tf.contrib.layers.fully_connected(inputs=l2, num_outputs=256)
        l4 = tf.contrib.layers.fully_connected(inputs=l3, num_outputs=256, activation_fn=None)
        self.predicted_action = tf.contrib.layers.fully_connected(inputs=l4, num_outputs=3, activation_fn=None)

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            self.path = dir_name + "/"
            self.sample_counter = 0
        else:
            self.path = None

    @staticmethod
    def print_results(obs, action):
        pos1 = obs['finger_pos']
        pos2 = obs['target_pos']

        print('finger_pos {} target_pos {} action {}'.format(pos1, pos2, action))

    @staticmethod
    def train_sample_from_dict(sample_dict):
        #
        # This method must use tensorflow primitives
        #
        pos1 = sample_dict['finger_screen_pos']
        pos2 = sample_dict['target_screen_pos']

        positions = tf.concat((pos1, pos2), axis=0, name='concat_positions')
        action = sample_dict['action']

        return (positions, action)

    @staticmethod
    def eval_sample_from_dict(sample_dict):
        #
        # this method must use numpy primitives
        #

        pos1 = sample_dict['finger_screen_pos']
        pos2 = sample_dict['target_screen_pos']
        positions = np.concatenate((pos1, pos2), axis=0)
        positions = np.expand_dims(positions, axis=0)

        return (positions,)

    def loss_feed_dict(self, batch):
        return {
            self.positions: batch[0],
            self.action: batch[1]}

    def eval_feed_dict(self, obs_dict):
        sample = self.eval_sample_from_dict(obs_dict)
        return {
            self.positions: sample[0]
        }

    def print_batch(self, batch):
        pass

    def get_output(self):
        return self.predicted_action

    def get_loss(self):
        return tf.losses.mean_squared_error(self.action, self.predicted_action)

    def policy_initializer(self):
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
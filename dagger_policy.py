import os
import tensorflow as tf
import numpy as np
from PIL import Image
from lib.networks.network import Network
import PIL

class DaggerPolicy:
    width = 224
    height = 224
    def __init__(self,  num_actions, dir_name):
        self.img1 = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='img1')
        self.img2 = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='img2')
        self.positions = tf.placeholder(tf.float32, shape=[None, 4], name='screen_positions')
        self.action = tf.placeholder(tf.int32, name="action", shape=(None,))
        self.num_actions = num_actions

        inputs = {'img1': self.img1, 'img2': self.img2, 'positions' : self.positions}
        self.base_network = vgg16_siamese(inputs)
        self.logits = tf.layers.dense(inputs=self.base_network.get_output("dagger_fc9"), units=num_actions, activation=None, name='logits')

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            self.path = dir_name + "/"
            self.sample_counter = 0
        else:
            self.path = None

    @staticmethod
    def train_sample_from_dict(sample_dict):
        img1 = tf.slice(sample_dict['centercam'], [0,0,0], [-1,-1,3])
        img1 = tf.cast(img1, tf.float32)
        img1 = img1 - vgg16_siamese.mean()
        img2 = tf.cast(sample_dict['multichanneldepthcam'], tf.float32) / (256.0 * 256.0)
        img2 = tf.stack((img2, img2, img2), axis=2, name='stack_depth_channels')
        pos1 = tf.slice(sample_dict['finger_screen_pos'],[0], [2])
        pos2 = tf.slice(sample_dict['target_screen_pos'], [0], [2])
        positions = tf.concat((pos1, pos2), axis=0, name='concat_positions')
        action = sample_dict['action']

        return (img1, img2,positions, action)

    @staticmethod
    def eval_sample_from_dict(sample_dict):
        img1 = sample_dict['centercam']
        img1 = img1 - vgg16_siamese.mean()
        img2 = sample_dict['multichanneldepthcam'] / (256.0 * 256.0)
        img2 = np.concatenate((img2, img2, img2), axis=2)
        pos1 = sample_dict['finger_screen_pos'][0:2]
        pos2 = sample_dict['target_screen_pos'][0: 2]
        positions = np.concatenate((pos1, pos2), axis=0)

    def loss_feed_dict(self, batch):
        return {
            self.img1: batch[0],
            self.img2: batch[1],
            self.positions: batch[2],
            self.action: batch[3]}

    def eval_feed_dict(self, obs_dict):
        sample = self.eval_sample_from_dict(obs_dict)
        return {
            self.img1: sample[0],
            self.img2: sample[1],
            self.positions: sample[2]
        }

    def print_batch(self, batch):
        # print("printintg batch")
        # img1 = batch[0]
        # img2 = batch[1]
        # positions = batch[2]
        # actions = batch[3]
        # for i in range(actions.shape[0]):
        #     print("positions {} action {}".format(positions[i,:], actions[i]))
        #     c = img1[i, :, :, :] + vgg16_siamese.mean()
        #     c = c.astype(np.uint8)
        #     print("image from array shape {} dtype {}".format(c.shape, c.dtype))
        #     c_i = Image.fromarray(c)
        #     c_i.save('center' + str(i) + '.png')
        #     d = img2[i, :, :, 0]
        #     d = d * 256.0 * 256.0
        #     d = d.astype(np.uint16)
        #     print(d.shape)
        #     print(d.dtype)
        #     d_i = Image.fromarray(d)
        #     d_i.save('depth' + str(i) + '.tiff')
        pass

    def get_output(self):
        return tf.argmax(self.logits, axis=1)

    def get_loss(self):
        onehot_labels = tf.one_hot(self.action, self.num_actions)
        return tf.contrib.losses.softmax_cross_entropy(self.logits, onehot_labels)

    def policy_initializer(self):
        self.base_network.load('vgg16.npy', tf.get_default_session(), ignore_missing=True)

    def save(self, sample_dict, phase = None, rollout=None, step=None):
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
            encode_int64('action', obs, feature)

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
            int64_feature('action', example_features)

            print("Calling parse single example")

            features = tf.parse_single_example(serialized_example, features=example_features)

            print("parse single example done")

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
            sample_dict['action'] = features['action']

            return self.train_sample_from_dict(sample_dict)

        if self.path == None:
            return tf.contrib.data.Dataset.from_tensor_slices(samples)
        else:
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

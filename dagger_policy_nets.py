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
        self.img2 = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name='img2')
        self.positions = tf.placeholder(tf.float32, shape=[None, 4], name='positions')

        inputs = {'img1': self.img1, 'img2': self.img2 }
        self.base_network = vgg16_siamese(inputs)
        self.predicted_positions = tf.contrib.layers.fully_connected(inputs=self.base_network.get_output("final"), num_outputs=4,
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
        img1 = img1 - vgg16_siamese.mean()
        # img1 = tf.image.resize_images(img1, [224, 224])
        img2 = sample_dict['multichanneldepthcam']
        img2 = tf.cast(img2, tf.float32) / (256.0 * 256.0)
        img2 = tf.stack((img2, img2, img2), axis=2, name='stack_depth_channels')
        # img2 = tf.image.resize_images(img2, [224, 224])
        pos1 = tf.slice(sample_dict['finger_screen_pos'],[0], [2])
        pos2 = tf.slice(sample_dict['target_screen_pos'], [0], [2])

        positions = tf.concat((pos1, pos2), axis=0, name='concat_positions')

        return (img1, img2, positions)

    @staticmethod
    def eval_sample_from_dict(sample_dict):
        #
        # this method must use numpy primitives
        #
        img1 = sample_dict['centercam']
        #img1 = img1.resize([224, 224], PIL.Image.BILINEAR)
        img1 = np.asarray(img1)
        img1 = img1[:,:,0:3]
        img1 = img1 - vgg16_siamese.mean()
        img1 = np.expand_dims(img1, axis=0)
        img2 = sample_dict['multichanneldepthcam']
        # img2 = img2.resize([224, 224], PIL.Image.BILINEAR)
        img2 = np.asarray(img2, dtype=np.float32) / (256.0 * 256.0)
        img2 = np.stack((img2, img2, img2), axis=2)
        img2 = np.expand_dims(img2, axis=0)

        return (img1, img2)

    def loss_feed_dict(self, batch):
        return {
            self.img1: batch[0],
            self.img2: batch[1],
            self.positions: batch[2]}

    def eval_feed_dict(self, obs_dict):
        sample = self.eval_sample_from_dict(obs_dict)
        return {
            self.img1: sample[0],
            self.img2: sample[1]
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

    @staticmethod
    def find_pixels(s, rgb, img):
        for x in range(img.shape[0]):
            for y in range (img.shape[1]):
                if img[x,y,0] == rgb[0] and img[x,y,1] == rgb[1] and img[x,y,2] == rgb[2]:
                    print("found {} at {}".format(s, (x,y)))



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

    @staticmethod
    def depth_map_lookup(img_depth, img_rgb, pos):
        from math import ceil, floor
        def lookup(img_depth, img_rgb, x, y):
            w = img_depth.shape[0]
            h = img_depth.shape[1]
            if x < 0 or x >= w:
                print("x index {} out of range ".format(x))
                return 0
            if y < 0 or y >= img_depth.shape[1]:
                print("y index {} out of range ".format(y))
                return 0

            x = w - x - 1
            y = h - y - 1

            print(img_rgb.shape)

            print("img[{},{}]={} rgb {}".format(x, y, img_depth[x,y], img_rgb[x,y,:]))

            d = float(img_depth[x, y]) / (256.0 * 256.0)
            return d

        x = pos[0]
        y = pos[1]

        points = list()
        points.append((x, y, lookup(img_depth, img_rgb, ceil(x), ceil(y))))
        points.append((x, y, lookup(img_depth, img_rgb, floor(x), ceil(y))))
        points.append((x, y, lookup(img_depth, img_rgb, ceil(x), floor(y))))
        points.append((x, y, lookup(img_depth, img_rgb, floor(x), floor(y))))
        d = DaggerPolicy.bilinear_interpolation(x, y, points)

        print("depth {} vs {}".format(d, pos[2]))
        return d

    @staticmethod
    def bilinear_interpolation(x, y, points):
        points = sorted(points)  # order points by x, then by y
        (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

        if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
            raise ValueError('points do not form a rectangle')
        if not x1 <= x <= x2 or not y1 <= y <= y2:
            raise ValueError('(x, y) not within the rectangle')

        return (q11 * (x2 - x) * (y2 - y) +
                q21 * (x - x1) * (y2 - y) +
                q12 * (x2 - x) * (y - y1) +
                q22 * (x - x1) * (y - y1)
                ) / ((x2 - x1) * (y2 - y1) + 0.0)

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
         .fc(256, name="dagger_fc8")
         .fc(256, name="dagger_fc9")
         .fc(256, name="dagger_fc10")
         .fc(256, name="final"))

    @staticmethod
    def mean():
        # Pixel mean values (BGR order) as a (1, 1, 3) array
        # These are the values originally used for training VGG16
        return np.array([[103.939, 116.779, 123.68]])

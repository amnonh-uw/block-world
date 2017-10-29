import os
import numpy as np
import tensorflow as tf

class DaggerPolicy:
    def __init__(self, x, y, num_actions, dir_name):
        self.x = x
        self.y = y
        self.num_actions = num_actions

        h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu, name='h1')
        h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu, name='h2')
        h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu, name='h3')
        self.logits = tf.layers.dense(inputs=h3, units=num_actions, activation=None, name='logits')

        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
            self.path = dir_name + "/"
            self.sample_counter = 0
        else:
            self.path = None

    def get_output(self):
        return tf.argmax(self.logits, axis=1)

    def get_loss(self):
        onehot_labels = tf.one_hot(self.y, self.num_actions)
        return tf.contrib.losses.softmax_cross_entropy(self.logits, onehot_labels)

    def policy_initializer(self):
        pass

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def save(self, obs, action):
        if self.path == None:
            return (obs, action)
        else:
            sample_file = self.path + "s" + str(self.sample_counter) + ".tfrecord"
            self.sample_counter += 1
            writer = tf.python_io.TFRecordWriter(sample_file)

            shape = np.array(obs.shape, np.int32).tobytes()
            obs = obs.astype(np.float32)
            obs = obs.tobytes()
            # write label, shape, and image content to the TFRecord file
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'action': self.int64_feature(action),
                    'obs_shape': self.bytes_feature(shape),
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
                        'obs_shape': tf.FixedLenFeature([], tf.string),
                        'obs': tf.FixedLenFeature([], tf.string)
                    })

                obs = tf.decode_raw(features['obs'], tf.float32)
                obs_shape = tf.decode_raw(features['obs_shape'], tf.int32)
                obs = tf.reshape(obs, obs_shape)
                action = features['action']

                return (obs, action)

            dataset = tf.contrib.data.TFRecordDataset(samples)
            return dataset.map(tfrecord_map)

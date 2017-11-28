import os
import tensorflow as tf
import numpy as np
from PIL import Image

class record_io:
    def __init__(self, dir_name, type_dict):
        if dir_name != None:
            os.makedirs(dir_name, exist_ok=True)
            self.path = dir_name + "/"
            self.sample_counter = 0
        self.type_dict = type_dict

    def make_sample_path(self):
        if self.path == None:
            sample_path = None
        else:
            self.sample_counter += 1
            sample_path = self.path + "s" + str(self.sample_counter) + ".tfrecord"
        return sample_path

    def get_tf_dtype(self, key):
        tf_dtype = self.type_dict[key]
        if tf_dtype == "img8":
            return tf.uint8

        if tf_dtype == "img16":
            return tf.int32

        return tf_dtype

    def get_np_dtype(self, key):
        tf_dtype = self.type_dict[key]
        if tf_dtype == "img8":
            return(np.uint8)
        if tf_dtype == "img16":
            return(np.uint16)

        if tf_dtype == tf.float32:
            return(np.float32)

        if tf_dtype == tf.float32:
            return(np.float32)

        if tf_dtype == tf.int32:
            return np.int32

        raise ValueError("can't map {} for to numpy key {}".format(tf_dtype, key))

    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def encode_ndarray(self, a, key, feature):
        np_dtype = self.get_np_dtype(key)
        a = a.astype(np_dtype)
        feature[key] = record_io.bytes_feature(a.tobytes())
        img_array_shape = np.array(a.shape, np.int32)
        feature[key + '_shape'] = record_io.bytes_feature(img_array_shape.tobytes())

    @staticmethod
    def feature_ndarray(key, feat):
        feat[key] = tf.FixedLenFeature([], tf.string)
        feat[key + '_shape'] = tf.FixedLenFeature([], tf.string)

    def tf_decode_ndarray(self, key, sample, features):
        dtype = self.get_tf_dtype(key)
        a = tf.decode_raw(features[key], dtype)
        a_shape = tf.decode_raw(features[key + '_shape'], tf.int32)
        sample[key] = tf.reshape(a, a_shape, name='reshape_array_' + key)

    def np_decode_ndarray(self, key, sample, feature):
        np_dtype = self.get_np_dtype(key)
        a = np.fromstring(feature[key].bytes_list.value[0], dtype=np_dtype)
        a_shape = np.fromstring(feature[key + '_shape'].bytes_list.value[0], dtype=np.int32)
        sample[key] = np.reshape(a, a_shape)

    def encode_image(self, img, key, feature):
        img_array = np.asarray(img)
        return self.encode_ndarray(img_array, key, feature)

    def encode_sample(self, sample):
        feature = dict()

        for key in self.type_dict:
            if key not in sample:
                print(list(sample.keys()))
                raise ValueError("key {} is not in sample".format(key))
            field = sample[key]
            if isinstance(field, Image.Image):
                self.encode_image(field, key, feature)
            elif isinstance(field, np.ndarray):
                self.encode_ndarray(field, key, feature)
            elif isinstance(field, bool):
                self.encode_ndarray(np.full(1, field), key, feature)
            elif isinstance(field, float):
                self.encode_ndarray(np.full(1, field), key, feature)
            else:
                raise ValueError("dont know how to encode type " + str(type(field)))

        return tf.train.Example(features=tf.train.Features(feature=feature))

    def decode_sample(self, example):
        feature = example.features.feature
        sample = dict()
        for key in self.type_dict:
            if key not in feature:
                print(list(example.keys()))
                raise ValueError("key {} is not in example".format(key))
            self.np_decode_ndarray(key, sample, feature)

        return sample

    def save_sample(self, sample):
        sample_path = self.make_sample_path()
        writer = tf.python_io.TFRecordWriter(sample_path)
        example = self.encode_sample(sample)
        writer.write(example.SerializeToString())
        writer.close()
        return sample_path

    def load_sample(self, path):
        record_iterator = tf.python_io.tf_record_iterator(path=path)
        string_record = next(record_iterator, None)
        example = tf.train.Example()
        example.ParseFromString(string_record)
        sample = self.decode_sample(example)
        record_iterator.close()
        return sample

    def tfrecord_map(self, serialized_example):
        example_features = dict()

        for key in self.type_dict:
            record_io.feature_ndarray(key, example_features)

        features = tf.parse_single_example(serialized_example, features=example_features)

        new_sample = dict()
        for key in self.type_dict:
            self.tf_decode_ndarray(key, new_sample, features)

        return new_sample

    def get_dataset(self, sample_paths):
        def local_tfrecord_map(serialized_example):
            return self.tfrecord_map(serialized_example)

        dataset = tf.contrib.data.TFRecordDataset(sample_paths)
        return dataset.map(local_tfrecord_map)

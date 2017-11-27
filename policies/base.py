import tensorflow as tf
import numpy as np
from PIL import Image
from record_io import record_io
from dagger_data import get_type_dict

class DaggerPolicyBase:
    def build_graph(self, dir_name):
        self.io = record_io(dir_name, get_type_dict())

    def tf_resize(self, img, width, height):
        shape = tf.shape(img)
        if shape[0] != width or shape[1] != height:
            print("tf resizing image {} to {},{}".format(shape, width, height))
            img = tf.image.resize_images(img, [width, height])

        return img

    def im_resize(self, img, width, height):
        if width != img.width or height != img.height:
            print("tf resizing image from {},{}to {},{}".format(img.width, img.height, width, height))
            img = img.resize([width, height], Image.BILINEAR)

        return img

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

    def save_sample(self, sample_dict):
        if self.invalid_sample(sample_dict):
            return None

        return self.io.save_sample(sample_dict)

    def get_dataset(self, sample_paths):
        return self.io.get_dataset(sample_paths)

    def loss_feed_dict(self, batch):
       raise NotImplemented("loss_feed_dict")

    def eval_feed_dict(self, obs_dict):
        raise NotImplemented("eval_feed_dict")
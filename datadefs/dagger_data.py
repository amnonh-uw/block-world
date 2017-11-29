import tensorflow as tf
import sys

def get_type_dict():
    type_dict = dict()
    type_dict['leftcam'] = "img8"
    type_dict['rightcam'] = "img8"
    type_dict['centercam'] = "img8"
    type_dict['multichanneldepthcam'] = "img16"
    type_dict['normalcam'] = "img8"
    type_dict['target_pos'] = tf.float32
    type_dict['target_rot'] = tf.float32
    type_dict['target_screen_pos'] = tf.float32
    type_dict['finger_pos'] = tf.float32
    type_dict['finger_rot'] = tf.float32
    type_dict['finger_screen_pos'] = tf.float32
    type_dict['action'] = tf.float32

    return type_dict


from dump import dump
if __name__ == "__main__":
    dump(sys.argv[1], get_type_dict())

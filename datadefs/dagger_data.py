import tensorflow as tf

def get_type_dict():
    type_dict = dict()
    type_dict['leftcam'] = tf.uint8
    type_dict['rightcam'] = tf.uint8
    type_dict['centercam'] = tf.uint8
    type_dict['multichanneldepthcam'] = tf.int32
    type_dict['normalcam'] = tf.uint8
    type_dict['target_pos'] = tf.float32
    type_dict['target_rot'] = tf.float32
    type_dict['target_screen_pos'] = tf.float32
    type_dict['finger_pos'] = tf.float32
    type_dict['finger_rot'] = tf.float32
    type_dict['finger_screen_pos'] = tf.float32
    type_dict['action'] = tf.float32

    return type_dict
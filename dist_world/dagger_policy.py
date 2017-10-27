import tensorflow as tf

class DaggerPolicy:
    def __init__(self, x, y, num_actions):
        self.x = x
        self.y = y
        self.num_actions = num_actions

        h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu, name='h1')
        h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu, name='h2')
        h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu, name='h3')
        self.logits = tf.layers.dense(inputs=h3, units=num_actions, activation=None, name='logits')

    def get_output(self):
        return tf.argmax(self.logits, axis=1)

    def get_loss(self):
        onehot_labels = tf.one_hot(self.y, self.num_actions)
        return tf.contrib.losses.softmax_cross_entropy(self.logits, onehot_labels)

    def policy_initializer(self):
        pass

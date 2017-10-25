import tensorflow as tf

def policy(x, y, act_dim):
    h1 = tf.layers.dense(inputs=x, units=128, activation=tf.nn.relu, name='h1')
    h2 = tf.layers.dense(inputs=h1, units=64, activation=tf.nn.relu, name='h2')
    h3 = tf.layers.dense(inputs=h2, units=32, activation=tf.nn.relu, name='h3')
    logits = tf.layers.dense(inputs=h3, units=act_dim, activation=None, name='logits')

    onehot_labels = tf.one_hot(y, act_dim)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, onehot_labels)
    yhat = tf.argmax(logits, axis=1)

    return yhat, loss

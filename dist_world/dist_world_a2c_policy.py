import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

def ortho_init(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        #lasagne ortho init for tf
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: # assumes NHWC
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError

        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v # pick the one with the correct shape
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)
    return _ortho_init


def sample(logits):
    noise = tf.random_uniform(tf.shape(logits))
    return tf.argmax(logits - tf.log(-tf.log(noise)), 1)
    #
    # eps = 0.1
    # batch_size = tf.shape(logits)[0]
    # num_actions = tf.shape(logits)[1]
    # num_actions = tf.to_int64(num_actions)
    #
    # deterministic_actions = tf.argmax(logits, axis=1)
    # random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
    # chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
    # stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)
    #
    # return stochastic_actions

class Policy(object):
    def __init__(self, sess, ob_space, ac_space, nenv, nsteps, nstack, reuse=False):
        nbatch = nenv*nsteps
        ob_shape = (nbatch,) + ob_space.shape

        nact = ac_space.n
        X = tf.placeholder(ob_space.dtype, ob_shape) #obs

        def fc(x, scope, nh, act=tf.nn.relu, init_scale=1.0):
            with tf.variable_scope(scope):
                nin = x.get_shape()[1].value
                w = tf.get_variable("w", [nin, nh], initializer=ortho_init(init_scale))
                b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(0.0))
                z = tf.matmul(x, w) + b
                h = act(z)

        with tf.variable_scope("model", reuse=reuse):
            out = X
            for _ in range(4):
                out = layers.fully_connected(out, num_outputs=64, activation_fn=None, weights_initializer=ortho_init(0.1))
                out = tf.nn.relu(out)

            pi = layers.fully_connected(out, num_outputs=nact, scope = 'pi', activation_fn=None, weights_initializer=ortho_init(0.1))
            vf = layers.fully_connected(out, num_outputs=1, scope='v', activation_fn=None, weights_initializer=ortho_init(0.1))

        v0 = vf[:, 0]
        a0 = sample(pi)
        self.initial_state = [] #not stateful

        def step(ob, *_args, **_kwargs):
            a, v = sess.run([a0, v0], {X:ob})
            return a, v, [] #dummy state

        def value(ob, *_args, **_kwargs):
            return sess.run(v0, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
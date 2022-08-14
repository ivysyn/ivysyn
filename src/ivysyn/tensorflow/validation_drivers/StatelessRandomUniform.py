# StatelessRandomOpBase

import tensorflow as tf

dtype = tf.float64
shape = tf.constant(1, shape=[1], dtype=tf.int32)
seed = tf.constant(0, shape=[1], dtype=tf.int32)
tf.raw_ops.StatelessRandomUniform(shape=shape, seed=seed, dtype=dtype)

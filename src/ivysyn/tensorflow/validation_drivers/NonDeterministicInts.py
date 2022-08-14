# NonDeterministicIntsOp

import tensorflow as tf

dtype = tf.int64
shape = tf.constant(3, shape=[1], dtype=tf.int64)
tf.raw_ops.NonDeterministicInts(shape=shape, dtype=dtype)

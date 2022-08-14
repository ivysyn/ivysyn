# UnaryOp

import tensorflow as tf

input = tf.constant(0, shape=[5, 2], dtype=tf.complex64)
tf.raw_ops.Imag(input=input)

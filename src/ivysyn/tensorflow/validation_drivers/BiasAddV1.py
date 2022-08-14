# BiasOp

import tensorflow as tf

value = tf.constant(0.23656249, shape=[5,2], dtype=tf.float32)
bias = tf.constant([0,0], shape=[2], dtype=tf.float32)
tf.raw_ops.BiasAddV1(value=value, bias=bias)

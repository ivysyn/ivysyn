# TileOp

import tensorflow as tf

input = tf.constant(0.2, shape=[], dtype=tf.float32)
multiples = tf.constant([], shape=[0], dtype=tf.int32)
tf.raw_ops.Tile(input=input, multiples=multiples)

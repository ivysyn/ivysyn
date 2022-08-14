# ShapeOp

import tensorflow as tf

out_type = tf.int32
input = tf.constant(0.23656249, shape=[5,2], dtype=tf.float32)
tf.raw_ops.Shape(input=input, out_type=out_type)

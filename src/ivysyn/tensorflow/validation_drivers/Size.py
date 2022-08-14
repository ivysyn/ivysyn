# SizeOp

import tensorflow as tf

out_type = tf.int32
input = tf.constant(11, shape=[2,1], dtype=tf.float16)
tf.raw_ops.Size(input=input, out_type=out_type)

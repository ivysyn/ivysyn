# ShapeNOp

import tensorflow as tf

out_type = tf.int32
input = tf.constant(0.0443103798, shape=[32,10,10], dtype=tf.float32)
tf.raw_ops.ShapeN(input=input, out_type=out_type)

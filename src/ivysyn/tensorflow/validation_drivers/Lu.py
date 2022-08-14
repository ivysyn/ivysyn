# LuOp

import tensorflow as tf

output_idx_type = tf.int32
input = tf.constant(4, shape=[3, 3], dtype=tf.float32)
tf.raw_ops.Lu(input=input, output_idx_type=output_idx_type)

# UpperBoundOp

import tensorflow as tf

out_type = tf.int32
sorted_inputs = tf.constant(0.150729537, shape=[6,4], dtype=tf.float32)
values = tf.constant(0.909425139, shape=[6,3], dtype=tf.float32)
tf.raw_ops.UpperBound(sorted_inputs=sorted_inputs, values=values, out_type=out_type)

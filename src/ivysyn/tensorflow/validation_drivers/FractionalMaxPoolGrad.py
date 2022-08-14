# FractionalMaxPoolGradOp

import tensorflow as tf

overlapping = True
orig_input = tf.constant(.453409232, shape=[1,7,13,1], dtype=tf.float32)
orig_output = tf.constant(.219215487, shape=[1,4,7,1], dtype=tf.float32)
out_backprop = tf.constant(100, shape=[1,4,7,1], dtype=tf.float32)
row_pooling_sequence = tf.constant(0, shape=[5], dtype=tf.int64)
col_pooling_sequence = tf.constant(0, shape=[8], dtype=tf.int64)
tf.raw_ops.FractionalMaxPoolGrad(orig_input=orig_input, orig_output=orig_output, out_backprop=out_backprop, row_pooling_sequence=row_pooling_sequence, col_pooling_sequence=col_pooling_sequence, overlapping=overlapping)

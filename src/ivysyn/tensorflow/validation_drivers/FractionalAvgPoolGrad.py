# FractionalAvgPoolGradOp

import tensorflow as tf

overlapping = True
orig_input_tensor_shape = tf.constant(1, shape=[4], dtype=tf.int64)
out_backprop = tf.constant(100, shape=[1,4,7,1], dtype=tf.float64)
row_pooling_sequence = tf.constant(0, shape=[5], dtype=tf.int64)
col_pooling_sequence = tf.constant(0, shape=[8], dtype=tf.int64)
tf.raw_ops.FractionalAvgPoolGrad(orig_input_tensor_shape=orig_input_tensor_shape, out_backprop=out_backprop, row_pooling_sequence=row_pooling_sequence, col_pooling_sequence=col_pooling_sequence, overlapping=overlapping)

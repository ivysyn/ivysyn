# SparseFillEmptyRowsGradOp

import tensorflow as tf

reverse_index_map = tf.constant(0, shape=[6], dtype=tf.int64)
grad_values = tf.constant(1, shape=[8], dtype=tf.float64)
tf.raw_ops.SparseFillEmptyRowsGrad(reverse_index_map=reverse_index_map, grad_values=grad_values)

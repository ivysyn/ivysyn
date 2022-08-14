# SparseSliceGradOp

import tensorflow as tf

backprop_val_grad = tf.constant(1, shape=[4], dtype=tf.float32)
input_indices = tf.constant(0, shape=[14,2], dtype=tf.int64)
input_start = tf.constant([0,0], shape=[2], dtype=tf.int64)
output_indices = tf.constant(0, shape=[4,2], dtype=tf.int64)
tf.raw_ops.SparseSliceGrad(backprop_val_grad=backprop_val_grad, input_indices=input_indices, input_start=input_start, output_indices=output_indices)

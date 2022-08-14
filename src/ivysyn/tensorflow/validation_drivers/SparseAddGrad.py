# SparseAddGradOp

import tensorflow as tf

backprop_val_grad = tf.constant(1, shape=[19], dtype=tf.float32)
a_indices = tf.constant(0, shape=[11,2], dtype=tf.int64)
b_indices = tf.constant(1, shape=[10,2], dtype=tf.int64)
sum_indices = tf.constant(0, shape=[19,2], dtype=tf.int64)
tf.raw_ops.SparseAddGrad(backprop_val_grad=backprop_val_grad, a_indices=a_indices, b_indices=b_indices, sum_indices=sum_indices)

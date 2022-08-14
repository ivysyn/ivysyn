# SparseTensorDenseAddOp

import tensorflow as tf

a_indices = tf.constant(0, shape=[17,2], dtype=tf.int64)
a_values = tf.constant(1.69652808, shape=[17], dtype=tf.float32)
a_shape = tf.constant([6,12], shape=[2], dtype=tf.int64)
b = tf.constant(-0.223668531, shape=[6,12], dtype=tf.float32)
tf.raw_ops.SparseTensorDenseAdd(a_indices=a_indices, a_values=a_values, a_shape=a_shape, b=b)

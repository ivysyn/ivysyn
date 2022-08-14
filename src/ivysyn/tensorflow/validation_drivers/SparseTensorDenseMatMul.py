# SparseTensorDenseMatMulOp

import tensorflow as tf

adjoint_a = False
adjoint_b = True
a_indices = tf.constant(0, shape=[360,2], dtype=tf.int32)
a_values = tf.constant(0.0152700469, shape=[360], dtype=tf.float32)
a_shape = tf.constant([24,40], shape=[2], dtype=tf.int64)
b = tf.constant(1.34634328, shape=[2,40], dtype=tf.float32)
tf.raw_ops.SparseTensorDenseMatMul(a_indices=a_indices, a_values=a_values, a_shape=a_shape, b=b, adjoint_a=adjoint_a, adjoint_b=adjoint_b)

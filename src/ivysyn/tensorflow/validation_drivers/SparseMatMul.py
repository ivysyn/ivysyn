# SparseMatMulOp

import tensorflow as tf

transpose_a = True
transpose_b = False
a_is_sparse = True
b_is_sparse = False
a = tf.constant(1.34634328, shape=[2,40], dtype=tf.float32)
b = tf.constant(0.0244865417, shape=[40,12], dtype=tf.float32)
tf.raw_ops.SparseMatMul(a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b, a_is_sparse=a_is_sparse, b_is_sparse=b_is_sparse)

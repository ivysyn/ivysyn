# DenseToCSRSparseMatrixCPUOp

import tensorflow as tf

dense_input = tf.constant(1, shape=[5,11], dtype=tf.complex64)
indices = tf.constant(0, shape=[31,2], dtype=tf.int64)
tf.raw_ops.DenseToCSRSparseMatrix(dense_input=dense_input, indices=indices)

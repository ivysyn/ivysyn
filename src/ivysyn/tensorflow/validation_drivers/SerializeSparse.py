# SerializeSparseOp

import tensorflow as tf

out_type = tf.variant
sparse_indices = tf.constant(0, shape=[1,2], dtype=tf.int64)
sparse_values = tf.constant(0, shape=[1], dtype=tf.int64)
sparse_shape = tf.constant([1,1], shape=[2], dtype=tf.int64)
tf.raw_ops.SerializeSparse(sparse_indices=sparse_indices, sparse_values=sparse_values, sparse_shape=sparse_shape, out_type=out_type)

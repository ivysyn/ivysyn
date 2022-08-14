# SerializeManySparseOp

import tensorflow as tf

out_type = tf.variant
sparse_indices = tf.constant(0, shape=[10,3], dtype=tf.int64)
sparse_values = tf.constant(0, shape=[10], dtype=tf.float32)
sparse_shape = tf.constant([10,1,1], shape=[3], dtype=tf.int64)
tf.raw_ops.SerializeManySparse(sparse_indices=sparse_indices, sparse_values=sparse_values, sparse_shape=sparse_shape, out_type=out_type)

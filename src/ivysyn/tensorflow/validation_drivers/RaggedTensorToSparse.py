# RaggedTensorToSparseOp

import tensorflow as tf

rt_nested_splits = tf.constant([0,2,4], shape=[3], dtype=tf.int64)
rt_dense_values = tf.constant(0, shape=[5], dtype=tf.int64)
tf.raw_ops.RaggedTensorToSparse(rt_nested_splits=rt_nested_splits, rt_dense_values=rt_dense_values)

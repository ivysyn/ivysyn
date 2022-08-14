# RaggedTensorToTensorBaseOp

import tensorflow as tf

row_partition_types = ["ROW_SPLITS"]
shape = tf.constant(-1, shape=[], dtype=tf.int64)
values = tf.constant(True, shape=[5], dtype=tf.bool)
default_value = tf.constant(False, shape=[], dtype=tf.bool)
row_partition_tensors = tf.constant(0, shape=[4], dtype=tf.int64)
tf.raw_ops.RaggedTensorToTensor(shape=shape, values=values, default_value=default_value, row_partition_tensors=row_partition_tensors, row_partition_types=row_partition_types)

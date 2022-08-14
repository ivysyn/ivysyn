# TakeManySparseFromTensorsMapOp

import tensorflow as tf

dtype = tf.int32
container = ""
shared_name = "batch_join/AddManySparseToTensorsMap"
sparse_handles = tf.constant(0, shape=[1], dtype=tf.int64)
tf.raw_ops.TakeManySparseFromTensorsMap(sparse_handles=sparse_handles, dtype=dtype, container=container, shared_name=shared_name)

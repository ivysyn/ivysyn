# MapUnstageOp

import tensorflow as tf

dtypes = [tf.float32, tf.float32]
capacity = 0
memory_limit = 0
container = ""
shared_name = "MapStagingArea"
key = tf.constant(0, shape=[], dtype=tf.int64)
indices = tf.constant([0,1], shape=[2], dtype=tf.int32)
tf.raw_ops.OrderedMapUnstage(key=key, indices=indices, dtypes=dtypes, capacity=capacity, memory_limit=memory_limit, container=container, shared_name=shared_name)

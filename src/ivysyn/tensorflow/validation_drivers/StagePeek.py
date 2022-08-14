# StagePeekOp

import tensorflow as tf

dtypes = [tf.int32]
capacity = 0
memory_limit = 0
container = ""
shared_name = "StagingArea"
index = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.StagePeek(index=index, dtypes=dtypes, capacity=capacity, memory_limit=memory_limit, container=container, shared_name=shared_name)

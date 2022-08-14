# MapStageOp

import tensorflow as tf

dtypes = [tf.int32]
capacity = 3
memory_limit = 0
container = ""
shared_name = "MapStagingArea"
key = tf.constant(0, shape=[], dtype=tf.int64)
indices = [tf.constant(0, shape=[1], dtype=tf.int32)]
values = [tf.constant(0, shape=[], dtype=tf.int32)]
tf.raw_ops.OrderedMapStage(key=key, indices=indices, values=values, dtypes=dtypes,
                           capacity=capacity, memory_limit=memory_limit, container=container, shared_name=shared_name)

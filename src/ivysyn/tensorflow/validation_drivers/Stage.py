# StageOp

import tensorflow as tf

capacity = 0
memory_limit = 0
container = ""
shared_name = "StagingArea"
values = [tf.constant(-1, shape=[], dtype=tf.float32)]
tf.raw_ops.Stage(values=values, capacity=capacity, memory_limit=memory_limit,
                 container=container, shared_name=shared_name)

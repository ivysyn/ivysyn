# TensorScatterOp

import tensorflow as tf

tensor = tf.constant(0, shape=[7,5,2], dtype=tf.int32)
indices = tf.constant(0, shape=[15,2], dtype=tf.int32)
updates = tf.constant(1, shape=[15,2], dtype=tf.int32)
tf.raw_ops.TensorScatterSub(tensor=tensor, indices=indices, updates=updates)

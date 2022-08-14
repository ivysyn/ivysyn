# SnapshotOp

import tensorflow as tf

input = tf.constant(0, shape=[4], dtype=tf.int32)
tf.raw_ops.Snapshot(input=input)

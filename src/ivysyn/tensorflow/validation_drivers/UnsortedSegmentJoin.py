# UnsortedSegmentJoinOp

import tensorflow as tf

separator = ""
inputs = tf.constant("this", shape=[12], dtype=tf.string)
segment_ids = tf.constant(0, shape=[12], dtype=tf.int64)
num_segments = tf.constant(7, shape=[], dtype=tf.int64)
tf.raw_ops.UnsortedSegmentJoin(inputs=inputs, segment_ids=segment_ids, num_segments=num_segments, separator=separator)

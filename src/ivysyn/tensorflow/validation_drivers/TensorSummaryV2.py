# SummaryTensorOpV2

import tensorflow as tf

tag = tf.constant("foo", shape=[], dtype=tf.string)
tensor = tf.constant(True, shape=[6], dtype=tf.bool)
serialized_summary_metadata = tf.constant("[]", shape=[], dtype=tf.string)
tf.raw_ops.TensorSummaryV2(tag=tag, tensor=tensor, serialized_summary_metadata=serialized_summary_metadata)

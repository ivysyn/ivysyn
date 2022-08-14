# SummaryAudioOp

import tensorflow as tf

max_outputs = 3
tag = tf.constant("outer/inner", shape=[], dtype=tf.string)
tensor = tf.constant(1, shape=[5, 3, 4], dtype=tf.float32)
sample_rate = tf.constant(0.2, shape=[], dtype=tf.float32)
tf.raw_ops.AudioSummaryV2(tag=tag, tensor=tensor,
                          sample_rate=sample_rate, max_outputs=max_outputs)

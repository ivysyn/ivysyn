# NonMaxSuppressionWithOverlapsOp

import tensorflow as tf

overlaps = tf.constant(1, shape=[3,3], dtype=tf.float32)
scores = tf.constant([0.7,0.9,0.1], shape=[3], dtype=tf.float32)
max_output_size = tf.constant(3, shape=[], dtype=tf.int32)
overlap_threshold = tf.constant(0.6, shape=[], dtype=tf.float32)
score_threshold = tf.constant(0.4, shape=[], dtype=tf.float32)
tf.raw_ops.NonMaxSuppressionWithOverlaps(overlaps=overlaps, scores=scores, max_output_size=max_output_size, overlap_threshold=overlap_threshold, score_threshold=score_threshold)

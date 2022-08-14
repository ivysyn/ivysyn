# NonMaxSuppressionV4Op

import tensorflow as tf

pad_to_max_output_size = False
boxes = tf.constant(0, shape=[6, 4], dtype=tf.float16)
scores = tf.constant(0.899902344, shape=[6], dtype=tf.float16)
max_output_size = tf.constant(3, shape=[], dtype=tf.int32)
iou_threshold = tf.constant(0.5, shape=[], dtype=tf.float16)
score_threshold = tf.constant(5, shape=[], dtype=tf.float16)
tf.raw_ops.NonMaxSuppressionV4(boxes=boxes, scores=scores, max_output_size=max_output_size,
                               iou_threshold=iou_threshold, score_threshold=score_threshold, pad_to_max_output_size=pad_to_max_output_size)

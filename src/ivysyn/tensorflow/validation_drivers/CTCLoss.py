# CTCLossOp

import tensorflow as tf

preprocess_collapse_repeated = False
ctc_merge_repeated = True
ignore_longer_outputs_than_inputs = False
inputs = tf.constant(0.554979503, shape=[2,2,3], dtype=tf.float32)
labels_indices = tf.constant(0, shape=[4,2], dtype=tf.int64)
labels_values = tf.constant(0, shape=[4], dtype=tf.int32)
sequence_length = tf.constant([2,2], shape=[2], dtype=tf.int32)
tf.raw_ops.CTCLoss(inputs=inputs, labels_indices=labels_indices, labels_values=labels_values, sequence_length=sequence_length, preprocess_collapse_repeated=preprocess_collapse_repeated, ctc_merge_repeated=ctc_merge_repeated, ignore_longer_outputs_than_inputs=ignore_longer_outputs_than_inputs)

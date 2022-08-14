# CTCGreedyDecoderOp

import tensorflow as tf

merge_repeated = True
blank_index = -1
inputs = tf.constant(0, shape=[6,2,4], dtype=tf.float32)
sequence_length = tf.constant([4,5], shape=[2], dtype=tf.int32)
tf.raw_ops.CTCGreedyDecoder(inputs=inputs, sequence_length=sequence_length, merge_repeated=merge_repeated, blank_index=blank_index)

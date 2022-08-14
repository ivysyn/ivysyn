# StringNGramsOp

import tensorflow as tf

separator = "|"
ngram_widths = [3]
left_pad = "LP"
right_pad = "RP"
pad_width = -1
preserve_short_sequences = False
data = tf.constant("a", shape=[6], dtype=tf.string)
data_splits = tf.constant(0, shape=[4], dtype=tf.int64)
tf.raw_ops.StringNGrams(data=data, data_splits=data_splits, separator=separator, ngram_widths=ngram_widths, left_pad=left_pad, right_pad=right_pad, pad_width=pad_width, preserve_short_sequences=preserve_short_sequences)

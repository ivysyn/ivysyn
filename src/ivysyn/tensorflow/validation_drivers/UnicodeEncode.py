# UnicodeEncodeOp

import tensorflow as tf

output_encoding = "UTF-8"
errors = "replace"
replacement_char = 65533
input_values = tf.constant(72, shape=[5], dtype=tf.int32)
input_splits = tf.constant([0, 5], shape=[2], dtype=tf.int64)
tf.raw_ops.UnicodeEncode(input_values=input_values, input_splits=input_splits,
                         output_encoding=output_encoding, errors=errors, replacement_char=replacement_char)

# DecodePaddedRawOp

import tensorflow as tf

out_type = tf.uint8
little_endian = True
input_bytes = tf.constant("[string,longer_string]", shape=[2], dtype=tf.string)
fixed_length = tf.constant(8, shape=[], dtype=tf.int32)
tf.raw_ops.DecodePaddedRaw(input_bytes=input_bytes, fixed_length=fixed_length, out_type=out_type, little_endian=little_endian)

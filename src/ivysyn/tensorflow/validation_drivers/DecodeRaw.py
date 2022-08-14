# DecodeRawOp

import tensorflow as tf

out_type = tf.half
little_endian = True
bytes = tf.constant("[]", shape=[0], dtype=tf.string)
tf.raw_ops.DecodeRaw(bytes=bytes, out_type=out_type, little_endian=little_endian)

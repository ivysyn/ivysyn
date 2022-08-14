# DecodeCompressedOp

import tensorflow as tf

compression_type = ""
bytes = tf.constant("[abcd,efgh,aabbccdd]", shape=[3], dtype=tf.string)
tf.raw_ops.DecodeCompressed(bytes=bytes, compression_type=compression_type)

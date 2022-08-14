# DataFormatDimMapOp

import tensorflow as tf

src_format = "NHWC"
dst_format = "NCHW"
x = tf.constant(0, shape=[], dtype=tf.int32)
tf.raw_ops.DataFormatDimMap(x=x, src_format=src_format, dst_format=dst_format)

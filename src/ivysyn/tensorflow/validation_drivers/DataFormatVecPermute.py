# DataFormatVecPermuteOp

import tensorflow as tf

src_format = "1234"
dst_format = "4321"
x = tf.constant(0, shape=[2,2], dtype=tf.int32)
tf.raw_ops.DataFormatVecPermute(x=x, src_format=src_format, dst_format=dst_format)

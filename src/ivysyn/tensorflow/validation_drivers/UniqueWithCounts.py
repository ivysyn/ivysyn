# UniqueOp

import tensorflow as tf

out_idx = tf.int32
x = tf.constant([0,1], shape=[2], dtype=tf.int32)
tf.raw_ops.UniqueWithCounts(x=x, out_idx=out_idx)

# RankOp

import tensorflow as tf

input = tf.constant(111, shape=[12,2,1], dtype=tf.float64)
tf.raw_ops.Rank(input=input)

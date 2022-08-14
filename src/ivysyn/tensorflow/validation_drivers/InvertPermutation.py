# InvertPermutationOp

import tensorflow as tf

x = tf.constant([1,0,2], shape=[3], dtype=tf.int32)
tf.raw_ops.InvertPermutation(x=x)

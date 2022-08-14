# StringSplitV2Op

import tensorflow as tf

maxsplit = -1
input = tf.constant("1,2,3", shape=[4], dtype=tf.string)
sep = tf.constant(",", shape=[], dtype=tf.string)
tf.raw_ops.StringSplitV2(input=input, sep=sep, maxsplit=maxsplit)

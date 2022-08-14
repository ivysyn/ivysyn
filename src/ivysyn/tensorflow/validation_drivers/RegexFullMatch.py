# RegexFullMatchOp

import tensorflow as tf

input = tf.constant("[abc,1]", shape=[2], dtype=tf.string)
pattern = tf.constant("[]", shape=[], dtype=tf.string)
tf.raw_ops.RegexFullMatch(input=input, pattern=pattern)

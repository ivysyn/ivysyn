# StringStripOp

import tensorflow as tf

input = tf.constant("[abcd,efgh,aabbccdd]", shape=[3], dtype=tf.string)
tf.raw_ops.StringStrip(input=input)

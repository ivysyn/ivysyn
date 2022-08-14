# DecodeBase64Op

import tensorflow as tf

input = tf.constant("[abcd,efgh,aabbccdd]", shape=[3], dtype=tf.string)
tf.raw_ops.DecodeBase64(input=input)

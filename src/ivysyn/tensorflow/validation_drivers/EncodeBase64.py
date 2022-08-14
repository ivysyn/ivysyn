# EncodeBase64Op

import tensorflow as tf

pad = False
input = tf.constant("[abcd,efgh,aabbccdd]", shape=[3], dtype=tf.string)
tf.raw_ops.EncodeBase64(input=input, pad=pad)

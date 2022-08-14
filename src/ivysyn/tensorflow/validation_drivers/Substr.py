# SubstrOp

import tensorflow as tf

unit = "BYTE"
input = tf.constant("[hello,123,1+1]", shape=[3], dtype=tf.string)
pos = tf.constant(2, shape=[], dtype=tf.int32)
len = tf.constant(3, shape=[], dtype=tf.int32)
tf.raw_ops.Substr(input=input, pos=pos, len=len, unit=unit)

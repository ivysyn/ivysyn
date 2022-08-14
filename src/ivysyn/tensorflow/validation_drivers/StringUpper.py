# StringUpperOp

import tensorflow as tf

encoding = ""
input = tf.constant("[Pigs,on,The,Wing,aNimals]", shape=[2], dtype=tf.string)
tf.raw_ops.StringUpper(input=input, encoding=encoding)

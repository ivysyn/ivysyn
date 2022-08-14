# StringSplitOp

import tensorflow as tf

skip_empty = True
input = tf.constant("[pigs,on,the,wing,animals]", shape=[2], dtype=tf.string)
delimiter = tf.constant("[]", shape=[], dtype=tf.string)
tf.raw_ops.StringSplit(input=input, delimiter=delimiter, skip_empty=skip_empty)

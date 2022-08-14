# UnpackOp

import tensorflow as tf

num = 10
axis = 0
value = tf.constant(1, shape=[10,12,8,4,4], dtype=tf.float32)
tf.raw_ops.Unpack(value=value, num=num, axis=axis)
